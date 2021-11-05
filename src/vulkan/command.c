/*
 * This file is part of libplacebo.
 *
 * libplacebo is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 *
 * libplacebo is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with libplacebo. If not, see <http://www.gnu.org/licenses/>.
 */

#include "command.h"
#include "utils.h"

// returns VK_SUCCESS (completed), VK_TIMEOUT (not yet completed) or an error
static VkResult vk_cmd_poll(struct vk_ctx *vk, struct vk_cmd *cmd,
                            uint64_t timeout)
{
    return vk->WaitForFences(vk->dev, 1, &cmd->fence, false, timeout);
}

static void flush_callbacks(struct vk_ctx *vk)
{
    while (vk->num_pending_callbacks) {
        const struct vk_callback *cb = vk->pending_callbacks++;
        vk->num_pending_callbacks--;
        cb->run(cb->priv, cb->arg);
    }
}

static void vk_cmd_reset(struct vk_ctx *vk, struct vk_cmd *cmd)
{
    // Flush possible callbacks left over from a previous command still in the
    // process of being reset, whose callback triggered this command being
    // reset.
    flush_callbacks(vk);
    vk->pending_callbacks = cmd->callbacks.elem;
    vk->num_pending_callbacks = cmd->callbacks.num;
    flush_callbacks(vk);

    cmd->callbacks.num = 0;
    cmd->deps.num = 0;
    cmd->depstages.num = 0;
    cmd->depvalues.num = 0;
    cmd->sigs.num = 0;
    cmd->sigvalues.num = 0;
    cmd->objs.num = 0;

    // also make sure to reset vk->last_cmd in case this was the last command
    pl_mutex_lock(&vk->lock);
    if (vk->last_cmd == cmd)
        vk->last_cmd = NULL;
    pl_mutex_unlock(&vk->lock);
}

static void vk_cmd_destroy(struct vk_ctx *vk, struct vk_cmd *cmd)
{
    if (!cmd)
        return;

    vk_cmd_poll(vk, cmd, UINT64_MAX);
    vk_cmd_reset(vk, cmd);
    vk->DestroyFence(vk->dev, cmd->fence, PL_VK_ALLOC);
    vk->FreeCommandBuffers(vk->dev, cmd->pool->pool, 1, &cmd->buf);

    pl_free(cmd);
}

static struct vk_cmd *vk_cmd_create(struct vk_ctx *vk, struct vk_cmdpool *pool)
{
    struct vk_cmd *cmd = pl_zalloc_ptr(NULL, cmd);
    cmd->pool = pool;

    VkCommandBufferAllocateInfo ainfo = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
        .commandPool = pool->pool,
        .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
        .commandBufferCount = 1,
    };

    VK(vk->AllocateCommandBuffers(vk->dev, &ainfo, &cmd->buf));

    VkFenceCreateInfo finfo = {
        .sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO,
        .flags = VK_FENCE_CREATE_SIGNALED_BIT,
    };

    VK(vk->CreateFence(vk->dev, &finfo, PL_VK_ALLOC, &cmd->fence));
    PL_VK_NAME(FENCE, cmd->fence, "cmd");

    return cmd;

error:
    vk_cmd_destroy(vk, cmd);
    vk->failed = true;
    return NULL;
}

void vk_dev_callback(struct vk_ctx *vk, vk_cb callback,
                     const void *priv, const void *arg)
{
    pl_mutex_lock(&vk->lock);
    if (vk->last_cmd) {
        vk_cmd_callback(vk->last_cmd, callback, priv, arg);
    } else {
        // The device was already idle, so we can just immediately call it
        callback((void *) priv, (void *) arg);
    }
    pl_mutex_unlock(&vk->lock);
}

void vk_cmd_callback(struct vk_cmd *cmd, vk_cb callback,
                     const void *priv, const void *arg)
{
    PL_ARRAY_APPEND(cmd, cmd->callbacks, (struct vk_callback) {
        .run  = callback,
        .priv = (void *) priv,
        .arg  = (void *) arg,
    });
}

void vk_cmd_dep(struct vk_cmd *cmd, VkPipelineStageFlags stage, pl_vulkan_sem dep)
{
    assert(cmd->deps.num == cmd->depstages.num);
    assert(cmd->deps.num == cmd->depvalues.num);
    PL_ARRAY_APPEND(cmd, cmd->deps, dep.sem);
    PL_ARRAY_APPEND(cmd, cmd->depvalues, dep.value);
    PL_ARRAY_APPEND(cmd, cmd->depstages, stage);
}

void vk_cmd_obj(struct vk_cmd *cmd, const void *obj)
{
    PL_ARRAY_APPEND(cmd, cmd->objs, obj);
}

void vk_cmd_sig(struct vk_cmd *cmd, pl_vulkan_sem sig)
{
    assert(cmd->sigs.num == cmd->sigvalues.num);
    PL_ARRAY_APPEND(cmd, cmd->sigs, sig.sem);
    PL_ARRAY_APPEND(cmd, cmd->sigvalues, sig.value);
}

struct vk_signal {
    VkSemaphore semaphore;
    VkEvent event;
    enum vk_wait_type type; // last signal type
    VkQueue source;         // last signal source
};

struct vk_signal *vk_cmd_signal(struct vk_ctx *vk, struct vk_cmd *cmd,
                                VkPipelineStageFlags stage)
{
    struct vk_signal *sig = NULL;
    if (PL_ARRAY_POP(vk->signals, &sig))
        goto done;

    // no available signal => initialize a new one
    sig = pl_zalloc_ptr(NULL, sig);
    static const VkSemaphoreCreateInfo sinfo = {
        .sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO,
    };

    // We can skip creating the semaphores if there's only one queue
    if (vk->pools.num > 1 || vk->pools.elem[0]->num_queues > 1) {
        VK(vk->CreateSemaphore(vk->dev, &sinfo, PL_VK_ALLOC, &sig->semaphore));
        PL_VK_NAME(SEMAPHORE, sig->semaphore, "sig");
    }

    static const VkEventCreateInfo einfo = {
        .sType = VK_STRUCTURE_TYPE_EVENT_CREATE_INFO,
    };

    if (!vk->disable_events) {
        VkResult res = vk->CreateEvent(vk->dev, &einfo, PL_VK_ALLOC, &sig->event);
        if (res == VK_ERROR_FEATURE_NOT_PRESENT) {
            // Some vulkan implementations don't support VkEvents since they are
            // not part of the vulkan portable subset. So fail gracefully here.
            sig->event = VK_NULL_HANDLE;
            vk->disable_events = true;
            PL_INFO(vk, "VkEvent creation failed.. disabling events");
        } else {
            PL_VK_ASSERT(res, "Creating VkEvent");
            PL_VK_NAME(EVENT, sig->event, "sig");
        }
    }

done:
    // Signal both the semaphore, and the event if possible. (We will only
    // end up using one or the other)
    sig->type = VK_WAIT_NONE;
    sig->source = cmd->queue;
    if (sig->semaphore)
        vk_cmd_sig(cmd, (pl_vulkan_sem){ sig->semaphore });

    VkQueueFlags req = VK_QUEUE_GRAPHICS_BIT | VK_QUEUE_COMPUTE_BIT;
    if (sig->event && (cmd->pool->props.queueFlags & req)) {
        vk->CmdSetEvent(cmd->buf, sig->event, stage);
        sig->type = VK_WAIT_EVENT;
    }

    return sig;

error:
    vk_signal_destroy(vk, &sig);
    vk->failed = true;
    return NULL;
}

static bool unsignal_cmd(struct vk_cmd *cmd, VkSemaphore sem)
{
    if (!sem)
        return true;

    for (int n = 0; n < cmd->sigs.num; n++) {
        if (cmd->sigs.elem[n] == sem) {
            PL_ARRAY_REMOVE_AT(cmd->sigs, n);
            PL_ARRAY_REMOVE_AT(cmd->sigvalues, n);
            return true;
        }
    }

    return false;
}

// Attempts to remove a queued signal operation. Returns true if successful,
// i.e. the signal could be removed before it ever got fired.
static bool unsignal(struct vk_ctx *vk, struct vk_cmd *cmd, VkSemaphore sem)
{
    if (unsignal_cmd(cmd, sem))
        return true;

    // Attempt to remove it from any queued commands
    pl_mutex_lock(&vk->lock);
    for (int i = 0; i < vk->cmds_queued.num; i++) {
        if (unsignal_cmd(vk->cmds_queued.elem[i], sem)) {
            pl_mutex_unlock(&vk->lock);
            return true;
        }
    }
    pl_mutex_unlock(&vk->lock);

    return false;
}

static void release_signal(struct vk_ctx *vk, struct vk_signal *sig)
{
    // The semaphore never needs to be recreated, because it's either
    // unsignaled while still queued, or unsignaled as a result of a device
    // wait. But the event *may* need to be reset, so just always reset it.
    if (sig->event)
        vk->ResetEvent(vk->dev, sig->event);
    sig->source = NULL;

    pl_mutex_lock(&vk->lock);
    PL_ARRAY_APPEND(vk->alloc, vk->signals, sig);
    pl_mutex_unlock(&vk->lock);
}

enum vk_wait_type vk_cmd_wait(struct vk_ctx *vk, struct vk_cmd *cmd,
                              struct vk_signal **sigptr,
                              VkPipelineStageFlags stage,
                              VkEvent *out_event)
{
    struct vk_signal *sig = *sigptr;
    if (!sig)
        return VK_WAIT_NONE;

    if (sig->source == cmd->queue && unsignal(vk, cmd, sig->semaphore)) {
        // If we can remove the semaphore signal operation from the history and
        // pretend it never happened, then we get to use the more efficient
        // synchronization primitives. However, this requires that we're still
        // in the same VkQueue.
        if (sig->type == VK_WAIT_EVENT && out_event) {
            *out_event = sig->event;
        } else {
            sig->type = VK_WAIT_BARRIER;
        }
    } else {
        // Otherwise, we use the semaphore. (This also unsignals it as a result
        // of the command execution)
        vk_cmd_dep(cmd, stage, (pl_vulkan_sem){ sig->semaphore });
        sig->type = VK_WAIT_NONE;
    }

    // In either case, once the command completes, we can release the signal
    // resource back to the pool.
    vk_cmd_callback(cmd, (vk_cb) release_signal, vk, sig);
    *sigptr = NULL;
    return sig->type;
}

void vk_signal_destroy(struct vk_ctx *vk, struct vk_signal **sig)
{
    if (!*sig)
        return;

    vk->DestroySemaphore(vk->dev, (*sig)->semaphore, PL_VK_ALLOC);
    vk->DestroyEvent(vk->dev, (*sig)->event, PL_VK_ALLOC);
    pl_free(*sig);
    *sig = NULL;
}

struct vk_cmdpool *vk_cmdpool_create(struct vk_ctx *vk,
                                     VkDeviceQueueCreateInfo qinfo,
                                     VkQueueFamilyProperties props)
{
    struct vk_cmdpool *pool = pl_alloc_ptr(NULL, pool);
    *pool = (struct vk_cmdpool) {
        .props = props,
        .qf = qinfo.queueFamilyIndex,
        .queues = pl_calloc(pool, qinfo.queueCount, sizeof(VkQueue)),
        .num_queues = qinfo.queueCount,
    };

    for (int n = 0; n < pool->num_queues; n++)
        vk->GetDeviceQueue(vk->dev, pool->qf, n, &pool->queues[n]);

    VkCommandPoolCreateInfo cinfo = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
        .flags = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT |
                 VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
        .queueFamilyIndex = pool->qf,
    };

    VK(vk->CreateCommandPool(vk->dev, &cinfo, PL_VK_ALLOC, &pool->pool));

    return pool;

error:
    vk_cmdpool_destroy(vk, pool);
    vk->failed = true;
    return NULL;
}

void vk_cmdpool_destroy(struct vk_ctx *vk, struct vk_cmdpool *pool)
{
    if (!pool)
        return;

    for (int i = 0; i < pool->cmds.num; i++)
        vk_cmd_destroy(vk, pool->cmds.elem[i]);

    vk->DestroyCommandPool(vk->dev, pool->pool, PL_VK_ALLOC);
    pl_free(pool);
}

struct vk_cmd *vk_cmd_begin(struct vk_ctx *vk, struct vk_cmdpool *pool)
{
    // Garbage collect the cmdpool first, to increase the chances of getting
    // an already-available command buffer.
    vk_poll_commands(vk, 0);

    struct vk_cmd *cmd = NULL;
    pl_mutex_lock(&vk->lock);
    if (!PL_ARRAY_POP(pool->cmds, &cmd)) {
        cmd = vk_cmd_create(vk, pool);
        if (!cmd) {
            pl_mutex_unlock(&vk->lock);
            goto error;
        }
    }

    cmd->queue = pool->queues[pool->idx_queues];
    pl_mutex_unlock(&vk->lock);

    VkCommandBufferBeginInfo binfo = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
        .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
    };

    VK(vk->BeginCommandBuffer(cmd->buf, &binfo));
    return cmd;

error:
    // Something has to be seriously messed up if we get to this point
    vk_cmd_destroy(vk, cmd);
    vk->failed = true;
    return NULL;
}

bool vk_cmd_queue(struct vk_ctx *vk, struct vk_cmd **pcmd)
{
    struct vk_cmd *cmd = *pcmd;
    if (!cmd)
        return true;

    *pcmd = NULL;
    struct vk_cmdpool *pool = cmd->pool;

    VK(vk->EndCommandBuffer(cmd->buf));
    VK(vk->ResetFences(vk->dev, 1, &cmd->fence));

    pl_mutex_lock(&vk->lock);
    PL_ARRAY_APPEND(vk->alloc, vk->cmds_queued, cmd);
    vk->last_cmd = cmd;

    if (vk->cmds_queued.num >= PL_VK_MAX_QUEUED_CMDS) {
        PL_WARN(vk, "Exhausted the queued command limit.. forcing a flush now. "
                "Consider using pl_gpu_flush after submitting a batch of work?");
        vk_flush_commands(vk);
    }

    pl_mutex_unlock(&vk->lock);
    return true;

error:
    vk_cmd_reset(vk, cmd);
    pl_mutex_lock(&vk->lock);
    PL_ARRAY_APPEND(pool, pool->cmds, cmd);
    pl_mutex_unlock(&vk->lock);
    vk->failed = true;
    return false;
}

bool vk_poll_commands(struct vk_ctx *vk, uint64_t timeout)
{
    bool ret = false;
    pl_mutex_lock(&vk->lock);

    while (vk->cmds_pending.num) {
        struct vk_cmd *cmd = vk->cmds_pending.elem[0];
        struct vk_cmdpool *pool = cmd->pool;
        pl_mutex_unlock(&vk->lock); // don't hold mutex while blocking
        if (vk_cmd_poll(vk, cmd, timeout) == VK_TIMEOUT)
            return ret;
        pl_mutex_lock(&vk->lock);
        if (!vk->cmds_pending.num || vk->cmds_pending.elem[0] != cmd)
            continue; // another thread modified this state while blocking

        PL_TRACE(vk, "VkFence signalled: %p", (void *) cmd->fence);
        PL_ARRAY_REMOVE_AT(vk->cmds_pending, 0); // remove before callbacks
        vk_cmd_reset(vk, cmd);
        PL_ARRAY_APPEND(pool, pool->cmds, cmd);
        ret = true;

        // If we've successfully spent some time waiting for at least one
        // command, disable the timeout. This has the dual purpose of both
        // making sure we don't over-wait due to repeat timeout application,
        // but also makes sure we don't block on future commands if we've
        // already spend time waiting for one.
        timeout = 0;
    }

    pl_mutex_unlock(&vk->lock);
    return ret;
}

bool vk_flush_commands(struct vk_ctx *vk)
{
    return vk_flush_obj(vk, NULL);
}

bool vk_flush_obj(struct vk_ctx *vk, const void *obj)
{
    pl_mutex_lock(&vk->lock);

    // Count how many commands we want to flush
    int num_to_flush = vk->cmds_queued.num;
    if (obj) {
        num_to_flush = 0;
        for (int i = 0; i < vk->cmds_queued.num; i++) {
            struct vk_cmd *cmd = vk->cmds_queued.elem[i];
            for (int o = 0; o < cmd->objs.num; o++) {
                if (cmd->objs.elem[o] == obj) {
                    num_to_flush = i+1;
                    goto next_cmd;
                }
            }

next_cmd: ;
        }
    }

    if (!num_to_flush) {
        pl_mutex_unlock(&vk->lock);
        return true;
    }

    PL_TRACE(vk, "Flushing %d/%d queued commands",
             num_to_flush, vk->cmds_queued.num);

    bool ret = true;

    for (int i = 0; i < num_to_flush; i++) {
        struct vk_cmd *cmd = vk->cmds_queued.elem[i];
        struct vk_cmdpool *pool = cmd->pool;

        VkTimelineSemaphoreSubmitInfo tinfo = {
            .sType = VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO,
            .waitSemaphoreValueCount = cmd->depvalues.num,
            .pWaitSemaphoreValues = cmd->depvalues.elem,
            .signalSemaphoreValueCount = cmd->sigvalues.num,
            .pSignalSemaphoreValues = cmd->sigvalues.elem,
        };

        VkSubmitInfo sinfo = {
            .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
            .pNext = &tinfo,
            .commandBufferCount = 1,
            .pCommandBuffers = &cmd->buf,
            .waitSemaphoreCount = cmd->deps.num,
            .pWaitSemaphores = cmd->deps.elem,
            .pWaitDstStageMask = cmd->depstages.elem,
            .signalSemaphoreCount = cmd->sigs.num,
            .pSignalSemaphores = cmd->sigs.elem,
        };

        if (pl_msg_test(vk->log, PL_LOG_TRACE)) {
            PL_TRACE(vk, "Submitting command on queue %p (QF %d):",
                     (void *)cmd->queue, pool->qf);
            for (int n = 0; n < cmd->objs.num; n++)
                PL_TRACE(vk, "    uses object %p", cmd->objs.elem[n]);
            for (int n = 0; n < cmd->deps.num; n++) {
                PL_TRACE(vk, "    waits on semaphore %p = %"PRIu64,
                         (void *) cmd->deps.elem[n], cmd->depvalues.elem[n]);
            }
            for (int n = 0; n < cmd->sigs.num; n++) {
                PL_TRACE(vk, "    signals semaphore %p = %"PRIu64,
                        (void *) cmd->sigs.elem[n], cmd->sigvalues.elem[n]);
            }
            PL_TRACE(vk, "    signals fence %p", (void *) cmd->fence);
            if (cmd->callbacks.num)
                PL_TRACE(vk, "    signals %d callbacks", cmd->callbacks.num);
        }

        VK(vk->QueueSubmit(cmd->queue, 1, &sinfo, cmd->fence));
        PL_ARRAY_APPEND(vk->alloc, vk->cmds_pending, cmd);
        continue;

error:
        vk_cmd_reset(vk, cmd);
        PL_ARRAY_APPEND(pool, pool->cmds, cmd);
        vk->failed = true;
        ret = false;
    }

    // Move remaining commands back to index 0
    vk->cmds_queued.num -= num_to_flush;
    if (vk->cmds_queued.num) {
        memmove(vk->cmds_queued.elem, &vk->cmds_queued.elem[num_to_flush],
                vk->cmds_queued.num * sizeof(vk->cmds_queued.elem[0]));
    }

    // Wait until we've processed some of the now pending commands
    while (vk->cmds_pending.num > PL_VK_MAX_PENDING_CMDS) {
        pl_mutex_unlock(&vk->lock); // don't hold mutex while blocking
        vk_poll_commands(vk, UINT64_MAX);
        pl_mutex_lock(&vk->lock);
    }

    pl_mutex_unlock(&vk->lock);
    return ret;
}

void vk_rotate_queues(struct vk_ctx *vk)
{
    pl_mutex_lock(&vk->lock);

    // Rotate the queues to ensure good parallelism across frames
    for (int i = 0; i < vk->pools.num; i++) {
        struct vk_cmdpool *pool = vk->pools.elem[i];
        pool->idx_queues = (pool->idx_queues + 1) % pool->num_queues;
        PL_TRACE(vk, "QF %d: %d/%d", pool->qf, pool->idx_queues, pool->num_queues);
    }

    pl_mutex_unlock(&vk->lock);
}

void vk_wait_idle(struct vk_ctx *vk)
{
    vk_flush_commands(vk);
    while (vk_poll_commands(vk, UINT64_MAX)) ;
}
