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
    return vkWaitForFences(vk->dev, 1, &cmd->fence, false, timeout);
}

static void vk_cmd_reset(struct vk_ctx *vk, struct vk_cmd *cmd)
{
    for (int i = 0; i < cmd->num_callbacks; i++) {
        struct vk_callback *cb = &cmd->callbacks[i];
        cb->run(cb->priv, cb->arg);
    }

    cmd->num_callbacks = 0;
    cmd->num_deps = 0;
    cmd->num_sigs = 0;

    // also make sure to reset vk->last_cmd in case this was the last command
    if (vk->last_cmd == cmd)
        vk->last_cmd = NULL;
}

static void vk_cmd_destroy(struct vk_ctx *vk, struct vk_cmd *cmd)
{
    if (!cmd)
        return;

    vk_cmd_poll(vk, cmd, UINT64_MAX);
    vk_cmd_reset(vk, cmd);
    vkDestroyFence(vk->dev, cmd->fence, VK_ALLOC);
    vkFreeCommandBuffers(vk->dev, cmd->pool->pool, 1, &cmd->buf);

    talloc_free(cmd);
}

static struct vk_cmd *vk_cmd_create(struct vk_ctx *vk, struct vk_cmdpool *pool)
{
    struct vk_cmd *cmd = talloc_zero(NULL, struct vk_cmd);
    cmd->pool = pool;

    VkCommandBufferAllocateInfo ainfo = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
        .commandPool = pool->pool,
        .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
        .commandBufferCount = 1,
    };

    VK(vkAllocateCommandBuffers(vk->dev, &ainfo, &cmd->buf));

    VkFenceCreateInfo finfo = {
        .sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO,
        .flags = VK_FENCE_CREATE_SIGNALED_BIT,
    };

    VK(vkCreateFence(vk->dev, &finfo, VK_ALLOC, &cmd->fence));

    return cmd;

error:
    vk_cmd_destroy(vk, cmd);
    return NULL;
}

void vk_dev_callback(struct vk_ctx *vk, vk_cb callback,
                     const void *priv, const void *arg)
{
    if (vk->last_cmd) {
        vk_cmd_callback(vk->last_cmd, callback, priv, arg);
    } else {
        // The device was already idle, so we can just immediately call it
        callback((void *) priv, (void *) arg);
    }
}

void vk_cmd_callback(struct vk_cmd *cmd, vk_cb callback,
                     const void *priv, const void *arg)
{
    TARRAY_APPEND(cmd, cmd->callbacks, cmd->num_callbacks, (struct vk_callback) {
        .run  = callback,
        .priv = (void *) priv,
        .arg  = (void *) arg,
    });
}

void vk_cmd_dep(struct vk_cmd *cmd, VkSemaphore dep, VkPipelineStageFlags stage)
{
    int idx = cmd->num_deps++;
    TARRAY_GROW(cmd, cmd->deps, idx);
    TARRAY_GROW(cmd, cmd->depstages, idx);
    cmd->deps[idx] = dep;
    cmd->depstages[idx] = stage;
}

void vk_cmd_sig(struct vk_cmd *cmd, VkSemaphore sig)
{
    TARRAY_APPEND(cmd, cmd->sigs, cmd->num_sigs, sig);
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
    if (TARRAY_POP(vk->signals, vk->num_signals, &sig))
        goto done;

    // no available signal => initialize a new one
    sig = talloc_zero(NULL, struct vk_signal);
    static const VkSemaphoreCreateInfo sinfo = {
        .sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO,
    };

    VK(vkCreateSemaphore(vk->dev, &sinfo, VK_ALLOC, &sig->semaphore));

    static const VkEventCreateInfo einfo = {
        .sType = VK_STRUCTURE_TYPE_EVENT_CREATE_INFO,
    };

    VK(vkCreateEvent(vk->dev, &einfo, VK_ALLOC, &sig->event));

done:
    // Signal both the semaphore, and the event if possible. (We will only
    // end up using one or the other)
    vk_cmd_sig(cmd, sig->semaphore);
    sig->type = VK_WAIT_NONE;
    sig->source = cmd->queue;

    VkQueueFlags req = VK_QUEUE_GRAPHICS_BIT | VK_QUEUE_COMPUTE_BIT;
    if (cmd->pool->props.queueFlags & req) {
        vkCmdSetEvent(cmd->buf, sig->event, stage);
        sig->type = VK_WAIT_EVENT;
    }

    return sig;

error:
    vk_signal_destroy(vk, &sig);
    return NULL;
}

static bool unsignal_cmd(struct vk_cmd *cmd, VkSemaphore sem)
{
    for (int n = 0; n < cmd->num_sigs; n++) {
        if (cmd->sigs[n] == sem) {
            TARRAY_REMOVE_AT(cmd->sigs, cmd->num_sigs, n);
            return true;
        }
    }

    return false;
}

// Attempts to remove a queued signal operation. Returns true if sucessful,
// i.e. the signal could be removed before it ever got fired.
static bool unsignal(struct vk_ctx *vk, struct vk_cmd *cmd, VkSemaphore sem)
{
    if (unsignal_cmd(cmd, sem))
        return true;

    // Attempt to remove it from any queued commands
    for (int i = 0; i < vk->num_cmds_queued; i++) {
        if (unsignal_cmd(vk->cmds_queued[i], sem))
            return true;
    }

    return false;
}

static void release_signal(struct vk_ctx *vk, struct vk_signal *sig)
{
    // The semaphore never needs to be recreated, because it's either
    // unsignaled while still queued, or unsignaled as a result of a device
    // wait. But the event *may* need to be reset, so just always reset it.
    vkResetEvent(vk->dev, sig->event);
    sig->source = NULL;
    TARRAY_APPEND(vk, vk->signals, vk->num_signals, sig);
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
        vk_cmd_dep(cmd, sig->semaphore, stage);
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

    vkDestroySemaphore(vk->dev, (*sig)->semaphore, VK_ALLOC);
    vkDestroyEvent(vk->dev, (*sig)->event, VK_ALLOC);
    talloc_free(*sig);
    *sig = NULL;
}

struct vk_cmdpool *vk_cmdpool_create(struct vk_ctx *vk,
                                     VkDeviceQueueCreateInfo qinfo,
                                     VkQueueFamilyProperties props)
{
    struct vk_cmdpool *pool = talloc_ptrtype(NULL, pool);
    *pool = (struct vk_cmdpool) {
        .props = props,
        .qf = qinfo.queueFamilyIndex,
        .queues = talloc_array(pool, VkQueue, qinfo.queueCount),
        .num_queues = qinfo.queueCount,
    };

    for (int n = 0; n < pool->num_queues; n++)
        vkGetDeviceQueue(vk->dev, pool->qf, n, &pool->queues[n]);

    VkCommandPoolCreateInfo cinfo = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
        .flags = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT |
                 VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
        .queueFamilyIndex = pool->qf,
    };

    VK(vkCreateCommandPool(vk->dev, &cinfo, VK_ALLOC, &pool->pool));

    return pool;

error:
    vk_cmdpool_destroy(vk, pool);
    return NULL;
}

void vk_cmdpool_destroy(struct vk_ctx *vk, struct vk_cmdpool *pool)
{
    if (!pool)
        return;

    for (int i = 0; i < pool->num_cmds; i++)
        vk_cmd_destroy(vk, pool->cmds[i]);

    vkDestroyCommandPool(vk->dev, pool->pool, VK_ALLOC);
    talloc_free(pool);
}

struct vk_cmd *vk_cmd_begin(struct vk_ctx *vk, struct vk_cmdpool *pool)
{
    // garbage collect the cmdpool first, to increase the chances of getting
    // an already-available command buffer
    vk_poll_commands(vk, 0);

    struct vk_cmd *cmd = NULL;
    if (TARRAY_POP(pool->cmds, pool->num_cmds, &cmd))
        goto done;

    // No free command buffers => allocate another one
    cmd = vk_cmd_create(vk, pool);
    if (!cmd)
        goto error;

done: ;

    VkCommandBufferBeginInfo binfo = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
        .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
    };

    VK(vkBeginCommandBuffer(cmd->buf, &binfo));

    cmd->queue = pool->queues[pool->idx_queues];
    return cmd;

error:
    // Something has to be seriously messed up if we get to this point
    vk_cmd_destroy(vk, cmd);
    return NULL;
}

void vk_cmd_queue(struct vk_ctx *vk, struct vk_cmd *cmd)
{
    struct vk_cmdpool *pool = cmd->pool;

    VK(vkEndCommandBuffer(cmd->buf));

    VK(vkResetFences(vk->dev, 1, &cmd->fence));
    TARRAY_APPEND(vk, vk->cmds_queued, vk->num_cmds_queued, cmd);
    vk->last_cmd = cmd;
    return;

error:
    vk_cmd_reset(vk, cmd);
    TARRAY_APPEND(pool, pool->cmds, pool->num_cmds, cmd);
}

bool vk_poll_commands(struct vk_ctx *vk, uint64_t timeout)
{
    bool ret = false;

    if (timeout && vk->num_cmds_queued)
        vk_flush_commands(vk);

    while (vk->num_cmds_pending > 0) {
        struct vk_cmd *cmd = vk->cmds_pending[0];
        struct vk_cmdpool *pool = cmd->pool;
        VkResult res = vk_cmd_poll(vk, cmd, timeout);
        if (res == VK_TIMEOUT)
            break;
        PL_TRACE(vk, "VkFence signalled: %p", (void *) cmd->fence);
        vk_cmd_reset(vk, cmd);
        TARRAY_REMOVE_AT(vk->cmds_pending, vk->num_cmds_pending, 0);
        TARRAY_APPEND(pool, pool->cmds, pool->num_cmds, cmd);
        ret = true;

        // If we've successfully spent some time waiting for at least one
        // command, disable the timeout. This has the dual purpose of both
        // making sure we don't over-wait due to repeat timeout applicaiton,
        // but also makes sure we don't block on future commands if we've
        // already spend time waiting for one.
        timeout = 0;
    }

    return ret;
}

bool vk_flush_commands(struct vk_ctx *vk)
{
    bool ret = true;

    for (int i = 0; i < vk->num_cmds_queued; i++) {
        struct vk_cmd *cmd = vk->cmds_queued[i];
        struct vk_cmdpool *pool = cmd->pool;

        VkSubmitInfo sinfo = {
            .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
            .commandBufferCount = 1,
            .pCommandBuffers = &cmd->buf,
            .waitSemaphoreCount = cmd->num_deps,
            .pWaitSemaphores = cmd->deps,
            .pWaitDstStageMask = cmd->depstages,
            .signalSemaphoreCount = cmd->num_sigs,
            .pSignalSemaphores = cmd->sigs,
        };

        if (pl_msg_test(vk->ctx, PL_LOG_TRACE)) {
            PL_TRACE(vk, "Submitting command on queue %p (QF %d):",
                     (void *)cmd->queue, pool->qf);
            for (int n = 0; n < cmd->num_deps; n++)
                PL_TRACE(vk, "    waits on semaphore %p", (void *) cmd->deps[n]);
            for (int n = 0; n < cmd->num_sigs; n++)
                PL_TRACE(vk, "    signals semaphore %p", (void *) cmd->sigs[n]);
            PL_TRACE(vk, "    signals fence %p", (void *) cmd->fence);
        }

        VK(vkQueueSubmit(cmd->queue, 1, &sinfo, cmd->fence));
        TARRAY_APPEND(vk, vk->cmds_pending, vk->num_cmds_pending, cmd);
        continue;

error:
        vk_cmd_reset(vk, cmd);
        TARRAY_APPEND(pool, pool->cmds, pool->num_cmds, cmd);
        ret = false;
    }

    vk->num_cmds_queued = 0;

    // Rotate the queues to ensure good parallelism across frames
    for (int i = 0; i < vk->num_pools; i++) {
        struct vk_cmdpool *pool = vk->pools[i];
        pool->idx_queues = (pool->idx_queues + 1) % pool->num_queues;
    }

    return ret;
}

void vk_wait_idle(struct vk_ctx *vk)
{
    while (vk_poll_commands(vk, UINT64_MAX)) ;
}
