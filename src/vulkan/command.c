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
    if (vk->cmds_pending.num > 0) {
        struct vk_cmd *last_cmd = vk->cmds_pending.elem[vk->cmds_pending.num - 1];
        vk_cmd_callback(last_cmd, callback, priv, arg);
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

void vk_cmd_sig(struct vk_cmd *cmd, pl_vulkan_sem sig)
{
    assert(cmd->sigs.num == cmd->sigvalues.num);
    PL_ARRAY_APPEND(cmd, cmd->sigs, sig.sem);
    PL_ARRAY_APPEND(cmd, cmd->sigvalues, sig.value);
}

void vk_sem_uninit(struct vk_ctx *vk, struct vk_sem *sem)
{
    vk->DestroySemaphore(vk->dev, sem->semaphore, PL_VK_ALLOC);
    *sem = (struct vk_sem) {0};
}

bool vk_sem_init(struct vk_ctx *vk, struct vk_sem *sem, pl_debug_tag debug_tag)
{
    *sem = (struct vk_sem) {
        .write.stage = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
        .read.stage = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
    };

    static const VkSemaphoreTypeCreateInfo stinfo = {
        .sType = VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO,
        .semaphoreType  = VK_SEMAPHORE_TYPE_TIMELINE,
        .initialValue   = 0,
    };

    static const VkSemaphoreCreateInfo sinfo = {
        .sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO,
        .pNext = &stinfo,
    };

    // We always create a semaphore, so we can perform host waits on it
    VK(vk->CreateSemaphore(vk->dev, &sinfo, PL_VK_ALLOC, &sem->semaphore));
    PL_VK_NAME(SEMAPHORE, sem->semaphore, PL_DEF(debug_tag, "vk_sem"));
    return true;

error:
    vk->failed = true;
    return false;
}

struct vk_sync_scope vk_sem_barrier(struct vk_ctx *vk, struct vk_cmd *cmd,
                                    struct vk_sem *sem, VkPipelineStageFlags stage,
                                    VkAccessFlags access, bool is_trans)
{
    bool is_write = (access & vk_access_write) || is_trans;

    // Writes need to be synchronized against the last *read* (which is
    // transitively synchronized against the last write), reads only
    // need to be synchronized against the last write.
    struct vk_sync_scope last = sem->write;
    if (is_write && sem->read.access)
        last = sem->read;

    if (last.queue != cmd->queue) {
        if (!is_write && sem->read.queue == cmd->queue) {
            // No semaphore needed in this case because the implicit submission
            // order execution dependencies already transitively imply a wait
            // for the previous write
        } else if (last.queue) {
            vk_cmd_dep(cmd, stage, (pl_vulkan_sem) {
                .sem = sem->semaphore,
                .value = last.value,
            });
        }
        last.stage = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
        last.access = 0;
    }

    if (!is_write && sem->read.queue == cmd->queue &&
        (sem->read.stage & stage) == stage &&
        (sem->read.access & access) == access)
    {
        // A past pipeline barrier already covers this access transitively, so
        // we don't need to emit another pipeline barrier at all
        last.stage = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
        last.access = 0;
    }

    pl_assert(sem->read.value >= sem->write.value);
    uint64_t next_value = sem->read.value + 1;
    vk_cmd_sig(cmd, (pl_vulkan_sem) {
        .sem = sem->semaphore,
        .value = next_value,
    });

    if (is_write) {
        sem->write = (struct vk_sync_scope) {
            .value = next_value,
            .queue = cmd->queue,
            .stage = stage,
            .access = access,
        };

        sem->read = (struct vk_sync_scope) {
            .value = next_value,
            .queue = cmd->queue,
            // no stage or access scope, because no reads happened yet
        };
    } else if (sem->read.queue == cmd->queue) {
        // Coalesce multiple same-queue reads into a single access scope
        sem->read.value = next_value;
        sem->read.stage |= stage;
        sem->read.access |= access;
    } else {
        sem->read = (struct vk_sync_scope) {
            .value = next_value,
            .queue = cmd->queue,
            .stage = stage,
            .access = access,
        };
    }

    return last;
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

struct vk_cmd *vk_cmd_begin(struct vk_ctx *vk, struct vk_cmdpool *pool,
                            pl_debug_tag debug_tag)
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

    debug_tag = PL_DEF(debug_tag, "vk_cmd");
    PL_VK_NAME(COMMAND_BUFFER, cmd->buf, debug_tag);
    PL_VK_NAME(FENCE, cmd->fence, debug_tag);
    return cmd;

error:
    // Something has to be seriously messed up if we get to this point
    vk_cmd_destroy(vk, cmd);
    vk->failed = true;
    return NULL;
}

bool vk_cmd_submit(struct vk_ctx *vk, struct vk_cmd **pcmd)
{
    struct vk_cmd *cmd = *pcmd;
    if (!cmd)
        return true;

    *pcmd = NULL;
    struct vk_cmdpool *pool = cmd->pool;

    VK(vk->EndCommandBuffer(cmd->buf));
    VK(vk->ResetFences(vk->dev, 1, &cmd->fence));

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
        PL_TRACE(vk, "Submitting command %p on queue %p (QF %d):",
                 (void *) cmd->buf, (void *) cmd->queue, pool->qf);
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
    pl_mutex_lock(&vk->lock);
    PL_ARRAY_APPEND(vk->alloc, vk->cmds_pending, cmd);
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
    while (vk_poll_commands(vk, UINT64_MAX)) ;
}
