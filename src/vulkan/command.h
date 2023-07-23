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

#pragma once
#include "common.h"

// Since lots of vulkan operations need to be done lazily once the affected
// resources are no longer in use, provide an abstraction for tracking these.
// In practice, these are only checked and run when submitting new commands, so
// the actual execution may be delayed by a frame.
typedef void (*vk_cb)(void *p, void *arg);

struct vk_callback {
    vk_cb run;
    void *priv;
    void *arg;
};

// Associate a callback with the completion of all currently pending commands.
// This will essentially run once the device is completely idle.
void vk_dev_callback(struct vk_ctx *vk, vk_cb callback,
                     const void *priv, const void *arg);

// Helper wrapper around command buffers that also track dependencies,
// callbacks and synchronization primitives
//
// Thread-safety: Unsafe
struct vk_cmd {
    struct vk_cmdpool *pool; // pool it was allocated from
    pl_vulkan_sem sync;      // pending execution, tied to lifetime of device
    VkQueue queue;           // the submission queue (for recording/pending)
    int qindex;              // the index of `queue` in `pool`
    VkCommandBuffer buf;     // the command buffer itself
    // Command dependencies and signals. Not owned by the vk_cmd.
    PL_ARRAY(VkSemaphoreSubmitInfo) deps;
    PL_ARRAY(VkSemaphoreSubmitInfo) sigs;
    // "Callbacks" to fire once a command completes. These are used for
    // multiple purposes, ranging from resource deallocation to fencing.
    PL_ARRAY(struct vk_callback) callbacks;
};

// Associate a callback with the completion of the current command. This
// function will be run once the command completes, or shortly thereafter.
void vk_cmd_callback(struct vk_cmd *cmd, vk_cb callback,
                     const void *priv, const void *arg);

// Associate a raw dependency for the current command. This semaphore must
// signal by the corresponding stage before the command may execute.
void vk_cmd_dep(struct vk_cmd *cmd, VkPipelineStageFlags2 stage, pl_vulkan_sem dep);

// Associate a raw signal with the current command. This semaphore will signal
// after the given stage completes.
void vk_cmd_sig(struct vk_cmd *cmd, VkPipelineStageFlags2 stage, pl_vulkan_sem sig);

// Synchronization scope
struct vk_sync_scope {
    pl_vulkan_sem sync;         // semaphore of last access
    VkQueue queue;              // source queue of last access
    VkPipelineStageFlags2 stage;// stage bitmask of last access
    VkAccessFlags2 access;      // access type bitmask
};

// Synchronization primitive
struct vk_sem {
    struct vk_sync_scope read, write;
};

// Updates the `vk_sem` state for a given access. If `is_trans` is set, this
// access is treated as a write (since it alters the resource's state).
//
// Returns a struct describing the previous access to a resource. A pipeline
// barrier is only required if the previous access scope is nonzero.
struct vk_sync_scope vk_sem_barrier(struct vk_cmd *cmd, struct vk_sem *sem,
                                    VkPipelineStageFlags2 stage,
                                    VkAccessFlags2 access, bool is_trans);

// Command pool / queue family hybrid abstraction
struct vk_cmdpool {
    struct vk_ctx *vk;
    VkQueueFamilyProperties props;
    int qf; // queue family index
    VkCommandPool pool;
    VkQueue *queues;
    int num_queues;
    int idx_queues;
    // Command buffers associated with this queue. These are available for
    // re-recording
    PL_ARRAY(struct vk_cmd *) cmds;
};

// Set up a vk_cmdpool corresponding to a queue family. `qnum` may be less than
// `props.queueCount`, to restrict the number of queues in this queue family.
struct vk_cmdpool *vk_cmdpool_create(struct vk_ctx *vk, int qf, int qnum,
                                     VkQueueFamilyProperties props);

void vk_cmdpool_destroy(struct vk_cmdpool *pool);

// Fetch a command buffer from a command pool and begin recording to it.
// Returns NULL on failure.
struct vk_cmd *vk_cmd_begin(struct vk_cmdpool *pool, pl_debug_tag debug_tag);

// Finish recording a command buffer and submit it for execution. This function
// takes over ownership of **cmd, and sets *cmd to NULL in doing so.
bool vk_cmd_submit(struct vk_cmd **cmd);

// Block until some commands complete executing. This is the only function that
// actually processes the callbacks. Will wait at most `timeout` nanoseconds
// for the completion of any command. The timeout may also be passed as 0, in
// which case this function will not block, but only poll for completed
// commands. Returns whether any forward progress was made.
//
// This does *not* flush any queued commands, forgetting to do so may result
// in infinite loops if waiting for the completion of callbacks that were
// never flushed!
bool vk_poll_commands(struct vk_ctx *vk, uint64_t timeout);

// Rotate through queues in each command pool. Call this once per frame, after
// submitting all of the command buffers for that frame. Calling this more
// often than that is possible but bad for performance.
void vk_rotate_queues(struct vk_ctx *vk);

// Wait until all commands are complete, i.e. the device is idle. This is
// basically equivalent to calling `vk_poll_commands` with a timeout of
// UINT64_MAX until it returns `false`.
void vk_wait_idle(struct vk_ctx *vk);
