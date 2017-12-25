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
struct vk_cmd {
    struct vk_cmdpool *pool; // pool it was allocated from
    VkQueue queue;           // the submission queue (for recording/pending)
    VkCommandBuffer buf;     // the command buffer itself
    VkFence fence;           // the fence guards cmd buffer reuse
    // The semaphores represent dependencies that need to complete before
    // this command can be executed. These are *not* owned by the vk_cmd
    VkSemaphore *deps;
    VkPipelineStageFlags *depstages;
    int num_deps;
    // The signals represent semaphores that fire once the command finishes
    // executing. These are also not owned by the vk_cmd
    VkSemaphore *sigs;
    int num_sigs;
    // Since VkFences are useless, we have to manually track "callbacks"
    // to fire once the VkFence completes. These are used for multiple purposes,
    // ranging from garbage collection (resource deallocation) to fencing.
    struct vk_callback *callbacks;
    int num_callbacks;
};

// Associate a callback with the completion of the current command. This
// bool will be set to `true` once the command completes, or shortly thereafter.
void vk_cmd_callback(struct vk_cmd *cmd, vk_cb callback,
                     const void *priv, const void *arg);

// Associate a raw dependency for the current command. This semaphore must
// signal by the corresponding stage before the command may execute.
void vk_cmd_dep(struct vk_cmd *cmd, VkSemaphore dep, VkPipelineStageFlags stage);

// Associate a raw signal with the current command. This semaphore will signal
// after the command completes.
void vk_cmd_sig(struct vk_cmd *cmd, VkSemaphore sig);

enum vk_wait_type {
    VK_WAIT_NONE,    // no synchronization needed
    VK_WAIT_BARRIER, // synchronization via pipeline barriers
    VK_WAIT_EVENT,   // synchronization via events
};

// Signal abstraction: represents an abstract synchronization mechanism.
// Internally, this may either resolve as a semaphore or an event depending
// on whether the appropriate conditions are met.
struct vk_signal;

// Generates a signal after the execution of all previous commands matching the
// given the pipeline stage. The signal is owned by the caller, and must be
// consumed eith vk_cmd_wait or released with vk_signal_cancel in order to
// free the resources.
struct vk_signal *vk_cmd_signal(struct vk_ctx *vk, struct vk_cmd *cmd,
                                VkPipelineStageFlags stage);

// Consumes a previously generated signal. This signal must fire by the
// indicated stage before the command can run. This function takes over
// ownership of the signal (and the signal will be released/reused
// automatically)
//
// The return type indicates what the caller needs to do:
//   VK_SIGNAL_NONE:    no further handling needed, caller can use TOP_OF_PIPE
//   VK_SIGNAL_BARRIER: caller must use pipeline barrier from last stage
//   VK_SIGNAL_EVENT:   caller must use VkEvent from last stage
//                      (never returned if out_event is NULL)
enum vk_wait_type vk_cmd_wait(struct vk_ctx *vk, struct vk_cmd *cmd,
                              struct vk_signal **sigptr,
                              VkPipelineStageFlags stage,
                              VkEvent *out_event);

// Destroys a currently pending signal, for example if the resource is no
// longer relevant.
void vk_signal_destroy(struct vk_ctx *vk, struct vk_signal **sig);

// Command pool / queue family hybrid abstraction
struct vk_cmdpool {
    VkQueueFamilyProperties props;
    int qf; // queue family index
    VkCommandPool pool;
    VkQueue *queues;
    int num_queues;
    int idx_queues;
    // Command buffers associated with this queue. These are available for
    // re-recording
    struct vk_cmd **cmds;
    int num_cmds;
};

// Set up a vk_cmdpool corresponding to a queue family.
struct vk_cmdpool *vk_cmdpool_create(struct vk_ctx *vk,
                                     VkDeviceQueueCreateInfo qinfo,
                                     VkQueueFamilyProperties props);

void vk_cmdpool_destroy(struct vk_ctx *vk, struct vk_cmdpool *pool);

// Fetch a command buffer from a command pool and begin recording to it.
// Returns NULL on failure.
struct vk_cmd *vk_cmd_begin(struct vk_ctx *vk, struct vk_cmdpool *pool);

// Finish recording a command buffer and queue it for execution. This function
// takes over ownership of *cmd, i.e. the caller should not touch it again.
void vk_cmd_queue(struct vk_ctx *vk, struct vk_cmd *cmd);

// Block until some commands complete executing. This is the only function that
// actually processes the callbacks. Will wait at most `timeout` nanoseconds
// for the completion of any command. The timeout may also be passed as 0, in
// which case this function will not block, but only poll for completed
// commands. Returns whether any forward progress was made.
bool vk_poll_commands(struct vk_ctx *vk, uint64_t timeout);

// Flush all currently queued commands. Call this once per frame, after
// submitting all of the command buffers for that frame. Calling this more
// often than that is possible but bad for performance.
// Returns whether successful. Failed commands will be implicitly dropped.
bool vk_flush_commands(struct vk_ctx *vk);

// Wait until all commands are complete, i.e. the device is idle. This is
// basically equivalent to calling `vk_poll_commands` with a timeout of
// UINT64_MAX until it returns `false`.
void vk_wait_idle(struct vk_ctx *vk);
