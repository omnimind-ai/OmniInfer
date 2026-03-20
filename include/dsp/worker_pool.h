#ifndef WORKER_H
#define WORKER_H

/**=============================================================================

@file
   worker_pool.h

@brief
   Utility providing a thread worker pool for multi-threaded computer vision
   (or other compute) applications.

Copyright (c) 2019-2020 Qualcomm Technologies Incorporated.
All Rights Reserved. Qualcomm Proprietary and Confidential.

Export of this technology or software is regulated by the U.S.
Government. Diversion contrary to U.S. law prohibited.

All ideas, data and information contained in or disclosed by
this document are confidential and proprietary information of
Qualcomm Technologies Incorporated and all rights therein are expressly reserved.
By accepting this material the recipient agrees that this material
and the information contained therein are held in confidence and in
trust and will not be used, copied, reproduced in whole or in part,
nor its contents revealed in any manner to others without the express
written permission of Qualcomm Technologies Incorporated.

=============================================================================**/
//==============================================================================
// Defines
//==============================================================================
/// MACRO enables function to be visible in shared-library case.
#define WORKERPOOL_API __attribute__((visibility("default")))

//==============================================================================
// Include Files
//==============================================================================

#include <AEEStdDef.h>
#include <AEEStdErr.h>

#ifdef __cplusplus
extern "C" {
#endif

/*===========================================================================
    TYPEDEF
===========================================================================*/
/// signature of callbacks to be invoked by worker threads
typedef void (*worker_callback_t)(void *, int);

/// Typedef of worker_pool context
typedef void *worker_pool_context_t;

/// descriptor for requested callback
typedef struct {
  /// function pointer
  worker_callback_t fptr;
  /// data pointer
  void             *dptr;
} worker_pool_job_t;

/// opaque client view of synchronization token for job submitter and workers. Internals hidden in implementation.
typedef struct {
  /// opaque array to store synchronization token for job
  unsigned int dummy[8];  // large enough to hold a counter and a semaphore
} worker_synctoken_t __attribute__((aligned(8)));

/*===========================================================================
    CONSTANTS
===========================================================================*/
/// Maximum supported number of worker threads.

#define MAX_NUM_WORKERS 6
/// Number of workers
WORKERPOOL_API extern unsigned int num_workers;
/// Maximum number of hvx 128 bytes units available
WORKERPOOL_API extern unsigned int num_hvx128_contexts;

//==============================================================================
// Declarations
//==============================================================================

WORKERPOOL_API AEEResult worker_pool_init_ex(worker_pool_context_t *context, int stack_size, int n_workers, int allow_hmx);

//---------------------------------------------------------------------------
/// @brief
///   Initialize a worker pool. Should be called by each control thread that
///   requires its own worker pool.
///
///
/// @param *context
///   pointer to worker_pool_context_t variable.
///
/// @return
///   0 - success.
///   any other value - failure.
//---------------------------------------------------------------------------
WORKERPOOL_API AEEResult worker_pool_init(worker_pool_context_t *context);

//---------------------------------------------------------------------------
/// @brief
///   Initialize a worker pool with custom stack size of worker threads.
//    Should be called by each control thread that requires its own worker pool.
///
///
/// @param *context
///   pointer to worker_pool_context_t variable.
/// @param *stack_size
///   stack size of each worker thread.
///
/// @return
///   0 - success.
///   any other value - failure.
//---------------------------------------------------------------------------
WORKERPOOL_API AEEResult worker_pool_init_with_stack_size(worker_pool_context_t *context, int stack_size);

//---------------------------------------------------------------------------
/// @brief
///   Kill worker threads and release worker pool resources. Must be called
///   when pool owner no longer requires the pool.
///
///
/// @param *context
///   worker_pool_context_t.
///
//---------------------------------------------------------------------------
WORKERPOOL_API void worker_pool_deinit(worker_pool_context_t *context);

//---------------------------------------------------------------------------
/// @brief
///   Function to determine if there is an established worker pool available to
///   the calling thread. This is an optional call - if no pool is available
///   but attempted to be used, everything works seamlessly, in the client's
///   context (instead of worker context).
///
///
/// @param context
///   worker_pool_context_t.
///
/// @return
///   0 - no worker pool available.
///   any other value - worker pool available.
//---------------------------------------------------------------------------
WORKERPOOL_API AEEResult worker_pool_available(worker_pool_context_t context);

//---------------------------------------------------------------------------
/// @brief
///   Submit a job to the worker pool.
///
///
/// @param context
///   worker pool context where job is to be submitted.
///
/// @param job
///   callback function pointer and data.
///
/// @return
///   0 - success.
///   any other value - failure.
//---------------------------------------------------------------------------
WORKERPOOL_API AEEResult worker_pool_submit(worker_pool_context_t context, worker_pool_job_t job);

//---------------------------------------------------------------------------
/// @brief
///   Initialize a synchronization token for job submitter and workers to use.
///   Each worker callback must be given access to the token to release it, and
///   job submitter will wait for all jobs to release the token. Internals are
///   hidden from client.
///
///
/// @param token
///   pointer to the synctoken structure.
///
/// @param njobs
///   number of jobs that will be releasing the token
//---------------------------------------------------------------------------
WORKERPOOL_API void worker_pool_synctoken_init(worker_synctoken_t *token, unsigned int njobs);

//---------------------------------------------------------------------------
/// @brief
///   Needs to be called by the worker in the callback before exiting. The
///   token must be available to the callback via the data pointer given
///   to the callback during job submission.
///
///
/// @param token
///   pointer to the synctoken structure held by the job submitter
//---------------------------------------------------------------------------
WORKERPOOL_API void worker_pool_synctoken_jobdone(worker_synctoken_t *token);

//---------------------------------------------------------------------------
/// @brief
///   Job submitter calls this function after submitting all jobs to await
///   their completion.
///
///
/// @param token
///   pointer to the synctoken structure
//---------------------------------------------------------------------------
WORKERPOOL_API void worker_pool_synctoken_wait(worker_synctoken_t *token);

//---------------------------------------------------------------------------
/// @brief
///   Set the thread priority of the worker threads. Specified priority will
///   be applied to all threads in the default worker pool. The threads
///   that service boosted and background job requests will also be adjusted to be relative
///   to the new default thread priority.
///
///
/// @param context
///   worker pool context whose workers' priorities are to be changed.
///
/// @param prio
///   desired priority. 1 is the highest priority allowed. 255 is the lowest priority allowed.
///
/// @return
///   0 - success.
///   any other value - failure.
//---------------------------------------------------------------------------
WORKERPOOL_API AEEResult worker_pool_set_thread_priority(worker_pool_context_t context, unsigned int prio);

//---------------------------------------------------------------------------
/// @brief
///   Query the thread priority of the default worker threads. This will return
///   the current priority for one of the workers, which are all created
///   with the same priority. If a user callback has changed one or more worker threads independently,
///   there is no guarantee on which worker's priority is returned by this function.
///
///
/// @param context
///   worker pool context whose workers' priorities are asked.
///
/// @param prio
///   desired priority. 1 is the highest priority allowed. 255 is the lowest priority allowed.
///
/// @return
///   0 - success.
///   any other value - failure.
//---------------------------------------------------------------------------
WORKERPOOL_API AEEResult worker_pool_get_thread_priority(worker_pool_context_t context, unsigned int *prio);

//---------------------------------------------------------------------------
/// @brief
///   Utility inline to atomically increment a variable. Useful in
///   synchronizing jobs among worker threads, in cases where all
///   job-related info can be determined by the job number.
///
///
/// @param target
///   pointer to the variable being incremented
///
/// @return
///   the value after incrementing
//---------------------------------------------------------------------------
static inline unsigned int worker_pool_atomic_inc_return(unsigned int *target) {
  unsigned int result;
  __asm__ __volatile__(
    "1:     %0 = memw_locked(%2)\n"
    "       %0 = add(%0, #1)\n"
    "       memw_locked(%2, p0) = %0\n"
    "       if !p0 jump 1b\n"
    : "=&r"(result), "+m"(*target)
    : "r"(target)
    : "p0");
  return result;
}

//---------------------------------------------------------------------------
/// @brief
///   Utility inline to atomically decrement a variable.
///
///
/// @param target
///   pointer to the variable being incremented
///
/// @return
///   the value after decrementing
//---------------------------------------------------------------------------
static inline unsigned int worker_pool_atomic_dec_return(unsigned int *target) {
  unsigned int result;

  __asm__ __volatile__(
    "1:     %0 = memw_locked(%2)\n"
    "       %0 = add(%0, #-1)\n"
    "       memw_locked(%2, p0) = %0\n"
    "       if !p0 jump 1b\n"
    : "=&r"(result), "+m"(*target)
    : "r"(target)
    : "p0");
  return result;
}

#ifdef __cplusplus
}
#endif

#define EXPAND_COMMON_TASK_STATE_MEMBERS \
  worker_synctoken_t sync_ctx;           \
  unsigned int       task_id;            \
  int                n_tasks;            \
  int                n_tot_chunks;       \
  int                n_chunks_per_task;

// `sync_ctx` need to be initialized separately
#define INIT_COMMON_TASK_STATE_MEMBERS(state, n_tot_chunks, n_chunks_per_task)            \
  do {                                                                                    \
    state.task_id           = 0;                                                          \
    state.n_tasks           = (n_tot_chunks + n_chunks_per_task - 1) / n_chunks_per_task; \
    state.n_tot_chunks      = n_tot_chunks;                                               \
    state.n_chunks_per_task = n_chunks_per_task;                                          \
  } while (0)

#endif  // #ifndef WORKER_H
