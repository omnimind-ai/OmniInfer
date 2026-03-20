/*===========================================================================
    INCLUDE FILE
===========================================================================*/
#include "dsp/worker_pool.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "dsp/hmx_mgr.h"

#ifndef _DEBUG
#  define _DEBUG
#endif
#include "HAP_farf.h"

#ifdef __cplusplus
extern "C" {
#endif
#include "hexagon_protos.h"
#include "qurt.h"

void worker_pool_constructor(void) __attribute__((constructor));
void worker_pool_destructor(void) __attribute__((destructor));

#ifdef __cplusplus
}
#endif

/*===========================================================================
    DEFINE
===========================================================================*/
#define WORKER_THREAD_STACK_SZ  2 * 16384
#define WORKER_KILL_SIGNAL      31                     // signal to kill the worker threads
#define NUM_JOB_SLOTS           (MAX_NUM_WORKERS + 1)  // max queued jobs, slightly more than number of workers.
#define LOWEST_USABLE_QURT_PRIO 254

/*===========================================================================
    TYPEDEF
===========================================================================*/
// internal structure kept in thread-local storage per instance of worker pool
typedef struct {
  qurt_anysignal_t  empty_jobs;               // available job nodes
  qurt_anysignal_t  queued_jobs;              // jobs that are waiting for a worker
  qurt_mutex_t      empty_jobs_mutex;         // mutex for multiple threads trying to send a job
  qurt_mutex_t      queued_jobs_mutex;        // mutex for multiple threads trying to acquire a job
  unsigned int      job_queue_mask;           // mask for job queue nodes
  unsigned int      num_workers;              // number of workers in this pool
  worker_pool_job_t job[NUM_JOB_SLOTS];       // list of job descriptors
  qurt_thread_t     thread[MAX_NUM_WORKERS];  // thread ID's of the workers
  void             *stack[MAX_NUM_WORKERS];   // thread stack pointers
} worker_pool_t;

// internal structure containing OS primitives to sync caller with all its spawned jobs.
typedef union {
  worker_synctoken_t raw;

  struct {
    unsigned int atomic_countdown;
    unsigned int reserved;  // reserved to align next element to 8 bytes
    qurt_sem_t   sem;
  } sync;
} internal_synctoken_t;

// NOTE(hzx): This structure contains a pointer to worker pool and a worker index
typedef struct {
  worker_pool_t *pool;
  int            worker_index;
  int            allow_hmx;
} worker_info_t;

/*===========================================================================
    GLOBAL VARIABLES (per PD)
===========================================================================*/
// initialized in constructor
unsigned int num_workers         = 1;
unsigned int num_hvx128_contexts = 0;

/*===========================================================================
    STATIC VARIABLES
===========================================================================*/

static worker_pool_context_t static_context = NULL;

/*===========================================================================
    LOCAL FUNCTION
===========================================================================*/
// the main workloop for each of the worker threads.
static void worker_pool_main(void *context) {
  worker_info_t *info = (worker_info_t *) context;
  worker_pool_t *me   = info->pool;

  // local pointer to owning pool's context
  // worker_pool_t *me = (worker_pool_t *) context;

  // some local vars to reduce dereferencing inside loop
  qurt_anysignal_t *signal = &me->queued_jobs;
  unsigned int      mask   = me->job_queue_mask;
  qurt_mutex_t     *mutex  = &me->queued_jobs_mutex;

  if (info->allow_hmx) {
    hmx_manager_enable_execution();
  }

  while (1) {
    qurt_mutex_lock(mutex);  // mutex only allows 1 thread to wait on signal at a time. QuRT restriction.
    (void) qurt_anysignal_wait(signal, mask);                             // wait for a job
    unsigned int sig_rx = Q6_R_ct0_R(mask & qurt_anysignal_get(signal));  // count trailing 0's to choose flagged job
    if (sig_rx < NUM_JOB_SLOTS)                                           // if real job
    {
      worker_pool_job_t job = me->job[sig_rx];                            // local copy of job descriptor
      (void) qurt_anysignal_clear(signal, (1 << sig_rx));                 // clear the queued job signal
      (void) qurt_anysignal_set(&me->empty_jobs, (1 << sig_rx));          // send node back to empty list
      qurt_mutex_unlock(mutex);                                           // unlock the mutex
      job.fptr(job.dptr, info->worker_index);                             // issue the callback
    } else if (WORKER_KILL_SIGNAL == sig_rx) {
      // don't clear the kill signal, leave it for all the workers to see, and exit
      qurt_mutex_unlock(mutex);
      break;
    } else {
      FARF(HIGH, "Worker pool received invalid job %d", sig_rx);
      qurt_mutex_unlock(mutex);
    }
    // else ignore
  }

  if (info->allow_hmx) {
    hmx_manager_enable_execution();
  }
  qurt_thread_exit(0);
}

void worker_pool_constructor() {
  FARF(HIGH, "In worker_pool constructor");
  qurt_sysenv_max_hthreads_t num_threads;
  if (QURT_EOK != qurt_sysenv_get_max_hw_threads(&num_threads)) {
    num_workers = MAX_NUM_WORKERS;  // Couldn't get number of threads from QuRT, default to 4.
    FARF(HIGH, "Failed to get number of threads. Defaulting to %u", num_workers);
  } else {
    num_workers = num_threads.max_hthreads;
  }

  /* Verify that number of hw threads isn't greater than max supported number of hw threads.
       Max threads is used as a constant value for array size. */
  if (num_workers > MAX_NUM_WORKERS) {
    num_workers = MAX_NUM_WORKERS;
    FARF(HIGH, "Limiting number of threads to maximum supported value %u", num_workers);
  }

  num_hvx128_contexts = (qurt_hvx_get_units() >> 8) & 0xFF;

  /* initialize static worker_pool for clients who pass NULL as context.*/
  if (worker_pool_init(&static_context) != AEE_SUCCESS) {
    FARF(ERROR, "Could not initialize default worker pool");
  }
}

AEEResult worker_pool_init_ex(worker_pool_context_t *context, int stack_size, int num_workers, int allow_hmx) {
  int nErr = 0;

  if (stack_size <= 0) {
    FARF(ERROR, "Stack size can not be negative");
    return AEE_EBADPARM;
  }

  if (NULL == context) {
    FARF(ERROR, "NULL context passed to worker_pool_init().");
    return AEE_EBADPARM;
  }

  // Allocations
  size_t size = (stack_size * num_workers) + (sizeof(worker_pool_t)) + (sizeof(worker_info_t) * num_workers);

  unsigned char *mem_blob = (unsigned char *) malloc(size);
  if (!mem_blob) {
    FARF(ERROR, "Could not allocate memory for worker pool!!");
    return AEE_ENOMEMORY;
  }

  worker_pool_t *me   = (worker_pool_t *) (mem_blob + stack_size * num_workers);
  worker_info_t *info = (worker_info_t *) (mem_blob + stack_size * num_workers + sizeof(worker_pool_t));

  // name for the first worker, useful in debugging threads
  char name[19];
  snprintf(name, 12, "%08x:", (int) me);
  strcat(name, "th0");
  me->num_workers = num_workers;
  // initializations
  for (unsigned int i = 0; i < me->num_workers; i++) {
    me->stack[i]  = NULL;
    me->thread[i] = 0;
  }

  // initialize job queue
  qurt_anysignal_init(&(me->queued_jobs));
  qurt_anysignal_init(&(me->empty_jobs));
  qurt_mutex_init(&(me->empty_jobs_mutex));
  qurt_mutex_init(&(me->queued_jobs_mutex));
  me->job_queue_mask = (1 << NUM_JOB_SLOTS) - 1;  // set a bit for each job node, number of job nodes = num_workers + 1
  (void) qurt_anysignal_set(&(me->empty_jobs), me->job_queue_mask);  // fill the empty pool.
  me->job_queue_mask |= (1 << WORKER_KILL_SIGNAL);                   // add the kill signal to the mask.

  // launch the workers
  qurt_thread_attr_t attr;
  qurt_thread_attr_init(&attr);

  for (unsigned int i = 0; i < me->num_workers; i++) {
    // set up stack
    me->stack[i] = mem_blob;
    mem_blob += stack_size;
    qurt_thread_attr_set_stack_addr(&attr, me->stack[i]);
    qurt_thread_attr_set_stack_size(&attr, stack_size);

    // set up name
    qurt_thread_attr_set_name(&attr, name);
    name[11] = (name[11] + 1);
    // name threads context:th0, context:th1, .. (recycle at 9, but num threads should be less than that anyway)
    if (name[11] > '9') {
      name[11] = '0';
    }
    // set up priority - by default, match the creating thread's prio
    int prio = qurt_thread_get_priority(qurt_thread_get_id());

    // If loading thread has priority less than 64, load static worker pool with 64 priority.
    if (context == &static_context && prio < 64) {
      prio = 64;
    }

    if (prio < 1) {
      prio = 1;
    }
    if (prio > LOWEST_USABLE_QURT_PRIO) {
      prio = LOWEST_USABLE_QURT_PRIO;
    }

    qurt_thread_attr_set_priority(&attr, prio);

    // extra worker info containing worker index
    info[i].pool         = me;
    info[i].worker_index = i;
    info[i].allow_hmx    = allow_hmx;

    // launch
    nErr = qurt_thread_create(&(me->thread[i]), &attr, worker_pool_main, (void *) &info[i]);
    if (nErr) {
      FARF(ERROR, "Could not launch worker threads!");
      worker_pool_deinit((worker_pool_context_t *) &me);
      return AEE_EQURTTHREADCREATE;
    }
  }
  *context = (worker_pool_context_t *) me;
  return AEE_SUCCESS;
}

AEEResult worker_pool_init_with_stack_size(worker_pool_context_t *context, int stack_size) {
  return worker_pool_init_ex(context, stack_size, num_workers /* global */, 0);
}

AEEResult worker_pool_init(worker_pool_context_t *context) {
  return worker_pool_init_with_stack_size(context, WORKER_THREAD_STACK_SZ);
}

// clean up worker pool
void worker_pool_deinit(worker_pool_context_t *context) {
  worker_pool_t *me = (worker_pool_t *) *context;

  // if no worker pool exists, return error.
  if (NULL == me) {
    return;
  }

  // de-initializations
  (void) qurt_anysignal_set(&(me->empty_jobs), (1 << WORKER_KILL_SIGNAL));   // notify to stop new jobs.
  (void) qurt_anysignal_set(&(me->queued_jobs), (1 << WORKER_KILL_SIGNAL));  // kill worker pool.
  for (unsigned int i = 0; i < me->num_workers; i++)                         // wait for workers to die
  {
    if (me->thread[i]) {
      int status;
      (void) qurt_thread_join(me->thread[i], &status);
    }
  }

  // release resources
  qurt_mutex_destroy(&(me->empty_jobs_mutex));
  qurt_mutex_destroy(&(me->queued_jobs_mutex));
  qurt_anysignal_destroy(&(me->queued_jobs));
  qurt_anysignal_destroy(&(me->empty_jobs));
  // free allocated memory (were allocated as a single buffer starting at stack[0])
  if (me->stack[0]) {
    free(me->stack[0]);
  }
  // Assign NULL to freed context so that further refence to it fails.
  *context = NULL;
}

// submit a job to the pool.
AEEResult worker_pool_submit(worker_pool_context_t context, worker_pool_job_t job) {
  worker_pool_t *me = (worker_pool_t *) context;

  // if NULL is passed as worker_pool_context, try to use default static worker_pool
  if (NULL == me) {
    if (static_context == NULL) {
      FARF(HIGH, "No default static worker pool found");
      return AEE_ERESOURCENOTFOUND;
    }
    FARF(MEDIUM, "Using default static worker pool");
    me = (worker_pool_t *) static_context;
  }

  // if a worker thread tries to submit a job, call it in-context to avoid recursion deadlock.
  unsigned int  i;
  qurt_thread_t id = qurt_thread_get_id();
  for (i = 0; i < me->num_workers; i++) {
    if (id == me->thread[i]) {
      job.fptr(job.dptr, i);  // issue the callback in caller's context
      return AEE_SUCCESS;
    }
  }

  // local vars to reduce dereferencing
  qurt_mutex_t     *mutex  = &me->empty_jobs_mutex;
  qurt_anysignal_t *signal = &me->empty_jobs;
  unsigned int      mask   = me->job_queue_mask;

  qurt_mutex_lock(mutex);                    // lock empty queue
  (void) qurt_anysignal_wait(signal, mask);  // wait for an empty job node
  unsigned int bitfield = qurt_anysignal_get(signal);

  // check if pool is being killed and return early
  if (bitfield & (1 << WORKER_KILL_SIGNAL)) {
    qurt_mutex_unlock(mutex);
    return AEE_ENOMORE;
  }

  // send the job to the queue.
  unsigned int sig_rx = Q6_R_ct0_R(mask & bitfield);           // count trailing 0's to find first avail node
  me->job[sig_rx]     = job;                                   // copy job descriptor
  (void) qurt_anysignal_clear(signal, (1 << sig_rx));          // clear the empty job node flag
  (void) qurt_anysignal_set(&me->queued_jobs, (1 << sig_rx));  // notify of pending job
  qurt_mutex_unlock(mutex);                                    // unlock the mutex

  return 0;
}

void worker_pool_destructor() {
  FARF(HIGH, "In worker_pool destructor");

  worker_pool_deinit(&static_context);
}

/*===========================================================================
    GLOBAL FUNCTION
===========================================================================*/
// initialize a synctoken - caller will wait on the synctoken and each job will release it.
// caller wakes when all jobs have released.
void worker_pool_synctoken_init(worker_synctoken_t *token, unsigned int njobs) {
  // cast input to usable struct
  internal_synctoken_t *internal_token = (internal_synctoken_t *) token;

  // initialize atomic counter and semaphore
  internal_token->sync.atomic_countdown = njobs;
  qurt_sem_init_val(&internal_token->sync.sem, 0);
}

// worker job responsible for calling this function to count down completed jobs.
void worker_pool_synctoken_jobdone(worker_synctoken_t *token) {
  // cast input to usable struct
  internal_synctoken_t *internal_token = (internal_synctoken_t *) token;

  // count down atomically, and raise semaphore if last job.
  if (0 == worker_pool_atomic_dec_return(&internal_token->sync.atomic_countdown)) {
    (void) qurt_sem_up(&internal_token->sync.sem);
  }
}

// job submitter waits on this function for all jobs to complete.
void worker_pool_synctoken_wait(worker_synctoken_t *token) {
  // cast input to usable struct
  internal_synctoken_t *internal_token = (internal_synctoken_t *) token;

  // Wait for all jobs to finish and raise the semaphore
  (void) qurt_sem_down(&internal_token->sync.sem);

  // clean up the semaphore
  (void) qurt_sem_destroy(&internal_token->sync.sem);
}

AEEResult worker_pool_set_thread_priority(worker_pool_context_t context, unsigned int prio) {
  worker_pool_t *me = (worker_pool_t *) context;

  // if no worker pool exists, return error.
  if (NULL == me) {
    return AEE_ENOMORE;
  }

  int result = AEE_SUCCESS;
  if (prio < 1) {
    prio = 1;
  }
  if (prio > LOWEST_USABLE_QURT_PRIO) {
    prio = LOWEST_USABLE_QURT_PRIO;
  }
  for (unsigned int i = 0; i < me->num_workers; i++) {
    int res = qurt_thread_set_priority(me->thread[i], (unsigned short) prio);
    if (0 != res) {
      result = AEE_EBADPARM;
      FARF(ERROR, "QURT failed to set priority of thread %d, ERROR = %d", me->thread[i], res);
    }
  }
  return result;
}

AEEResult worker_pool_get_thread_priority(worker_pool_context_t context, unsigned int *prio) {
  worker_pool_t *me = (worker_pool_t *) context;

  // if NULL is passed as context, share static_context's priority.
  if (NULL == me) {
    if (static_context == NULL) {
      return AEE_ENOMORE;
    }
    FARF(HIGH, "Using default static worker pool");
    me = (worker_pool_t *) static_context;
  }

  int priority = qurt_thread_get_priority(me->thread[0]);
  if (priority > 0) {
    *prio = priority;
    return 0;
  } else {
    *prio = 0;
    return AEE_EBADSTATE;
  }
}
