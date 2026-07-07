/* Copyright (C) 2013-2016, The Regents of The University of Michigan.
All rights reserved.
This software was developed in the APRIL Robotics Lab under the
direction of Edwin Olson, ebolson@umich.edu. This software may be
available under alternative licensing terms; contact the address above.
Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:
1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.
2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.
THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
The views and conclusions contained in the software and documentation are those
of the authors and should not be interpreted as representing official policies,
either expressed or implied, of the Regents of The University of Michigan.
*/
#include <errno.h>

#define _GNU_SOURCE  // Possible fix for 16.04
#define __USE_GNU
#include <assert.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#else
#include <unistd.h>
#endif

#include "workerpool.h"
#include "debug_print.h"

#include "turbo_thread.h"
typedef turbo_thread_t wp_thread_t;
typedef turbo_mutex_t wp_mutex_t;
typedef turbo_cond_t wp_cond_t;

struct workerpool {
    int nthreads;
    zarray_t *tasks;
    int taskspos;

    wp_thread_t *threads;
    int *status;

    wp_mutex_t mutex;
    wp_cond_t startcond;        // used to signal the availability of work
    bool start_predicate;       // predicate that prevents spurious wakeups on startcond
    wp_cond_t endcond;          // used to signal completion of all work

    int end_count; // how many threads are done?
};

struct task
{
    void (*f)(void *p);
    void *p;
};

static void worker_thread(void *p)
{
    workerpool_t *wp = (workerpool_t*) p;

    while (1) {
        struct task *task;

        turbo_mutex_lock(&wp->mutex);
        while (wp->taskspos == zarray_size(wp->tasks) || !wp->start_predicate) {
            wp->end_count++;
            turbo_cond_broadcast(&wp->endcond);
            turbo_cond_wait(&wp->startcond, &wp->mutex);
        }

        zarray_get_volatile(wp->tasks, wp->taskspos, &task);
        wp->taskspos++;
        turbo_mutex_unlock(&wp->mutex);
        turbo_thread_yield();

        // we've been asked to exit.
        if (task->f == NULL)
            return;

        task->f(task->p);
    }
}

workerpool_t *workerpool_create(int nthreads)
{
    assert(nthreads > 0);

    workerpool_t *wp = calloc(1, sizeof(workerpool_t));
    if (wp == NULL)
        return NULL;

    wp->nthreads = nthreads;
    wp->tasks = zarray_create(sizeof(struct task));
    wp->start_predicate = false;

    if (nthreads > 1) {
        wp->threads = calloc(wp->nthreads, sizeof(wp_thread_t));

        turbo_mutex_init(&wp->mutex);
        turbo_cond_init(&wp->startcond);
        turbo_cond_init(&wp->endcond);

        for (int i = 0; i < nthreads; i++) {
            int res = turbo_thread_create(&wp->threads[i], worker_thread, wp);
            if (res != 0) {
                debug_print("Insufficient system resources to create workerpool threads\n");
                errno = EAGAIN;
                return NULL;
            }
        }

        // Wait for the worker threads to be ready
        turbo_mutex_lock(&wp->mutex);
        while (wp->end_count < wp->nthreads) {
            turbo_cond_wait(&wp->endcond, &wp->mutex);
        }
        turbo_mutex_unlock(&wp->mutex);
    }

    return wp;
}

void workerpool_destroy(workerpool_t *wp)
{
    if (wp == NULL)
        return;

    // force all worker threads to exit.
    if (wp->nthreads > 1) {
        for (int i = 0; i < wp->nthreads; i++)
            workerpool_add_task(wp, NULL, NULL);

        turbo_mutex_lock(&wp->mutex);
        wp->start_predicate = true;
        turbo_cond_broadcast(&wp->startcond);
        turbo_mutex_unlock(&wp->mutex);

        for (int i = 0; i < wp->nthreads; i++)
            turbo_thread_join(&wp->threads[i]);

        turbo_mutex_destroy(&wp->mutex);
        turbo_cond_destroy(&wp->startcond);
        turbo_cond_destroy(&wp->endcond);
        free(wp->threads);
    }

    zarray_destroy(wp->tasks);
    free(wp);
}

int workerpool_get_nthreads(workerpool_t *wp)
{
    return wp->nthreads;
}

void workerpool_add_task(workerpool_t *wp, void (*f)(void *p), void *p)
{
    struct task t;
    t.f = f;
    t.p = p;

    if (wp->nthreads > 1) {
        turbo_mutex_lock(&wp->mutex);
        zarray_add(wp->tasks, &t);
        turbo_mutex_unlock(&wp->mutex);
    } else {
        zarray_add(wp->tasks, &t);
    }
}

void workerpool_run_single(workerpool_t *wp)
{
    for (int i = 0; i < zarray_size(wp->tasks); i++) {
        struct task *task;
        zarray_get_volatile(wp->tasks, i, &task);
        task->f(task->p);
    }

    zarray_clear(wp->tasks);
}

// runs all added tasks, waits for them to complete.
void workerpool_run(workerpool_t *wp)
{
    if (wp->nthreads > 1) {
        turbo_mutex_lock(&wp->mutex);
        wp->end_count = 0;
        wp->start_predicate = true;
        turbo_cond_broadcast(&wp->startcond);

        while (wp->end_count < wp->nthreads) {
//            printf("caught %d\n", wp->end_count);
            turbo_cond_wait(&wp->endcond, &wp->mutex);
        }

        wp->taskspos = 0;
        wp->start_predicate = false;
        turbo_mutex_unlock(&wp->mutex);

        zarray_clear(wp->tasks);

    } else {
        workerpool_run_single(wp);
    }
}

int workerpool_get_nprocs()
{
#ifdef _WIN32
    SYSTEM_INFO sysinfo;
    GetSystemInfo(&sysinfo);
    return sysinfo.dwNumberOfProcessors;
#else
    return sysconf (_SC_NPROCESSORS_ONLN);
#endif
}
