#include "serialport.h"

#include <limits.h>

#define NS_PER_MS 1000000ULL
#define NS_PER_SEC 1000000000ULL

extern uint64_t turbo_hrtime(void);

void time_get(struct time *time)
{
	time->ns = turbo_hrtime();
}

void time_set_ms(struct time *time, unsigned int ms)
{
	time->ns = (uint64_t) ms * NS_PER_MS;
}

void time_add(const struct time *a,
		const struct time *b, struct time *result)
{
	result->ns = a->ns + b->ns;
}

void time_sub(const struct time *a,
		const struct time *b, struct time *result)
{
	result->ns = a->ns > b->ns ? a->ns - b->ns : 0;
}

bool time_greater(const struct time *a, const struct time *b)
{
	return a->ns > b->ns;
}

void time_as_timeval(const struct time *time, struct timeval *tv)
{
	tv->tv_sec = (long) (time->ns / NS_PER_SEC);
	tv->tv_usec = (long) ((time->ns % NS_PER_SEC) / 1000);
}

unsigned int time_as_ms(const struct time *time)
{
	uint64_t ms = time->ns / NS_PER_MS;
	if (ms > UINT_MAX)
		return UINT_MAX;
	return (unsigned int) ms;
}

void timeout_start(struct timeout *timeout, unsigned int timeout_ms)
{
	timeout->ms = timeout_ms;

	/* Get time at start of operation. */
	time_get(&timeout->start);
	/* Define duration of timeout. */
	time_set_ms(&timeout->delta, timeout_ms);
	/* Calculate time at which we should give up. */
	time_add(&timeout->start, &timeout->delta, &timeout->end);
	/* Disable limit unless timeout_limit() called. */
	timeout->limit_ms = 0;
	/* First blocking call has not yet been made. */
	timeout->calls_started = false;
}

void timeout_limit(struct timeout *timeout, unsigned int limit_ms)
{
	timeout->limit_ms = limit_ms;
	timeout->overflow = (timeout->ms > timeout->limit_ms);
	time_set_ms(&timeout->delta_max, timeout->limit_ms);
}

bool timeout_check(struct timeout *timeout)
{
	if (!timeout->calls_started)
		return false;

	if (timeout->ms == 0)
		return false;

	time_get(&timeout->now);
	time_sub(&timeout->end, &timeout->now, &timeout->delta);
	if (timeout->limit_ms)
		if ((timeout->overflow = time_greater(&timeout->delta, &timeout->delta_max)))
			timeout->delta = timeout->delta_max;

	return time_greater(&timeout->now, &timeout->end);
}

void timeout_update(struct timeout *timeout)
{
	timeout->calls_started = true;
}

#ifndef _WIN32
struct timeval *timeout_timeval(struct timeout *timeout)
{
	if (timeout->ms == 0)
		return NULL;

	time_as_timeval(&timeout->delta, &timeout->delta_tv);

	return &timeout->delta_tv;
}
#endif

unsigned int timeout_remaining_ms(struct timeout *timeout)
{
	if (timeout->limit_ms && timeout->overflow)
		return timeout->limit_ms;
	else
		return time_as_ms(&timeout->delta);
}
