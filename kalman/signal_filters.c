/** @file signal_filters.c
 * KFCore
 *
 * @brief Lightweight scalar and vector signal filters.
 * @{ */

/******************************************************************************
 * SYSTEM INCLUDE FILES
 ******************************************************************************/

#include <math.h>

/******************************************************************************
 * PROJECT INCLUDE FILES
 ******************************************************************************/

#include "signal_filters.h"
#include "linalg.h"

/******************************************************************************
 * DEFINES
 ******************************************************************************/

#ifndef KF_SIGNAL_PI
#define KF_SIGNAL_PI 3.14159265358979323846f
#endif

/******************************************************************************
 * LOCAL FUNCTION PROTOTYPES
 ******************************************************************************/

static int valid_alpha(float alpha);
static void insertion_sort(float* values, int n);
static float median_of_values(float* scratch, const float* values, int n);
static float quantile_of_sorted(const float* sorted, int n, float quantile);
static int valid_biquad_frequency(float sample_rate_hz, float frequency_hz, float q);
static void normalize_biquad(float b0, float b1, float b2, float a0, float a1, float a2,
                             kf_signal_biquad_coeffs* coeffs);

/******************************************************************************
 * FUNCTION BODIES
 ******************************************************************************/

int kf_signal_lowpass_alpha(float cutoff_hz, float sample_period_s, float* alpha)
{
    float rc;

    if (!alpha || cutoff_hz <= 0.0f || sample_period_s <= 0.0f)
    {
        return -1;
    }

    rc     = 1.0f / (2.0f * KF_SIGNAL_PI * cutoff_hz);
    *alpha = sample_period_s / (rc + sample_period_s);

    return 0;
}

int kf_signal_highpass_alpha(float cutoff_hz, float sample_period_s, float* alpha)
{
    float rc;

    if (!alpha || cutoff_hz <= 0.0f || sample_period_s <= 0.0f)
    {
        return -1;
    }

    rc     = 1.0f / (2.0f * KF_SIGNAL_PI * cutoff_hz);
    *alpha = rc / (rc + sample_period_s);

    return 0;
}

int kf_signal_lowpass1f_init(kf_signal_lowpass1f* filter, float alpha, float initial_output)
{
    if (!filter || !valid_alpha(alpha))
    {
        return -1;
    }

    filter->alpha       = alpha;
    filter->y           = initial_output;
    filter->initialized = 1;

    return 0;
}

int kf_signal_lowpass1f_update(kf_signal_lowpass1f* filter, float x, float* y)
{
    if (!filter || !y || !filter->initialized || !valid_alpha(filter->alpha))
    {
        return -1;
    }

    filter->y += filter->alpha * (x - filter->y);
    *y = filter->y;

    return 0;
}

int kf_signal_lowpass_vector(float* y, const float* x, int n, float alpha)
{
    if (!y || !x || n <= 0 || !valid_alpha(alpha))
    {
        return -1;
    }

    for (int i = 0; i < n; ++i)
    {
        y[i] += alpha * (x[i] - y[i]);
    }

    return 0;
}

int kf_signal_highpass1f_init(kf_signal_highpass1f* filter, float alpha, float initial_input,
                              float initial_output)
{
    if (!filter || !valid_alpha(alpha))
    {
        return -1;
    }

    filter->alpha       = alpha;
    filter->prev_x      = initial_input;
    filter->y           = initial_output;
    filter->initialized = 1;

    return 0;
}

int kf_signal_highpass1f_update(kf_signal_highpass1f* filter, float x, float* y)
{
    if (!filter || !y || !filter->initialized || !valid_alpha(filter->alpha))
    {
        return -1;
    }

    filter->y      = filter->alpha * (filter->y + x - filter->prev_x);
    filter->prev_x = x;
    *y             = filter->y;

    return 0;
}

int kf_signal_highpass_vector(float* y, float* prev_x, const float* x, int n, float alpha)
{
    if (!y || !prev_x || !x || n <= 0 || !valid_alpha(alpha))
    {
        return -1;
    }

    for (int i = 0; i < n; ++i)
    {
        y[i]      = alpha * (y[i] + x[i] - prev_x[i]);
        prev_x[i] = x[i];
    }

    return 0;
}

int kf_signal_moving_average1f_init(kf_signal_moving_average1f* filter, float* window,
                                    int window_size)
{
    if (!filter || !window || window_size <= 0)
    {
        return -1;
    }

    for (int i = 0; i < window_size; ++i)
    {
        window[i] = 0.0f;
    }

    filter->window      = window;
    filter->window_size = window_size;
    filter->index       = 0;
    filter->count       = 0;
    filter->sum         = 0.0f;

    return 0;
}

int kf_signal_moving_average1f_update(kf_signal_moving_average1f* filter, float x, float* y)
{
    if (!filter || !filter->window || !y || filter->window_size <= 0)
    {
        return -1;
    }

    if (filter->count < filter->window_size)
    {
        filter->window[filter->index] = x;
        filter->sum += x;
        ++filter->count;
    }
    else
    {
        filter->sum -= filter->window[filter->index];
        filter->window[filter->index] = x;
        filter->sum += x;
    }

    ++filter->index;
    if (filter->index >= filter->window_size)
    {
        filter->index = 0;
    }

    *y = filter->sum / (float)filter->count;

    return 0;
}

int kf_signal_median1f_init(kf_signal_median1f* filter, float* window, float* scratch,
                            int window_size)
{
    if (!filter || !window || !scratch || window_size <= 0)
    {
        return -1;
    }

    for (int i = 0; i < window_size; ++i)
    {
        window[i]  = 0.0f;
        scratch[i] = 0.0f;
    }

    filter->window      = window;
    filter->scratch     = scratch;
    filter->window_size = window_size;
    filter->index       = 0;
    filter->count       = 0;

    return 0;
}

int kf_signal_median1f_update(kf_signal_median1f* filter, float x, float* y)
{
    if (!filter || !filter->window || !filter->scratch || !y || filter->window_size <= 0)
    {
        return -1;
    }

    filter->window[filter->index] = x;
    if (filter->count < filter->window_size)
    {
        ++filter->count;
    }

    ++filter->index;
    if (filter->index >= filter->window_size)
    {
        filter->index = 0;
    }

    *y = median_of_values(filter->scratch, filter->window, filter->count);

    return 0;
}

int kf_signal_moving_rms1f_init(kf_signal_moving_rms1f* filter, float* window, int window_size)
{
    if (!filter || !window || window_size <= 0)
    {
        return -1;
    }

    for (int i = 0; i < window_size; ++i)
    {
        window[i] = 0.0f;
    }

    filter->window      = window;
    filter->window_size = window_size;
    filter->index       = 0;
    filter->count       = 0;
    filter->sum_sq      = 0.0f;

    return 0;
}

int kf_signal_moving_rms1f_update(kf_signal_moving_rms1f* filter, float x, float* y)
{
    if (!filter || !filter->window || !y || filter->window_size <= 0)
    {
        return -1;
    }

    if (filter->count < filter->window_size)
    {
        ++filter->count;
    }
    else
    {
        filter->sum_sq -= filter->window[filter->index] * filter->window[filter->index];
    }

    filter->window[filter->index] = x;
    filter->sum_sq += x * x;

    ++filter->index;
    if (filter->index >= filter->window_size)
    {
        filter->index = 0;
    }

    *y = sqrtf(filter->sum_sq / (float)filter->count);

    return 0;
}

int kf_signal_trimmed_mean1f_init(kf_signal_trimmed_mean1f* filter, float* window,
                                  float* scratch, int window_size, int trim_each_side)
{
    if (!filter || !window || !scratch || window_size <= 0 || trim_each_side < 0 ||
        2 * trim_each_side >= window_size)
    {
        return -1;
    }

    for (int i = 0; i < window_size; ++i)
    {
        window[i]  = 0.0f;
        scratch[i] = 0.0f;
    }

    filter->window         = window;
    filter->scratch        = scratch;
    filter->window_size    = window_size;
    filter->trim_each_side = trim_each_side;
    filter->index          = 0;
    filter->count          = 0;

    return 0;
}

int kf_signal_trimmed_mean1f_update(kf_signal_trimmed_mean1f* filter, float x, float* y)
{
    int   trim;
    int   kept;
    float sum = 0.0f;

    if (!filter || !filter->window || !filter->scratch || !y || filter->window_size <= 0 ||
        filter->trim_each_side < 0)
    {
        return -1;
    }

    filter->window[filter->index] = x;
    if (filter->count < filter->window_size)
    {
        ++filter->count;
    }
    ++filter->index;
    if (filter->index >= filter->window_size)
    {
        filter->index = 0;
    }

    trim = filter->trim_each_side;
    if (2 * trim >= filter->count)
    {
        trim = 0;
    }

    for (int i = 0; i < filter->count; ++i)
    {
        filter->scratch[i] = filter->window[i];
    }
    insertion_sort(filter->scratch, filter->count);

    kept = filter->count - 2 * trim;
    for (int i = trim; i < filter->count - trim; ++i)
    {
        sum += filter->scratch[i];
    }
    *y = sum / (float)kept;

    return 0;
}

int kf_signal_winsorized_mean1f_init(kf_signal_winsorized_mean1f* filter, float* window,
                                     float* scratch, int window_size, int winsor_each_side)
{
    if (!filter || !window || !scratch || window_size <= 0 || winsor_each_side < 0 ||
        2 * winsor_each_side >= window_size)
    {
        return -1;
    }

    for (int i = 0; i < window_size; ++i)
    {
        window[i]  = 0.0f;
        scratch[i] = 0.0f;
    }

    filter->window           = window;
    filter->scratch          = scratch;
    filter->window_size      = window_size;
    filter->winsor_each_side = winsor_each_side;
    filter->index            = 0;
    filter->count            = 0;

    return 0;
}

int kf_signal_winsorized_mean1f_update(kf_signal_winsorized_mean1f* filter, float x, float* y)
{
    int   winsor;
    float low;
    float high;
    float sum = 0.0f;

    if (!filter || !filter->window || !filter->scratch || !y || filter->window_size <= 0 ||
        filter->winsor_each_side < 0)
    {
        return -1;
    }

    filter->window[filter->index] = x;
    if (filter->count < filter->window_size)
    {
        ++filter->count;
    }
    ++filter->index;
    if (filter->index >= filter->window_size)
    {
        filter->index = 0;
    }

    winsor = filter->winsor_each_side;
    if (2 * winsor >= filter->count)
    {
        winsor = 0;
    }

    for (int i = 0; i < filter->count; ++i)
    {
        filter->scratch[i] = filter->window[i];
    }
    insertion_sort(filter->scratch, filter->count);

    low  = filter->scratch[winsor];
    high = filter->scratch[filter->count - winsor - 1];
    for (int i = 0; i < filter->count; ++i)
    {
        float value = filter->scratch[i];
        if (value < low)
        {
            value = low;
        }
        else if (value > high)
        {
            value = high;
        }
        sum += value;
    }

    *y = sum / (float)filter->count;

    return 0;
}

int kf_signal_savgol1f_init(kf_signal_savgol1f* filter, const float* coeffs, float* window,
                            int window_size)
{
    if (!filter || !coeffs || !window || window_size <= 0)
    {
        return -1;
    }

    for (int i = 0; i < window_size; ++i)
    {
        window[i] = 0.0f;
    }

    filter->coeffs      = coeffs;
    filter->window      = window;
    filter->window_size = window_size;
    filter->index       = 0;
    filter->count       = 0;

    return 0;
}

int kf_signal_savgol1f_update(kf_signal_savgol1f* filter, float x, float* y)
{
    float acc = 0.0f;
    int   idx;

    if (!filter || !filter->coeffs || !filter->window || !y || filter->window_size <= 0)
    {
        return -1;
    }

    filter->window[filter->index] = x;
    if (filter->count < filter->window_size)
    {
        ++filter->count;
    }

    idx = filter->index;
    for (int i = 0; i < filter->window_size; ++i)
    {
        if (i < filter->count)
        {
            acc += filter->coeffs[i] * filter->window[idx];
        }
        --idx;
        if (idx < 0)
        {
            idx = filter->window_size - 1;
        }
    }

    ++filter->index;
    if (filter->index >= filter->window_size)
    {
        filter->index = 0;
    }

    *y = acc;

    return 0;
}

int kf_signal_clamp1f(float x, float min_value, float max_value, float* y)
{
    if (!y || min_value > max_value)
    {
        return -1;
    }

    if (x < min_value)
    {
        *y = min_value;
    }
    else if (x > max_value)
    {
        *y = max_value;
    }
    else
    {
        *y = x;
    }

    return 0;
}

int kf_signal_deadband1f(float x, float deadband, float* y)
{
    if (!y || deadband < 0.0f)
    {
        return -1;
    }

    if (fabsf(x) <= deadband)
    {
        *y = 0.0f;
    }
    else if (x > 0.0f)
    {
        *y = x - deadband;
    }
    else
    {
        *y = x + deadband;
    }

    return 0;
}

int kf_signal_slew_rate_limit1f(float previous, float target, float max_delta, float* y)
{
    const float delta = target - previous;

    if (!y || max_delta < 0.0f)
    {
        return -1;
    }

    if (delta > max_delta)
    {
        *y = previous + max_delta;
    }
    else if (delta < -max_delta)
    {
        *y = previous - max_delta;
    }
    else
    {
        *y = target;
    }

    return 0;
}

int kf_signal_hampel1f_init(kf_signal_hampel1f* filter, float* window, float* scratch,
                            int window_size)
{
    if (!filter || !window || !scratch || window_size <= 0)
    {
        return -1;
    }

    for (int i = 0; i < window_size; ++i)
    {
        window[i]  = 0.0f;
        scratch[i] = 0.0f;
    }

    filter->window      = window;
    filter->scratch     = scratch;
    filter->window_size = window_size;
    filter->index       = 0;
    filter->count       = 0;

    return 0;
}

int kf_signal_hampel1f_update(kf_signal_hampel1f* filter, float x, float n_sigma, float* y)
{
    float median;
    float mad;
    float threshold;
    int   accepted = 1;

    if (!filter || !filter->window || !filter->scratch || !y || filter->window_size <= 0 ||
        n_sigma < 0.0f)
    {
        return -1;
    }

    if (filter->count > 0)
    {
        median = median_of_values(filter->scratch, filter->window, filter->count);
        for (int i = 0; i < filter->count; ++i)
        {
            filter->scratch[i] = fabsf(filter->window[i] - median);
        }
        insertion_sort(filter->scratch, filter->count);
        if ((filter->count % 2) != 0)
        {
            mad = filter->scratch[filter->count / 2];
        }
        else
        {
            mad = 0.5f *
                  (filter->scratch[(filter->count / 2) - 1] + filter->scratch[filter->count / 2]);
        }

        threshold = n_sigma * 1.4826f * mad;
        if (threshold > 0.0f && fabsf(x - median) > threshold)
        {
            *y       = median;
            accepted = 0;
        }
        else
        {
            *y = x;
        }
    }
    else
    {
        *y = x;
    }

    filter->window[filter->index] = *y;
    if (filter->count < filter->window_size)
    {
        ++filter->count;
    }

    ++filter->index;
    if (filter->index >= filter->window_size)
    {
        filter->index = 0;
    }

    return accepted;
}

int kf_signal_biquad1f_init(kf_signal_biquad1f* filter, const kf_signal_biquad_coeffs* coeffs)
{
    if (!filter || !coeffs)
    {
        return -1;
    }

    filter->coeffs     = *coeffs;
    filter->x1         = 0.0f;
    filter->x2         = 0.0f;
    filter->y1         = 0.0f;
    filter->y2         = 0.0f;
    filter->initialized = 1;

    return 0;
}

int kf_signal_biquad1f_reset(kf_signal_biquad1f* filter, float x1, float x2, float y1, float y2)
{
    if (!filter || !filter->initialized)
    {
        return -1;
    }

    filter->x1 = x1;
    filter->x2 = x2;
    filter->y1 = y1;
    filter->y2 = y2;

    return 0;
}

int kf_signal_biquad1f_update(kf_signal_biquad1f* filter, float x, float* y)
{
    const kf_signal_biquad_coeffs* c;
    float                          out;

    if (!filter || !filter->initialized || !y)
    {
        return -1;
    }

    c   = &filter->coeffs;
    out = c->b0 * x + c->b1 * filter->x1 + c->b2 * filter->x2 - c->a1 * filter->y1 -
          c->a2 * filter->y2;

    filter->x2 = filter->x1;
    filter->x1 = x;
    filter->y2 = filter->y1;
    filter->y1 = out;
    *y         = out;

    return 0;
}

int kf_signal_biquad1f_init_lowpass(kf_signal_biquad1f* filter, float sample_rate_hz,
                                    float cutoff_hz, float q)
{
    kf_signal_biquad_coeffs coeffs;

    if (kf_signal_biquad_lowpass(sample_rate_hz, cutoff_hz, q, &coeffs) != 0)
    {
        return -1;
    }

    return kf_signal_biquad1f_init(filter, &coeffs);
}

int kf_signal_biquad1f_init_highpass(kf_signal_biquad1f* filter, float sample_rate_hz,
                                     float cutoff_hz, float q)
{
    kf_signal_biquad_coeffs coeffs;

    if (kf_signal_biquad_highpass(sample_rate_hz, cutoff_hz, q, &coeffs) != 0)
    {
        return -1;
    }

    return kf_signal_biquad1f_init(filter, &coeffs);
}

int kf_signal_biquad1f_init_bandpass(kf_signal_biquad1f* filter, float sample_rate_hz,
                                     float center_hz, float q)
{
    kf_signal_biquad_coeffs coeffs;

    if (kf_signal_biquad_bandpass(sample_rate_hz, center_hz, q, &coeffs) != 0)
    {
        return -1;
    }

    return kf_signal_biquad1f_init(filter, &coeffs);
}

int kf_signal_biquad1f_init_notch(kf_signal_biquad1f* filter, float sample_rate_hz,
                                  float center_hz, float q)
{
    kf_signal_biquad_coeffs coeffs;

    if (kf_signal_biquad_notch(sample_rate_hz, center_hz, q, &coeffs) != 0)
    {
        return -1;
    }

    return kf_signal_biquad1f_init(filter, &coeffs);
}

int kf_signal_biquad_lowpass(float sample_rate_hz, float cutoff_hz, float q,
                             kf_signal_biquad_coeffs* coeffs)
{
    float omega;
    float sin_omega;
    float cos_omega;
    float alpha;

    if (!coeffs || !valid_biquad_frequency(sample_rate_hz, cutoff_hz, q))
    {
        return -1;
    }

    omega     = 2.0f * KF_SIGNAL_PI * cutoff_hz / sample_rate_hz;
    sin_omega = sinf(omega);
    cos_omega = cosf(omega);
    alpha     = sin_omega / (2.0f * q);

    normalize_biquad((1.0f - cos_omega) * 0.5f, 1.0f - cos_omega,
                     (1.0f - cos_omega) * 0.5f, 1.0f + alpha, -2.0f * cos_omega,
                     1.0f - alpha, coeffs);

    return 0;
}

int kf_signal_biquad_highpass(float sample_rate_hz, float cutoff_hz, float q,
                              kf_signal_biquad_coeffs* coeffs)
{
    float omega;
    float sin_omega;
    float cos_omega;
    float alpha;

    if (!coeffs || !valid_biquad_frequency(sample_rate_hz, cutoff_hz, q))
    {
        return -1;
    }

    omega     = 2.0f * KF_SIGNAL_PI * cutoff_hz / sample_rate_hz;
    sin_omega = sinf(omega);
    cos_omega = cosf(omega);
    alpha     = sin_omega / (2.0f * q);

    normalize_biquad((1.0f + cos_omega) * 0.5f, -(1.0f + cos_omega),
                     (1.0f + cos_omega) * 0.5f, 1.0f + alpha, -2.0f * cos_omega,
                     1.0f - alpha, coeffs);

    return 0;
}

int kf_signal_biquad_bandpass(float sample_rate_hz, float center_hz, float q,
                              kf_signal_biquad_coeffs* coeffs)
{
    float omega;
    float sin_omega;
    float cos_omega;
    float alpha;

    if (!coeffs || !valid_biquad_frequency(sample_rate_hz, center_hz, q))
    {
        return -1;
    }

    omega     = 2.0f * KF_SIGNAL_PI * center_hz / sample_rate_hz;
    sin_omega = sinf(omega);
    cos_omega = cosf(omega);
    alpha     = sin_omega / (2.0f * q);

    normalize_biquad(sin_omega * 0.5f, 0.0f, -sin_omega * 0.5f, 1.0f + alpha,
                     -2.0f * cos_omega, 1.0f - alpha, coeffs);

    return 0;
}

int kf_signal_biquad_notch(float sample_rate_hz, float center_hz, float q,
                           kf_signal_biquad_coeffs* coeffs)
{
    float omega;
    float sin_omega;
    float cos_omega;
    float alpha;

    if (!coeffs || !valid_biquad_frequency(sample_rate_hz, center_hz, q))
    {
        return -1;
    }

    omega     = 2.0f * KF_SIGNAL_PI * center_hz / sample_rate_hz;
    sin_omega = sinf(omega);
    cos_omega = cosf(omega);
    alpha     = sin_omega / (2.0f * q);

    normalize_biquad(1.0f, -2.0f * cos_omega, 1.0f, 1.0f + alpha, -2.0f * cos_omega,
                     1.0f - alpha, coeffs);

    return 0;
}

int kf_signal_fir1f_init(kf_signal_fir1f* filter, const float* coeffs, float* state, int taps)
{
    if (!filter || !coeffs || !state || taps <= 0)
    {
        return -1;
    }

    for (int i = 0; i < taps; ++i)
    {
        state[i] = 0.0f;
    }

    filter->coeffs = coeffs;
    filter->state  = state;
    filter->taps   = taps;
    filter->index  = 0;

    return 0;
}

int kf_signal_fir1f_update(kf_signal_fir1f* filter, float x, float* y)
{
    float acc = 0.0f;
    int   idx;

    if (!filter || !filter->coeffs || !filter->state || !y || filter->taps <= 0)
    {
        return -1;
    }

    filter->state[filter->index] = x;
    idx                          = filter->index;
    for (int i = 0; i < filter->taps; ++i)
    {
        acc += filter->coeffs[i] * filter->state[idx];
        --idx;
        if (idx < 0)
        {
            idx = filter->taps - 1;
        }
    }

    ++filter->index;
    if (filter->index >= filter->taps)
    {
        filter->index = 0;
    }

    *y = acc;

    return 0;
}

int kf_signal_iir1f_init(kf_signal_iir1f* filter, const float* b, int nb, const float* a, int na,
                         float* x_state, float* y_state)
{
    if (!filter || !b || !a || !x_state || !y_state || nb <= 0 || na <= 0 || a[0] == 0.0f)
    {
        return -1;
    }

    for (int i = 0; i < nb; ++i)
    {
        x_state[i] = 0.0f;
    }
    for (int i = 0; i < na - 1; ++i)
    {
        y_state[i] = 0.0f;
    }

    filter->b       = b;
    filter->a       = a;
    filter->x_state = x_state;
    filter->y_state = y_state;
    filter->nb      = nb;
    filter->na      = na;

    return 0;
}

int kf_signal_iir1f_update(kf_signal_iir1f* filter, float x, float* y)
{
    float acc = 0.0f;

    if (!filter || !filter->b || !filter->a || !filter->x_state || !filter->y_state || !y ||
        filter->nb <= 0 || filter->na <= 0 || filter->a[0] == 0.0f)
    {
        return -1;
    }

    for (int i = filter->nb - 1; i > 0; --i)
    {
        filter->x_state[i] = filter->x_state[i - 1];
    }
    filter->x_state[0] = x;

    for (int i = 0; i < filter->nb; ++i)
    {
        acc += filter->b[i] * filter->x_state[i];
    }
    for (int i = 1; i < filter->na; ++i)
    {
        acc -= filter->a[i] * filter->y_state[i - 1];
    }

    *y = acc / filter->a[0];

    for (int i = filter->na - 2; i > 0; --i)
    {
        filter->y_state[i] = filter->y_state[i - 1];
    }
    if (filter->na > 1)
    {
        filter->y_state[0] = *y;
    }

    return 0;
}

int kf_signal_alpha_beta1f_init(kf_signal_alpha_beta1f* filter, float alpha, float beta, float dt,
                                float initial_position, float initial_velocity)
{
    if (!filter || !valid_alpha(alpha) || beta < 0.0f || dt <= 0.0f)
    {
        return -1;
    }

    filter->alpha       = alpha;
    filter->beta        = beta;
    filter->dt          = dt;
    filter->position    = initial_position;
    filter->velocity    = initial_velocity;
    filter->initialized = 1;

    return 0;
}

int kf_signal_alpha_beta1f_update(kf_signal_alpha_beta1f* filter, float measurement,
                                  float* position, float* velocity)
{
    float predicted_position;
    float residual;

    if (!filter || !filter->initialized || !position || !velocity || !valid_alpha(filter->alpha) ||
        filter->beta < 0.0f || filter->dt <= 0.0f)
    {
        return -1;
    }

    predicted_position = filter->position + filter->velocity * filter->dt;
    residual           = measurement - predicted_position;
    filter->position   = predicted_position + filter->alpha * residual;
    filter->velocity += (filter->beta / filter->dt) * residual;

    *position = filter->position;
    *velocity = filter->velocity;

    return 0;
}

int kf_signal_alpha_beta_gamma1f_init(kf_signal_alpha_beta_gamma1f* filter, float alpha,
                                      float beta, float gamma, float dt, float initial_position,
                                      float initial_velocity, float initial_acceleration)
{
    if (!filter || !valid_alpha(alpha) || beta < 0.0f || gamma < 0.0f || dt <= 0.0f)
    {
        return -1;
    }

    filter->alpha        = alpha;
    filter->beta         = beta;
    filter->gamma        = gamma;
    filter->dt           = dt;
    filter->position     = initial_position;
    filter->velocity     = initial_velocity;
    filter->acceleration = initial_acceleration;
    filter->initialized  = 1;

    return 0;
}

int kf_signal_alpha_beta_gamma1f_update(kf_signal_alpha_beta_gamma1f* filter, float measurement,
                                        float* position, float* velocity, float* acceleration)
{
    float predicted_position;
    float predicted_velocity;
    float residual;
    float dt;

    if (!filter || !filter->initialized || !position || !velocity || !acceleration ||
        !valid_alpha(filter->alpha) || filter->beta < 0.0f || filter->gamma < 0.0f ||
        filter->dt <= 0.0f)
    {
        return -1;
    }

    dt                 = filter->dt;
    predicted_position = filter->position + filter->velocity * dt +
                         0.5f * filter->acceleration * dt * dt;
    predicted_velocity = filter->velocity + filter->acceleration * dt;
    residual           = measurement - predicted_position;

    filter->position     = predicted_position + filter->alpha * residual;
    filter->velocity     = predicted_velocity + (filter->beta / dt) * residual;
    filter->acceleration = filter->acceleration + (2.0f * filter->gamma / (dt * dt)) * residual;

    *position     = filter->position;
    *velocity     = filter->velocity;
    *acceleration = filter->acceleration;

    return 0;
}

int kf_signal_complementary1f_init(kf_signal_complementary1f* filter, float alpha, float dt,
                                   float initial_value)
{
    if (!filter || !valid_alpha(alpha) || dt <= 0.0f)
    {
        return -1;
    }

    filter->alpha       = alpha;
    filter->dt          = dt;
    filter->value       = initial_value;
    filter->initialized = 1;

    return 0;
}

int kf_signal_complementary1f_update(kf_signal_complementary1f* filter, float rate,
                                     float absolute, float* y)
{
    if (!filter || !filter->initialized || !valid_alpha(filter->alpha) || filter->dt <= 0.0f ||
        !y)
    {
        return -1;
    }

    filter->value = filter->alpha * (filter->value + rate * filter->dt) +
                    (1.0f - filter->alpha) * absolute;
    *y = filter->value;

    return 0;
}

int kf_signal_hysteresis1f_init(kf_signal_hysteresis1f* filter, float low_threshold,
                                float high_threshold, int initial_state)
{
    if (!filter || low_threshold > high_threshold)
    {
        return -1;
    }

    filter->low_threshold  = low_threshold;
    filter->high_threshold = high_threshold;
    filter->state          = initial_state ? 1 : 0;
    filter->initialized    = 1;

    return 0;
}

int kf_signal_hysteresis1f_update(kf_signal_hysteresis1f* filter, float x, int* state)
{
    if (!filter || !filter->initialized || !state)
    {
        return -1;
    }

    if (x >= filter->high_threshold)
    {
        filter->state = 1;
    }
    else if (x <= filter->low_threshold)
    {
        filter->state = 0;
    }

    *state = filter->state;

    return 0;
}

int kf_signal_debounce_init(kf_signal_debounce* filter, int threshold_samples, int initial_state)
{
    if (!filter || threshold_samples <= 0)
    {
        return -1;
    }

    filter->stable_state      = initial_state ? 1 : 0;
    filter->candidate_state   = filter->stable_state;
    filter->counter           = 0;
    filter->threshold_samples = threshold_samples;
    filter->initialized       = 1;

    return 0;
}

int kf_signal_debounce_update(kf_signal_debounce* filter, int input_state, int* stable_state)
{
    const int normalized_input = input_state ? 1 : 0;

    if (!filter || !filter->initialized || !stable_state || filter->threshold_samples <= 0)
    {
        return -1;
    }

    if (normalized_input == filter->stable_state)
    {
        filter->candidate_state = normalized_input;
        filter->counter         = 0;
    }
    else if (normalized_input == filter->candidate_state)
    {
        ++filter->counter;
        if (filter->counter >= filter->threshold_samples)
        {
            filter->stable_state = normalized_input;
            filter->counter      = 0;
        }
    }
    else
    {
        filter->candidate_state = normalized_input;
        filter->counter         = 1;
    }

    *stable_state = filter->stable_state;

    return 0;
}

int kf_signal_rate_limiter1f_init(kf_signal_rate_limiter1f* filter, float max_rate, float dt,
                                  float initial_value)
{
    if (!filter || max_rate < 0.0f || dt <= 0.0f)
    {
        return -1;
    }

    filter->max_rate    = max_rate;
    filter->dt          = dt;
    filter->value       = initial_value;
    filter->initialized = 1;

    return 0;
}

int kf_signal_rate_limiter1f_update(kf_signal_rate_limiter1f* filter, float target, float* y)
{
    if (!filter || !filter->initialized || !y || filter->max_rate < 0.0f || filter->dt <= 0.0f)
    {
        return -1;
    }

    if (kf_signal_slew_rate_limit1f(filter->value, target, filter->max_rate * filter->dt,
                                    &filter->value) != 0)
    {
        return -1;
    }

    *y = filter->value;

    return 0;
}

int kf_signal_accel_limiter1f_init(kf_signal_accel_limiter1f* filter, float max_rate,
                                   float max_accel, float dt, float initial_value,
                                   float initial_rate)
{
    if (!filter || max_rate < 0.0f || max_accel < 0.0f || dt <= 0.0f)
    {
        return -1;
    }

    filter->max_rate    = max_rate;
    filter->max_accel   = max_accel;
    filter->dt          = dt;
    filter->value       = initial_value;
    filter->rate        = initial_rate;
    filter->initialized = 1;

    return 0;
}

int kf_signal_accel_limiter1f_update(kf_signal_accel_limiter1f* filter, float target, float* y,
                                     float* rate)
{
    float desired_rate;

    if (!filter || !filter->initialized || !y || !rate || filter->max_rate < 0.0f ||
        filter->max_accel < 0.0f || filter->dt <= 0.0f)
    {
        return -1;
    }

    desired_rate = (target - filter->value) / filter->dt;
    if (kf_signal_clamp1f(desired_rate, -filter->max_rate, filter->max_rate, &desired_rate) != 0)
    {
        return -1;
    }
    if (kf_signal_slew_rate_limit1f(filter->rate, desired_rate, filter->max_accel * filter->dt,
                                    &filter->rate) != 0)
    {
        return -1;
    }

    filter->value += filter->rate * filter->dt;
    if ((filter->rate > 0.0f && filter->value > target) ||
        (filter->rate < 0.0f && filter->value < target))
    {
        filter->value = target;
        filter->rate  = 0.0f;
    }

    *y    = filter->value;
    *rate = filter->rate;

    return 0;
}

int kf_signal_one_euro1f_init(kf_signal_one_euro1f* filter, float min_cutoff_hz, float beta,
                              float derivative_cutoff_hz, float dt, float initial_value)
{
    if (!filter || min_cutoff_hz <= 0.0f || beta < 0.0f || derivative_cutoff_hz <= 0.0f ||
        dt <= 0.0f)
    {
        return -1;
    }

    filter->min_cutoff_hz       = min_cutoff_hz;
    filter->beta                = beta;
    filter->derivative_cutoff_hz = derivative_cutoff_hz;
    filter->dt                  = dt;
    filter->x                   = initial_value;
    filter->dx                  = 0.0f;
    filter->initialized         = 1;

    return 0;
}

int kf_signal_one_euro1f_update(kf_signal_one_euro1f* filter, float x, float* y)
{
    float alpha_d;
    float alpha_x;
    float dx_raw;
    float cutoff;

    if (!filter || !filter->initialized || !y || filter->min_cutoff_hz <= 0.0f ||
        filter->beta < 0.0f || filter->derivative_cutoff_hz <= 0.0f || filter->dt <= 0.0f)
    {
        return -1;
    }

    if (kf_signal_lowpass_alpha(filter->derivative_cutoff_hz, filter->dt, &alpha_d) != 0)
    {
        return -1;
    }
    dx_raw = (x - filter->x) / filter->dt;
    filter->dx += alpha_d * (dx_raw - filter->dx);

    cutoff = filter->min_cutoff_hz + filter->beta * fabsf(filter->dx);
    if (kf_signal_lowpass_alpha(cutoff, filter->dt, &alpha_x) != 0)
    {
        return -1;
    }
    filter->x += alpha_x * (x - filter->x);
    *y = filter->x;

    return 0;
}

int kf_signal_adaptive_ema1f_init(kf_signal_adaptive_ema1f* filter, float alpha_min,
                                  float alpha_max, float residual_scale, float initial_output)
{
    if (!filter || !valid_alpha(alpha_min) || !valid_alpha(alpha_max) || alpha_min > alpha_max ||
        residual_scale <= 0.0f)
    {
        return -1;
    }

    filter->alpha_min      = alpha_min;
    filter->alpha_max      = alpha_max;
    filter->residual_scale = residual_scale;
    filter->y              = initial_output;
    filter->initialized    = 1;

    return 0;
}

int kf_signal_adaptive_ema1f_update(kf_signal_adaptive_ema1f* filter, float x, float* y,
                                    float* alpha)
{
    float residual;
    float ratio;
    float a;

    if (!filter || !filter->initialized || !y || !valid_alpha(filter->alpha_min) ||
        !valid_alpha(filter->alpha_max) || filter->alpha_min > filter->alpha_max ||
        filter->residual_scale <= 0.0f)
    {
        return -1;
    }

    residual = fabsf(x - filter->y);
    ratio    = residual / (residual + filter->residual_scale);
    a        = filter->alpha_min + (filter->alpha_max - filter->alpha_min) * ratio;
    filter->y += a * (x - filter->y);

    if (alpha)
    {
        *alpha = a;
    }
    *y = filter->y;

    return 0;
}

int kf_signal_running_stats1f_init(kf_signal_running_stats1f* stats)
{
    if (!stats)
    {
        return -1;
    }

    stats->count = 0;
    stats->mean  = 0.0f;
    stats->m2    = 0.0f;

    return 0;
}

int kf_signal_running_stats1f_update(kf_signal_running_stats1f* stats, float x)
{
    float delta;
    float delta2;

    if (!stats)
    {
        return -1;
    }

    ++stats->count;
    delta = x - stats->mean;
    stats->mean += delta / (float)stats->count;
    delta2 = x - stats->mean;
    stats->m2 += delta * delta2;

    return 0;
}

int kf_signal_running_stats1f_get(const kf_signal_running_stats1f* stats, float* mean,
                                  float* variance_population, float* variance_sample,
                                  float* stddev_population, float* stddev_sample)
{
    float var_pop;
    float var_sample = 0.0f;

    if (!stats || stats->count <= 0)
    {
        return -1;
    }

    var_pop = stats->m2 / (float)stats->count;
    if (stats->count > 1)
    {
        var_sample = stats->m2 / (float)(stats->count - 1);
    }

    if (mean)
    {
        *mean = stats->mean;
    }
    if (variance_population)
    {
        *variance_population = var_pop;
    }
    if (variance_sample)
    {
        *variance_sample = var_sample;
    }
    if (stddev_population)
    {
        *stddev_population = sqrtf(var_pop);
    }
    if (stddev_sample)
    {
        *stddev_sample = sqrtf(var_sample);
    }

    return 0;
}

int kf_signal_sos1f_init(kf_signal_sos1f* filter, kf_signal_biquad1f* sections, int count)
{
    if (!filter || !sections || count <= 0)
    {
        return -1;
    }

    for (int i = 0; i < count; ++i)
    {
        if (!sections[i].initialized)
        {
            return -1;
        }
    }

    filter->sections = sections;
    filter->count    = count;

    return 0;
}

int kf_signal_sos1f_update(kf_signal_sos1f* filter, float x, float* y)
{
    float value = x;

    if (!filter || !filter->sections || filter->count <= 0 || !y)
    {
        return -1;
    }

    for (int i = 0; i < filter->count; ++i)
    {
        if (kf_signal_biquad1f_update(&filter->sections[i], value, &value) != 0)
        {
            return -1;
        }
    }

    *y = value;

    return 0;
}

int kf_signal_dc_blocker1f_init(kf_signal_dc_blocker1f* filter, float r, float initial_input,
                                float initial_output)
{
    if (!filter || r < 0.0f || r > 1.0f)
    {
        return -1;
    }

    filter->r           = r;
    filter->prev_x      = initial_input;
    filter->y           = initial_output;
    filter->initialized = 1;

    return 0;
}

int kf_signal_dc_blocker1f_update(kf_signal_dc_blocker1f* filter, float x, float* y)
{
    if (!filter || !filter->initialized || !y || filter->r < 0.0f || filter->r > 1.0f)
    {
        return -1;
    }

    filter->y      = x - filter->prev_x + filter->r * filter->y;
    filter->prev_x = x;
    *y             = filter->y;

    return 0;
}

int kf_signal_quantile1f_init(kf_signal_quantile1f* filter, float* window, float* scratch,
                              int window_size)
{
    if (!filter || !window || !scratch || window_size <= 0)
    {
        return -1;
    }

    for (int i = 0; i < window_size; ++i)
    {
        window[i]  = 0.0f;
        scratch[i] = 0.0f;
    }

    filter->window      = window;
    filter->scratch     = scratch;
    filter->window_size = window_size;
    filter->index       = 0;
    filter->count       = 0;

    return 0;
}

int kf_signal_quantile1f_update(kf_signal_quantile1f* filter, float x, float quantile, float* y)
{
    if (!filter || !filter->window || !filter->scratch || !y || filter->window_size <= 0 ||
        quantile < 0.0f || quantile > 1.0f)
    {
        return -1;
    }

    filter->window[filter->index] = x;
    if (filter->count < filter->window_size)
    {
        ++filter->count;
    }
    ++filter->index;
    if (filter->index >= filter->window_size)
    {
        filter->index = 0;
    }

    for (int i = 0; i < filter->count; ++i)
    {
        filter->scratch[i] = filter->window[i];
    }
    insertion_sort(filter->scratch, filter->count);
    *y = quantile_of_sorted(filter->scratch, filter->count, quantile);

    return 0;
}

int kf_signal_mad_noise_sigma(const float* x, float* scratch, int n, float* median, float* sigma)
{
    float med;

    if (!x || !scratch || !median || !sigma || n <= 0)
    {
        return -1;
    }

    med = median_of_values(scratch, x, n);
    for (int i = 0; i < n; ++i)
    {
        scratch[i] = fabsf(x[i] - med);
    }
    insertion_sort(scratch, n);

    *median = med;
    *sigma  = 1.4826f * quantile_of_sorted(scratch, n, 0.5f);

    return 0;
}

int kf_signal_zscore_gate(float x, float mean, float stddev, float threshold, float* zscore)
{
    float z;

    if (stddev <= 0.0f || threshold < 0.0f)
    {
        return -1;
    }

    z = fabsf((x - mean) / stddev);
    if (zscore)
    {
        *zscore = z;
    }

    return z <= threshold ? 1 : 0;
}

int kf_signal_iqr_gate(float x, const float* values, float* scratch, int n, float multiplier,
                       float* q1, float* q3)
{
    float q1_local;
    float q3_local;
    float iqr;

    if (!values || !scratch || n <= 0 || multiplier < 0.0f)
    {
        return -1;
    }

    for (int i = 0; i < n; ++i)
    {
        scratch[i] = values[i];
    }
    insertion_sort(scratch, n);

    q1_local = quantile_of_sorted(scratch, n, 0.25f);
    q3_local = quantile_of_sorted(scratch, n, 0.75f);
    iqr      = q3_local - q1_local;

    if (q1)
    {
        *q1 = q1_local;
    }
    if (q3)
    {
        *q3 = q3_local;
    }

    return (x >= q1_local - multiplier * iqr && x <= q3_local + multiplier * iqr) ? 1 : 0;
}

int kf_signal_huber_weight(float residual, float delta, float* weight)
{
    const float abs_residual = fabsf(residual);

    if (!weight || delta <= 0.0f)
    {
        return -1;
    }

    *weight = abs_residual <= delta ? 1.0f : delta / abs_residual;

    return 0;
}

int kf_signal_tukey_weight(float residual, float c, float* weight)
{
    float r;
    float value;

    if (!weight || c <= 0.0f)
    {
        return -1;
    }

    r = residual / c;
    if (fabsf(r) >= 1.0f)
    {
        *weight = 0.0f;
    }
    else
    {
        value   = 1.0f - r * r;
        *weight = value * value;
    }

    return 0;
}

int kf_signal_schmitt_trigger(float x, float low_threshold, float high_threshold, int previous_state,
                              int* state)
{
    if (!state || low_threshold > high_threshold)
    {
        return -1;
    }

    *state = previous_state ? 1 : 0;
    if (x >= high_threshold)
    {
        *state = 1;
    }
    else if (x <= low_threshold)
    {
        *state = 0;
    }

    return 0;
}

int kf_signal_edge1f_init(kf_signal_edge1f* filter, int initial_state)
{
    if (!filter)
    {
        return -1;
    }

    filter->previous_state = initial_state ? 1 : 0;
    filter->initialized    = 1;

    return 0;
}

int kf_signal_edge1f_update(kf_signal_edge1f* filter, int state, int* rising, int* falling)
{
    const int normalized_state = state ? 1 : 0;

    if (!filter || !filter->initialized)
    {
        return -1;
    }

    if (rising)
    {
        *rising = !filter->previous_state && normalized_state;
    }
    if (falling)
    {
        *falling = filter->previous_state && !normalized_state;
    }
    filter->previous_state = normalized_state;

    return 0;
}

int kf_signal_sample_hold1f_init(kf_signal_sample_hold1f* filter, float initial_value,
                                 int max_hold_samples)
{
    if (!filter)
    {
        return -1;
    }

    filter->value            = initial_value;
    filter->age              = 0;
    filter->max_hold_samples = max_hold_samples;
    filter->initialized      = 1;

    return 0;
}

int kf_signal_sample_hold1f_update(kf_signal_sample_hold1f* filter, float x, int valid, float* y)
{
    if (!filter || !filter->initialized || !y)
    {
        return -1;
    }

    if (valid)
    {
        filter->value = x;
        filter->age   = 0;
        *y            = filter->value;
        return 1;
    }

    ++filter->age;
    *y = filter->value;

    if (filter->max_hold_samples >= 0 && filter->age > filter->max_hold_samples)
    {
        return -1;
    }

    return 0;
}

int kf_signal_moving_minmax1f_init(kf_signal_moving_minmax1f* filter, float* window,
                                   float* scratch, int window_size)
{
    if (!filter || !window || !scratch || window_size <= 0)
    {
        return -1;
    }

    for (int i = 0; i < window_size; ++i)
    {
        window[i]  = 0.0f;
        scratch[i] = 0.0f;
    }

    filter->window      = window;
    filter->scratch     = scratch;
    filter->window_size = window_size;
    filter->index       = 0;
    filter->count       = 0;

    return 0;
}

int kf_signal_moving_minmax1f_update(kf_signal_moving_minmax1f* filter, float x,
                                     float* min_value, float* max_value)
{
    float min_local;
    float max_local;

    if (!filter || !filter->window || filter->window_size <= 0 || (!min_value && !max_value))
    {
        return -1;
    }

    filter->window[filter->index] = x;
    if (filter->count < filter->window_size)
    {
        ++filter->count;
    }
    ++filter->index;
    if (filter->index >= filter->window_size)
    {
        filter->index = 0;
    }

    min_local = filter->window[0];
    max_local = filter->window[0];
    for (int i = 1; i < filter->count; ++i)
    {
        if (filter->window[i] < min_local)
        {
            min_local = filter->window[i];
        }
        if (filter->window[i] > max_local)
        {
            max_local = filter->window[i];
        }
    }

    if (min_value)
    {
        *min_value = min_local;
    }
    if (max_value)
    {
        *max_value = max_local;
    }

    return 0;
}

int kf_signal_ew_stats1f_init(kf_signal_ew_stats1f* stats, float alpha, float initial_mean,
                              float initial_variance)
{
    if (!stats || !valid_alpha(alpha) || initial_variance < 0.0f)
    {
        return -1;
    }

    stats->alpha       = alpha;
    stats->mean        = initial_mean;
    stats->variance    = initial_variance;
    stats->initialized = 1;

    return 0;
}

int kf_signal_ew_stats1f_update(kf_signal_ew_stats1f* stats, float x, float* mean,
                                float* variance)
{
    float delta;
    float new_mean;

    if (!stats || !stats->initialized || !valid_alpha(stats->alpha))
    {
        return -1;
    }

    delta       = x - stats->mean;
    new_mean    = stats->mean + stats->alpha * delta;
    stats->variance = (1.0f - stats->alpha) * (stats->variance + stats->alpha * delta * delta);
    stats->mean     = new_mean;

    if (mean)
    {
        *mean = stats->mean;
    }
    if (variance)
    {
        *variance = stats->variance;
    }

    return 0;
}

int kf_signal_peak_hold1f_init(kf_signal_peak_hold1f* filter, float initial_peak,
                               float decay_per_sample)
{
    if (!filter || decay_per_sample < 0.0f)
    {
        return -1;
    }

    filter->peak             = initial_peak;
    filter->decay_per_sample = decay_per_sample;
    filter->initialized      = 1;

    return 0;
}

int kf_signal_peak_hold1f_update(kf_signal_peak_hold1f* filter, float x, float* peak)
{
    if (!filter || !filter->initialized || !peak || filter->decay_per_sample < 0.0f)
    {
        return -1;
    }

    filter->peak -= filter->decay_per_sample;
    if (x > filter->peak)
    {
        filter->peak = x;
    }

    *peak = filter->peak;

    return 0;
}

int kf_signal_majority1f_init(kf_signal_majority1f* filter, int* window, int window_size,
                              int initial_state)
{
    const int state = initial_state ? 1 : 0;

    if (!filter || !window || window_size <= 0)
    {
        return -1;
    }

    for (int i = 0; i < window_size; ++i)
    {
        window[i] = state;
    }

    filter->window      = window;
    filter->window_size = window_size;
    filter->index       = 0;
    filter->count       = window_size;
    filter->ones        = state ? window_size : 0;

    return 0;
}

int kf_signal_majority1f_update(kf_signal_majority1f* filter, int input_state, int* output_state)
{
    const int state = input_state ? 1 : 0;

    if (!filter || !filter->window || filter->window_size <= 0 || !output_state)
    {
        return -1;
    }

    filter->ones -= filter->window[filter->index] ? 1 : 0;
    filter->window[filter->index] = state;
    filter->ones += state;

    ++filter->index;
    if (filter->index >= filter->window_size)
    {
        filter->index = 0;
    }

    *output_state = filter->ones * 2 >= filter->window_size ? 1 : 0;

    return 0;
}

int kf_signal_jerk_limiter1f_init(kf_signal_jerk_limiter1f* filter, float max_rate,
                                  float max_accel, float max_jerk, float dt, float initial_value,
                                  float initial_rate, float initial_accel)
{
    if (!filter || max_rate < 0.0f || max_accel < 0.0f || max_jerk < 0.0f || dt <= 0.0f)
    {
        return -1;
    }

    filter->max_rate    = max_rate;
    filter->max_accel   = max_accel;
    filter->max_jerk    = max_jerk;
    filter->dt          = dt;
    filter->value       = initial_value;
    filter->rate        = initial_rate;
    filter->accel       = initial_accel;
    filter->initialized = 1;

    return 0;
}

int kf_signal_jerk_limiter1f_update(kf_signal_jerk_limiter1f* filter, float target, float* y,
                                    float* rate, float* accel)
{
    float desired_rate;
    float desired_accel;

    if (!filter || !filter->initialized || !y || !rate || !accel || filter->max_rate < 0.0f ||
        filter->max_accel < 0.0f || filter->max_jerk < 0.0f || filter->dt <= 0.0f)
    {
        return -1;
    }

    desired_rate = (target - filter->value) / filter->dt;
    if (kf_signal_clamp1f(desired_rate, -filter->max_rate, filter->max_rate, &desired_rate) != 0)
    {
        return -1;
    }
    desired_accel = (desired_rate - filter->rate) / filter->dt;
    if (kf_signal_clamp1f(desired_accel, -filter->max_accel, filter->max_accel, &desired_accel) !=
        0)
    {
        return -1;
    }
    if (kf_signal_slew_rate_limit1f(filter->accel, desired_accel, filter->max_jerk * filter->dt,
                                    &filter->accel) != 0)
    {
        return -1;
    }

    filter->rate += filter->accel * filter->dt;
    if (kf_signal_clamp1f(filter->rate, -filter->max_rate, filter->max_rate, &filter->rate) != 0)
    {
        return -1;
    }
    filter->value += filter->rate * filter->dt;

    if ((filter->rate > 0.0f && filter->value > target) ||
        (filter->rate < 0.0f && filter->value < target))
    {
        filter->value = target;
        filter->rate  = 0.0f;
        filter->accel = 0.0f;
    }

    *y     = filter->value;
    *rate  = filter->rate;
    *accel = filter->accel;

    return 0;
}

int kf_signal_dropout_decay1f_init(kf_signal_dropout_decay1f* filter, float initial_value,
                                   float fallback, float decay_alpha)
{
    if (!filter || !valid_alpha(decay_alpha))
    {
        return -1;
    }

    filter->value       = initial_value;
    filter->fallback    = fallback;
    filter->decay_alpha = decay_alpha;
    filter->initialized = 1;

    return 0;
}

int kf_signal_dropout_decay1f_update(kf_signal_dropout_decay1f* filter, float x, int valid,
                                     float* y)
{
    if (!filter || !filter->initialized || !y || !valid_alpha(filter->decay_alpha))
    {
        return -1;
    }

    if (valid)
    {
        filter->value = x;
        *y            = filter->value;
        return 1;
    }

    filter->value += filter->decay_alpha * (filter->fallback - filter->value);
    *y = filter->value;

    return 0;
}

int kf_signal_mahalanobis_distance_sq(const float* residual, const float* covariance,
                                      float* covariance_work, float* residual_work, int n,
                                      float* distance_sq)
{
    float sum = 0.0f;

    if (!residual || !covariance || !covariance_work || !residual_work || !distance_sq || n <= 0)
    {
        return -1;
    }

    for (int i = 0; i < n * n; ++i)
    {
        covariance_work[i] = covariance[i];
    }
    for (int i = 0; i < n; ++i)
    {
        residual_work[i] = residual[i];
    }

    if (cholesky(covariance_work, n, 0) != 0)
    {
        return -1;
    }
    trisolve(covariance_work, residual_work, n, 1, "N");

    for (int i = 0; i < n; ++i)
    {
        sum += residual_work[i] * residual_work[i];
    }
    *distance_sq = sum;

    return 0;
}

int kf_signal_mahalanobis_gate(const float* residual, const float* covariance,
                               float* covariance_work, float* residual_work, int n,
                               float threshold_sq, float* distance_sq)
{
    float d2;

    if (threshold_sq < 0.0f)
    {
        return -1;
    }

    if (kf_signal_mahalanobis_distance_sq(residual, covariance, covariance_work, residual_work, n,
                                          &d2) != 0)
    {
        return -1;
    }

    if (distance_sq)
    {
        *distance_sq = d2;
    }

    return d2 <= threshold_sq ? 1 : 0;
}

int kf_signal_euclidean_norm(const float* x, int n, float* norm)
{
    if (!x || !norm || n <= 0)
    {
        return -1;
    }

    *norm = vecnorm(x, n);

    return 0;
}

int kf_signal_euclidean_distance(const float* a, const float* b, int n, float* distance)
{
    float sum = 0.0f;

    if (!a || !b || !distance || n <= 0)
    {
        return -1;
    }

    for (int i = 0; i < n; ++i)
    {
        const float diff = a[i] - b[i];
        sum += diff * diff;
    }

    *distance = sqrtf(sum);

    return 0;
}

int kf_signal_euclidean_filter(float* state, const float* sample, int n, float max_distance,
                               float alpha, float* distance)
{
    float d;

    if (!state || !sample || n <= 0 || max_distance < 0.0f || !valid_alpha(alpha))
    {
        return -1;
    }

    if (kf_signal_euclidean_distance(state, sample, n, &d) != 0)
    {
        return -1;
    }

    if (distance)
    {
        *distance = d;
    }

    if (max_distance > 0.0f && d > max_distance)
    {
        return 0;
    }

    for (int i = 0; i < n; ++i)
    {
        state[i] += alpha * (sample[i] - state[i]);
    }

    return 1;
}

static int valid_alpha(float alpha)
{
    return alpha >= 0.0f && alpha <= 1.0f;
}

static void insertion_sort(float* values, int n)
{
    for (int i = 1; i < n; ++i)
    {
        const float key = values[i];
        int         j   = i - 1;

        while (j >= 0 && values[j] > key)
        {
            values[j + 1] = values[j];
            --j;
        }
        values[j + 1] = key;
    }
}

static float median_of_values(float* scratch, const float* values, int n)
{
    for (int i = 0; i < n; ++i)
    {
        scratch[i] = values[i];
    }

    insertion_sort(scratch, n);
    if ((n % 2) != 0)
    {
        return scratch[n / 2];
    }

    return 0.5f * (scratch[(n / 2) - 1] + scratch[n / 2]);
}

static float quantile_of_sorted(const float* sorted, int n, float quantile)
{
    const float pos   = quantile * (float)(n - 1);
    const int   lower = (int)floorf(pos);
    const int   upper = lower + 1 < n ? lower + 1 : lower;
    const float frac  = pos - (float)lower;

    return sorted[lower] + frac * (sorted[upper] - sorted[lower]);
}

static int valid_biquad_frequency(float sample_rate_hz, float frequency_hz, float q)
{
    return sample_rate_hz > 0.0f && frequency_hz > 0.0f && frequency_hz < sample_rate_hz * 0.5f &&
           q > 0.0f;
}

static void normalize_biquad(float b0, float b1, float b2, float a0, float a1, float a2,
                             kf_signal_biquad_coeffs* coeffs)
{
    coeffs->b0 = b0 / a0;
    coeffs->b1 = b1 / a0;
    coeffs->b2 = b2 / a0;
    coeffs->a1 = a1 / a0;
    coeffs->a2 = a2 / a0;
}

/* @} */
