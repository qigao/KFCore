/** @file signal_filters.h
 * KFCore
 *
 * @brief Lightweight scalar and vector signal filters.
 * @{ */

#ifndef SIGNAL_FILTERS_H
#define SIGNAL_FILTERS_H

#ifdef __cplusplus
extern "C" {
#endif

/******************************************************************************
 * TYPEDEFS
 ******************************************************************************/

/** @brief First-order scalar low-pass filter state. */
typedef struct kf_signal_lowpass1f
{
    float alpha;       /**< Smoothing factor in [0, 1]. */
    float y;           /**< Current filtered output. */
    int   initialized; /**< Non-zero after initialization. */
} kf_signal_lowpass1f;

/** @brief First-order scalar high-pass filter state. */
typedef struct kf_signal_highpass1f
{
    float alpha;       /**< Smoothing factor in [0, 1]. */
    float prev_x;      /**< Previous input sample. */
    float y;           /**< Current filtered output. */
    int   initialized; /**< Non-zero after initialization. */
} kf_signal_highpass1f;

/** @brief Scalar moving-average filter state. */
typedef struct kf_signal_moving_average1f
{
    float* window;      /**< Caller supplied window buffer. */
    int    window_size; /**< Number of samples in window. */
    int    index;       /**< Next write index. */
    int    count;       /**< Number of valid samples, up to window_size. */
    float  sum;         /**< Running sum of valid samples. */
} kf_signal_moving_average1f;

/** @brief Scalar median filter state. */
typedef struct kf_signal_median1f
{
    float* window;      /**< Caller supplied window buffer. */
    float* scratch;     /**< Caller supplied scratch buffer, same length as window. */
    int    window_size; /**< Number of samples in window. */
    int    index;       /**< Next write index. */
    int    count;       /**< Number of valid samples, up to window_size. */
} kf_signal_median1f;

/** @brief Scalar moving RMS filter state. */
typedef struct kf_signal_moving_rms1f
{
    float* window;
    int    window_size;
    int    index;
    int    count;
    float  sum_sq;
} kf_signal_moving_rms1f;

/** @brief Scalar trimmed-mean filter state. */
typedef struct kf_signal_trimmed_mean1f
{
    float* window;
    float* scratch;
    int    window_size;
    int    trim_each_side;
    int    index;
    int    count;
} kf_signal_trimmed_mean1f;

/** @brief Scalar winsorized-mean filter state. */
typedef struct kf_signal_winsorized_mean1f
{
    float* window;
    float* scratch;
    int    window_size;
    int    winsor_each_side;
    int    index;
    int    count;
} kf_signal_winsorized_mean1f;

/** @brief Scalar Savitzky-Golay FIR smoother state.
 *
 * coeffs[0] applies to the newest sample, coeffs[window_size - 1] to the oldest.
 */
typedef struct kf_signal_savgol1f
{
    const float* coeffs;
    float*       window;
    int          window_size;
    int          index;
    int          count;
} kf_signal_savgol1f;

/** @brief Scalar Hampel filter state. */
typedef struct kf_signal_hampel1f
{
    float* window;      /**< Caller supplied window buffer. */
    float* scratch;     /**< Caller supplied scratch buffer, same length as window. */
    int    window_size; /**< Number of samples in window. */
    int    index;       /**< Next write index. */
    int    count;       /**< Number of valid samples, up to window_size. */
} kf_signal_hampel1f;

/** @brief Direct Form I biquad filter coefficients. */
typedef struct kf_signal_biquad_coeffs
{
    float b0;
    float b1;
    float b2;
    float a1;
    float a2;
} kf_signal_biquad_coeffs;

/** @brief Direct Form I biquad filter state. */
typedef struct kf_signal_biquad1f
{
    kf_signal_biquad_coeffs coeffs;
    float                   x1;
    float                   x2;
    float                   y1;
    float                   y2;
    int                     initialized;
} kf_signal_biquad1f;

/** @brief Scalar FIR filter state.
 *
 * coeffs[0] applies to the newest sample, coeffs[taps - 1] to the oldest.
 */
typedef struct kf_signal_fir1f
{
    const float* coeffs; /**< Caller supplied coefficient buffer. */
    float*       state;  /**< Caller supplied sample history buffer. */
    int          taps;   /**< Number of coefficients and state samples. */
    int          index;  /**< Next write index. */
} kf_signal_fir1f;

/** @brief Scalar direct-form I IIR filter state.
 *
 * a[0] must be non-zero. Feedback coefficients a[1..na-1] are applied with a
 * negative sign: y = (sum(b*x_history) - sum(a[i]*y_history[i-1])) / a[0].
 */
typedef struct kf_signal_iir1f
{
    const float* b;       /**< Feed-forward coefficients. */
    const float* a;       /**< Feedback coefficients, including a[0]. */
    float*       x_state; /**< Caller supplied nb sample history buffer. */
    float*       y_state; /**< Caller supplied na - 1 output history buffer. */
    int          nb;      /**< Number of feed-forward coefficients. */
    int          na;      /**< Number of feedback coefficients, including a[0]. */
} kf_signal_iir1f;

/** @brief Scalar alpha-beta tracking filter state. */
typedef struct kf_signal_alpha_beta1f
{
    float alpha;
    float beta;
    float dt;
    float position;
    float velocity;
    int   initialized;
} kf_signal_alpha_beta1f;

/** @brief Scalar alpha-beta-gamma tracking filter state. */
typedef struct kf_signal_alpha_beta_gamma1f
{
    float alpha;
    float beta;
    float gamma;
    float dt;
    float position;
    float velocity;
    float acceleration;
    int   initialized;
} kf_signal_alpha_beta_gamma1f;

/** @brief Scalar complementary filter state. */
typedef struct kf_signal_complementary1f
{
    float alpha;
    float dt;
    float value;
    int   initialized;
} kf_signal_complementary1f;

/** @brief Scalar hysteresis threshold state. */
typedef struct kf_signal_hysteresis1f
{
    float low_threshold;
    float high_threshold;
    int   state;
    int   initialized;
} kf_signal_hysteresis1f;

/** @brief Boolean debounce filter state. */
typedef struct kf_signal_debounce
{
    int stable_state;
    int candidate_state;
    int counter;
    int threshold_samples;
    int initialized;
} kf_signal_debounce;

/** @brief Stateful scalar rate limiter. */
typedef struct kf_signal_rate_limiter1f
{
    float max_rate;
    float dt;
    float value;
    int   initialized;
} kf_signal_rate_limiter1f;

/** @brief Stateful scalar acceleration limiter. */
typedef struct kf_signal_accel_limiter1f
{
    float max_rate;
    float max_accel;
    float dt;
    float value;
    float rate;
    int   initialized;
} kf_signal_accel_limiter1f;

/** @brief Scalar One Euro adaptive low-pass filter state. */
typedef struct kf_signal_one_euro1f
{
    float min_cutoff_hz;
    float beta;
    float derivative_cutoff_hz;
    float dt;
    float x;
    float dx;
    int   initialized;
} kf_signal_one_euro1f;

/** @brief Adaptive EMA filter state. */
typedef struct kf_signal_adaptive_ema1f
{
    float alpha_min;
    float alpha_max;
    float residual_scale;
    float y;
    int   initialized;
} kf_signal_adaptive_ema1f;

/** @brief Online running statistics state using Welford's algorithm. */
typedef struct kf_signal_running_stats1f
{
    int   count;
    float mean;
    float m2;
} kf_signal_running_stats1f;

/** @brief Cascaded biquad/SOS filter state. */
typedef struct kf_signal_sos1f
{
    kf_signal_biquad1f* sections;
    int                 count;
} kf_signal_sos1f;

/** @brief Scalar DC blocker state. */
typedef struct kf_signal_dc_blocker1f
{
    float r;
    float prev_x;
    float y;
    int   initialized;
} kf_signal_dc_blocker1f;

/** @brief Sliding-window quantile/percentile filter state. */
typedef struct kf_signal_quantile1f
{
    float* window;
    float* scratch;
    int    window_size;
    int    index;
    int    count;
} kf_signal_quantile1f;

/** @brief Scalar edge detector state. */
typedef struct kf_signal_edge1f
{
    int previous_state;
    int initialized;
} kf_signal_edge1f;

/** @brief Scalar sample-and-hold/dropout state. */
typedef struct kf_signal_sample_hold1f
{
    float value;
    int   age;
    int   max_hold_samples;
    int   initialized;
} kf_signal_sample_hold1f;

/** @brief Sliding-window min/max filter state. */
typedef struct kf_signal_moving_minmax1f
{
    float* window;
    float* scratch;
    int    window_size;
    int    index;
    int    count;
} kf_signal_moving_minmax1f;

/** @brief Exponentially weighted running mean/variance state. */
typedef struct kf_signal_ew_stats1f
{
    float alpha;
    float mean;
    float variance;
    int   initialized;
} kf_signal_ew_stats1f;

/** @brief Scalar peak-hold filter with linear decay. */
typedef struct kf_signal_peak_hold1f
{
    float peak;
    float decay_per_sample;
    int   initialized;
} kf_signal_peak_hold1f;

/** @brief Boolean majority-vote filter state. */
typedef struct kf_signal_majority1f
{
    int* window;
    int  window_size;
    int  index;
    int  count;
    int  ones;
} kf_signal_majority1f;

/** @brief Stateful scalar jerk limiter. */
typedef struct kf_signal_jerk_limiter1f
{
    float max_rate;
    float max_accel;
    float max_jerk;
    float dt;
    float value;
    float rate;
    float accel;
    int   initialized;
} kf_signal_jerk_limiter1f;

/** @brief Dropout filter that decays held values toward a fallback. */
typedef struct kf_signal_dropout_decay1f
{
    float value;
    float fallback;
    float decay_alpha;
    int   initialized;
} kf_signal_dropout_decay1f;

/******************************************************************************
 * FUNCTION PROTOTYPES
 ******************************************************************************/

/** @brief Compute the first-order low-pass alpha from cutoff frequency and sample period.
 *
 * Uses alpha = dt / (RC + dt), where RC = 1 / (2*pi*cutoff_hz).
 *
 * @return 0 on success, -1 on invalid input.
 */
int kf_signal_lowpass_alpha(float cutoff_hz, float sample_period_s, float* alpha);

/** @brief Compute the first-order high-pass alpha from cutoff frequency and sample period.
 *
 * Uses alpha = RC / (RC + dt), where RC = 1 / (2*pi*cutoff_hz).
 *
 * @return 0 on success, -1 on invalid input.
 */
int kf_signal_highpass_alpha(float cutoff_hz, float sample_period_s, float* alpha);

/** @brief Initialize a scalar first-order low-pass filter.
 *
 * @return 0 on success, -1 on invalid input.
 */
int kf_signal_lowpass1f_init(kf_signal_lowpass1f* filter, float alpha, float initial_output);

/** @brief Update a scalar first-order low-pass filter.
 *
 * Computes y = y + alpha * (x - y).
 *
 * @return 0 on success, -1 on invalid input.
 */
int kf_signal_lowpass1f_update(kf_signal_lowpass1f* filter, float x, float* y);

/** @brief Apply one vector low-pass update in place.
 *
 * Each output element is updated as y[i] = y[i] + alpha * (x[i] - y[i]).
 *
 * @return 0 on success, -1 on invalid input.
 */
int kf_signal_lowpass_vector(float* y, const float* x, int n, float alpha);

/** @brief Initialize a scalar first-order high-pass filter.
 *
 * @return 0 on success, -1 on invalid input.
 */
int kf_signal_highpass1f_init(kf_signal_highpass1f* filter, float alpha, float initial_input,
                              float initial_output);

/** @brief Update a scalar first-order high-pass filter.
 *
 * Computes y = alpha * (prev_y + x - prev_x).
 *
 * @return 0 on success, -1 on invalid input.
 */
int kf_signal_highpass1f_update(kf_signal_highpass1f* filter, float x, float* y);

/** @brief Apply one vector high-pass update in place.
 *
 * Each output element is updated as y[i] = alpha * (y[i] + x[i] - prev_x[i]).
 * prev_x is then overwritten with x.
 *
 * @return 0 on success, -1 on invalid input.
 */
int kf_signal_highpass_vector(float* y, float* prev_x, const float* x, int n, float alpha);

/** @brief Initialize a scalar moving-average filter.
 *
 * @param[in,out] filter Filter state.
 * @param[in,out] window Caller supplied buffer with window_size elements.
 * @param[in] window_size Number of elements in window.
 *
 * @return 0 on success, -1 on invalid input.
 */
int kf_signal_moving_average1f_init(kf_signal_moving_average1f* filter, float* window,
                                    int window_size);

/** @brief Update a scalar moving-average filter.
 *
 * Averages the available samples until the window is full, then maintains a
 * rolling average over window_size samples.
 *
 * @return 0 on success, -1 on invalid input.
 */
int kf_signal_moving_average1f_update(kf_signal_moving_average1f* filter, float x, float* y);

/** @brief Initialize a scalar median filter.
 *
 * @param[in,out] filter Filter state.
 * @param[in,out] window Caller supplied buffer with window_size elements.
 * @param[in,out] scratch Caller supplied scratch buffer with window_size elements.
 * @param[in] window_size Number of elements in window and scratch.
 *
 * @return 0 on success, -1 on invalid input.
 */
int kf_signal_median1f_init(kf_signal_median1f* filter, float* window, float* scratch,
                            int window_size);

/** @brief Update a scalar median filter.
 *
 * Uses the median of the currently available samples until the window is full.
 *
 * @return 0 on success, -1 on invalid input.
 */
int kf_signal_median1f_update(kf_signal_median1f* filter, float x, float* y);

/** @brief Initialize a scalar moving RMS filter.
 *
 * @return 0 on success, -1 on invalid input.
 */
int kf_signal_moving_rms1f_init(kf_signal_moving_rms1f* filter, float* window, int window_size);

/** @brief Update a scalar moving RMS filter.
 *
 * Uses available samples until the window is full.
 *
 * @return 0 on success, -1 on invalid input.
 */
int kf_signal_moving_rms1f_update(kf_signal_moving_rms1f* filter, float x, float* y);

/** @brief Initialize a scalar trimmed-mean filter.
 *
 * Drops trim_each_side smallest and largest samples before averaging.
 *
 * @return 0 on success, -1 on invalid input.
 */
int kf_signal_trimmed_mean1f_init(kf_signal_trimmed_mean1f* filter, float* window,
                                  float* scratch, int window_size, int trim_each_side);

/** @brief Update a scalar trimmed-mean filter.
 *
 * @return 0 on success, -1 on invalid input.
 */
int kf_signal_trimmed_mean1f_update(kf_signal_trimmed_mean1f* filter, float x, float* y);

/** @brief Initialize a scalar winsorized-mean filter.
 *
 * Replaces the lowest/highest winsor_each_side samples with boundary values
 * before averaging.
 *
 * @return 0 on success, -1 on invalid input.
 */
int kf_signal_winsorized_mean1f_init(kf_signal_winsorized_mean1f* filter, float* window,
                                     float* scratch, int window_size, int winsor_each_side);

/** @brief Update a scalar winsorized-mean filter.
 *
 * @return 0 on success, -1 on invalid input.
 */
int kf_signal_winsorized_mean1f_update(kf_signal_winsorized_mean1f* filter, float x, float* y);

/** @brief Initialize a coefficient-based Savitzky-Golay smoother.
 *
 * @return 0 on success, -1 on invalid input.
 */
int kf_signal_savgol1f_init(kf_signal_savgol1f* filter, const float* coeffs, float* window,
                            int window_size);

/** @brief Update a coefficient-based Savitzky-Golay smoother.
 *
 * Until the window is full, unavailable older samples are treated as zero.
 *
 * @return 0 on success, -1 on invalid input.
 */
int kf_signal_savgol1f_update(kf_signal_savgol1f* filter, float x, float* y);

/** @brief Clamp a scalar value to [min_value, max_value].
 *
 * @return 0 on success, -1 on invalid input.
 */
int kf_signal_clamp1f(float x, float min_value, float max_value, float* y);

/** @brief Apply symmetric deadband around zero.
 *
 * Values with absolute magnitude <= deadband are returned as 0. Values outside
 * the deadband have the deadband subtracted from their magnitude.
 *
 * @return 0 on success, -1 on invalid input.
 */
int kf_signal_deadband1f(float x, float deadband, float* y);

/** @brief Limit the per-update change from previous value to target.
 *
 * @return 0 on success, -1 on invalid input.
 */
int kf_signal_slew_rate_limit1f(float previous, float target, float max_delta, float* y);

/** @brief Initialize a scalar Hampel outlier filter.
 *
 * @param[in,out] filter Filter state.
 * @param[in,out] window Caller supplied buffer with window_size elements.
 * @param[in,out] scratch Caller supplied scratch buffer with window_size elements.
 * @param[in] window_size Number of elements in window and scratch.
 *
 * @return 0 on success, -1 on invalid input.
 */
int kf_signal_hampel1f_init(kf_signal_hampel1f* filter, float* window, float* scratch,
                            int window_size);

/** @brief Update a scalar Hampel outlier filter.
 *
 * The new sample is compared with the previous window median. If the difference
 * is greater than n_sigma * 1.4826 * MAD, the output is replaced by the median
 * and 0 is returned. Otherwise the sample is accepted and 1 is returned. When
 * there are not enough previous samples, the sample is accepted.
 *
 * @return 1 if accepted, 0 if replaced, -1 on invalid input.
 */
int kf_signal_hampel1f_update(kf_signal_hampel1f* filter, float x, float n_sigma, float* y);

/** @brief Initialize a scalar biquad filter with caller supplied coefficients.
 *
 * @return 0 on success, -1 on invalid input.
 */
int kf_signal_biquad1f_init(kf_signal_biquad1f* filter, const kf_signal_biquad_coeffs* coeffs);

/** @brief Reset biquad delay state.
 *
 * @return 0 on success, -1 on invalid input.
 */
int kf_signal_biquad1f_reset(kf_signal_biquad1f* filter, float x1, float x2, float y1, float y2);

/** @brief Update a scalar biquad filter.
 *
 * Computes y = b0*x + b1*x1 + b2*x2 - a1*y1 - a2*y2.
 *
 * @return 0 on success, -1 on invalid input.
 */
int kf_signal_biquad1f_update(kf_signal_biquad1f* filter, float x, float* y);

/** @brief Initialize a low-pass biquad filter from sample rate/cutoff/Q.
 *
 * @return 0 on success, -1 on invalid input.
 */
int kf_signal_biquad1f_init_lowpass(kf_signal_biquad1f* filter, float sample_rate_hz,
                                    float cutoff_hz, float q);

/** @brief Initialize a high-pass biquad filter from sample rate/cutoff/Q.
 *
 * @return 0 on success, -1 on invalid input.
 */
int kf_signal_biquad1f_init_highpass(kf_signal_biquad1f* filter, float sample_rate_hz,
                                     float cutoff_hz, float q);

/** @brief Initialize a band-pass biquad filter from sample rate/center/Q.
 *
 * @return 0 on success, -1 on invalid input.
 */
int kf_signal_biquad1f_init_bandpass(kf_signal_biquad1f* filter, float sample_rate_hz,
                                     float center_hz, float q);

/** @brief Initialize a notch biquad filter from sample rate/center/Q.
 *
 * @return 0 on success, -1 on invalid input.
 */
int kf_signal_biquad1f_init_notch(kf_signal_biquad1f* filter, float sample_rate_hz,
                                  float center_hz, float q);

/** @brief Design RBJ cookbook low-pass biquad coefficients.
 *
 * @return 0 on success, -1 on invalid input.
 */
int kf_signal_biquad_lowpass(float sample_rate_hz, float cutoff_hz, float q,
                             kf_signal_biquad_coeffs* coeffs);

/** @brief Design RBJ cookbook high-pass biquad coefficients.
 *
 * @return 0 on success, -1 on invalid input.
 */
int kf_signal_biquad_highpass(float sample_rate_hz, float cutoff_hz, float q,
                              kf_signal_biquad_coeffs* coeffs);

/** @brief Design RBJ cookbook band-pass biquad coefficients.
 *
 * Constant skirt gain form.
 *
 * @return 0 on success, -1 on invalid input.
 */
int kf_signal_biquad_bandpass(float sample_rate_hz, float center_hz, float q,
                              kf_signal_biquad_coeffs* coeffs);

/** @brief Design RBJ cookbook notch biquad coefficients.
 *
 * @return 0 on success, -1 on invalid input.
 */
int kf_signal_biquad_notch(float sample_rate_hz, float center_hz, float q,
                           kf_signal_biquad_coeffs* coeffs);

/** @brief Initialize a scalar FIR filter.
 *
 * @return 0 on success, -1 on invalid input.
 */
int kf_signal_fir1f_init(kf_signal_fir1f* filter, const float* coeffs, float* state, int taps);

/** @brief Update a scalar FIR filter.
 *
 * @return 0 on success, -1 on invalid input.
 */
int kf_signal_fir1f_update(kf_signal_fir1f* filter, float x, float* y);

/** @brief Initialize a scalar direct-form I IIR filter.
 *
 * @return 0 on success, -1 on invalid input.
 */
int kf_signal_iir1f_init(kf_signal_iir1f* filter, const float* b, int nb, const float* a, int na,
                         float* x_state, float* y_state);

/** @brief Update a scalar direct-form I IIR filter.
 *
 * @return 0 on success, -1 on invalid input.
 */
int kf_signal_iir1f_update(kf_signal_iir1f* filter, float x, float* y);

/** @brief Initialize a scalar alpha-beta tracking filter.
 *
 * @return 0 on success, -1 on invalid input.
 */
int kf_signal_alpha_beta1f_init(kf_signal_alpha_beta1f* filter, float alpha, float beta, float dt,
                                float initial_position, float initial_velocity);

/** @brief Update a scalar alpha-beta tracking filter with a position measurement.
 *
 * @return 0 on success, -1 on invalid input.
 */
int kf_signal_alpha_beta1f_update(kf_signal_alpha_beta1f* filter, float measurement,
                                  float* position, float* velocity);

/** @brief Initialize a scalar alpha-beta-gamma tracking filter.
 *
 * @return 0 on success, -1 on invalid input.
 */
int kf_signal_alpha_beta_gamma1f_init(kf_signal_alpha_beta_gamma1f* filter, float alpha,
                                      float beta, float gamma, float dt, float initial_position,
                                      float initial_velocity, float initial_acceleration);

/** @brief Update a scalar alpha-beta-gamma tracking filter with a position measurement.
 *
 * @return 0 on success, -1 on invalid input.
 */
int kf_signal_alpha_beta_gamma1f_update(kf_signal_alpha_beta_gamma1f* filter, float measurement,
                                        float* position, float* velocity, float* acceleration);

/** @brief Initialize a scalar complementary filter.
 *
 * @return 0 on success, -1 on invalid input.
 */
int kf_signal_complementary1f_init(kf_signal_complementary1f* filter, float alpha, float dt,
                                   float initial_value);

/** @brief Update a scalar complementary filter.
 *
 * Computes y = alpha * (previous + rate * dt) + (1 - alpha) * absolute.
 *
 * @return 0 on success, -1 on invalid input.
 */
int kf_signal_complementary1f_update(kf_signal_complementary1f* filter, float rate,
                                     float absolute, float* y);

/** @brief Initialize a scalar hysteresis threshold filter.
 *
 * @return 0 on success, -1 on invalid input.
 */
int kf_signal_hysteresis1f_init(kf_signal_hysteresis1f* filter, float low_threshold,
                                float high_threshold, int initial_state);

/** @brief Update a scalar hysteresis threshold filter.
 *
 * State switches to 1 when x >= high_threshold and to 0 when x <= low_threshold.
 *
 * @return 0 on success, -1 on invalid input.
 */
int kf_signal_hysteresis1f_update(kf_signal_hysteresis1f* filter, float x, int* state);

/** @brief Initialize a boolean debounce filter.
 *
 * @return 0 on success, -1 on invalid input.
 */
int kf_signal_debounce_init(kf_signal_debounce* filter, int threshold_samples, int initial_state);

/** @brief Update a boolean debounce filter.
 *
 * A new input state becomes stable after threshold_samples consecutive samples.
 *
 * @return 0 on success, -1 on invalid input.
 */
int kf_signal_debounce_update(kf_signal_debounce* filter, int input_state, int* stable_state);

/** @brief Initialize a stateful scalar rate limiter.
 *
 * @return 0 on success, -1 on invalid input.
 */
int kf_signal_rate_limiter1f_init(kf_signal_rate_limiter1f* filter, float max_rate, float dt,
                                  float initial_value);

/** @brief Update a stateful scalar rate limiter.
 *
 * @return 0 on success, -1 on invalid input.
 */
int kf_signal_rate_limiter1f_update(kf_signal_rate_limiter1f* filter, float target, float* y);

/** @brief Initialize a stateful scalar acceleration limiter.
 *
 * @return 0 on success, -1 on invalid input.
 */
int kf_signal_accel_limiter1f_init(kf_signal_accel_limiter1f* filter, float max_rate,
                                   float max_accel, float dt, float initial_value,
                                   float initial_rate);

/** @brief Update a stateful scalar acceleration limiter.
 *
 * @return 0 on success, -1 on invalid input.
 */
int kf_signal_accel_limiter1f_update(kf_signal_accel_limiter1f* filter, float target, float* y,
                                     float* rate);

/** @brief Initialize a scalar One Euro adaptive low-pass filter.
 *
 * @return 0 on success, -1 on invalid input.
 */
int kf_signal_one_euro1f_init(kf_signal_one_euro1f* filter, float min_cutoff_hz, float beta,
                              float derivative_cutoff_hz, float dt, float initial_value);

/** @brief Update a scalar One Euro adaptive low-pass filter.
 *
 * @return 0 on success, -1 on invalid input.
 */
int kf_signal_one_euro1f_update(kf_signal_one_euro1f* filter, float x, float* y);

/** @brief Initialize an adaptive EMA filter.
 *
 * alpha moves from alpha_min to alpha_max as abs(x - y) grows relative to
 * residual_scale.
 *
 * @return 0 on success, -1 on invalid input.
 */
int kf_signal_adaptive_ema1f_init(kf_signal_adaptive_ema1f* filter, float alpha_min,
                                  float alpha_max, float residual_scale, float initial_output);

/** @brief Update an adaptive EMA filter.
 *
 * @return 0 on success, -1 on invalid input.
 */
int kf_signal_adaptive_ema1f_update(kf_signal_adaptive_ema1f* filter, float x, float* y,
                                    float* alpha);

/** @brief Initialize/reset online running statistics.
 *
 * @return 0 on success, -1 on invalid input.
 */
int kf_signal_running_stats1f_init(kf_signal_running_stats1f* stats);

/** @brief Update online running statistics.
 *
 * @return 0 on success, -1 on invalid input.
 */
int kf_signal_running_stats1f_update(kf_signal_running_stats1f* stats, float x);

/** @brief Read online running statistics.
 *
 * variance_population or variance_sample can be NULL.
 *
 * @return 0 on success, -1 if no samples are available.
 */
int kf_signal_running_stats1f_get(const kf_signal_running_stats1f* stats, float* mean,
                                  float* variance_population, float* variance_sample,
                                  float* stddev_population, float* stddev_sample);

/** @brief Initialize cascaded biquad/SOS filter state.
 *
 * sections must contain initialized biquad filters.
 *
 * @return 0 on success, -1 on invalid input.
 */
int kf_signal_sos1f_init(kf_signal_sos1f* filter, kf_signal_biquad1f* sections, int count);

/** @brief Update cascaded biquad/SOS filter state.
 *
 * @return 0 on success, -1 on invalid input.
 */
int kf_signal_sos1f_update(kf_signal_sos1f* filter, float x, float* y);

/** @brief Initialize scalar DC blocker.
 *
 * Computes y = x - previous_x + r * previous_y. r should normally be close to 1.
 *
 * @return 0 on success, -1 on invalid input.
 */
int kf_signal_dc_blocker1f_init(kf_signal_dc_blocker1f* filter, float r, float initial_input,
                                float initial_output);

/** @brief Update scalar DC blocker.
 *
 * @return 0 on success, -1 on invalid input.
 */
int kf_signal_dc_blocker1f_update(kf_signal_dc_blocker1f* filter, float x, float* y);

/** @brief Initialize a sliding-window quantile filter.
 *
 * @return 0 on success, -1 on invalid input.
 */
int kf_signal_quantile1f_init(kf_signal_quantile1f* filter, float* window, float* scratch,
                              int window_size);

/** @brief Update and read a sliding-window quantile.
 *
 * quantile must be in [0, 1]. Use 0.5 for median and 0.95 for p95.
 *
 * @return 0 on success, -1 on invalid input.
 */
int kf_signal_quantile1f_update(kf_signal_quantile1f* filter, float x, float quantile, float* y);

/** @brief Estimate sigma from a buffer using MAD: sigma ~= 1.4826 * median(abs(x - median(x))).
 *
 * scratch must hold n elements.
 *
 * @return 0 on success, -1 on invalid input.
 */
int kf_signal_mad_noise_sigma(const float* x, float* scratch, int n, float* median, float* sigma);

/** @brief Z-score gate.
 *
 * @return 1 if accepted, 0 if rejected, -1 on invalid input.
 */
int kf_signal_zscore_gate(float x, float mean, float stddev, float threshold, float* zscore);

/** @brief IQR outlier gate over an input buffer.
 *
 * scratch must hold n elements.
 *
 * @return 1 if x is inside [Q1 - multiplier*IQR, Q3 + multiplier*IQR],
 *         0 if rejected, -1 on invalid input.
 */
int kf_signal_iqr_gate(float x, const float* values, float* scratch, int n, float multiplier,
                       float* q1, float* q3);

/** @brief Huber robust weight for residual.
 *
 * @return 0 on success, -1 on invalid input.
 */
int kf_signal_huber_weight(float residual, float delta, float* weight);

/** @brief Tukey biweight robust weight for residual.
 *
 * @return 0 on success, -1 on invalid input.
 */
int kf_signal_tukey_weight(float residual, float c, float* weight);

/** @brief Stateless Schmitt trigger helper.
 *
 * Uses the same switching rule as kf_signal_hysteresis1f_update().
 *
 * @return 0 on success, -1 on invalid input.
 */
int kf_signal_schmitt_trigger(float x, float low_threshold, float high_threshold, int previous_state,
                              int* state);

/** @brief Initialize scalar edge detector.
 *
 * @return 0 on success, -1 on invalid input.
 */
int kf_signal_edge1f_init(kf_signal_edge1f* filter, int initial_state);

/** @brief Update scalar edge detector.
 *
 * rising and falling can be NULL.
 *
 * @return 0 on success, -1 on invalid input.
 */
int kf_signal_edge1f_update(kf_signal_edge1f* filter, int state, int* rising, int* falling);

/** @brief Initialize sample-and-hold/dropout filter.
 *
 * If valid is false for more than max_hold_samples, update returns 0 but holds
 * the last value. max_hold_samples < 0 means hold forever.
 *
 * @return 0 on success, -1 on invalid input.
 */
int kf_signal_sample_hold1f_init(kf_signal_sample_hold1f* filter, float initial_value,
                                 int max_hold_samples);

/** @brief Update sample-and-hold/dropout filter.
 *
 * @return 1 if current sample is valid, 0 if held, -1 on invalid input.
 */
int kf_signal_sample_hold1f_update(kf_signal_sample_hold1f* filter, float x, int valid, float* y);

/** @brief Initialize sliding-window min/max filter.
 *
 * @return 0 on success, -1 on invalid input.
 */
int kf_signal_moving_minmax1f_init(kf_signal_moving_minmax1f* filter, float* window,
                                   float* scratch, int window_size);

/** @brief Update sliding-window min/max filter.
 *
 * min_value or max_value can be NULL.
 *
 * @return 0 on success, -1 on invalid input.
 */
int kf_signal_moving_minmax1f_update(kf_signal_moving_minmax1f* filter, float x,
                                     float* min_value, float* max_value);

/** @brief Initialize exponentially weighted running mean/variance.
 *
 * @return 0 on success, -1 on invalid input.
 */
int kf_signal_ew_stats1f_init(kf_signal_ew_stats1f* stats, float alpha, float initial_mean,
                              float initial_variance);

/** @brief Update exponentially weighted running mean/variance.
 *
 * mean or variance can be NULL.
 *
 * @return 0 on success, -1 on invalid input.
 */
int kf_signal_ew_stats1f_update(kf_signal_ew_stats1f* stats, float x, float* mean,
                                float* variance);

/** @brief Initialize scalar peak-hold filter.
 *
 * @return 0 on success, -1 on invalid input.
 */
int kf_signal_peak_hold1f_init(kf_signal_peak_hold1f* filter, float initial_peak,
                               float decay_per_sample);

/** @brief Update scalar peak-hold filter.
 *
 * Peak decays by decay_per_sample, then clamps upward to x if x is larger.
 *
 * @return 0 on success, -1 on invalid input.
 */
int kf_signal_peak_hold1f_update(kf_signal_peak_hold1f* filter, float x, float* peak);

/** @brief Initialize boolean majority-vote filter.
 *
 * window is caller supplied with window_size elements.
 *
 * @return 0 on success, -1 on invalid input.
 */
int kf_signal_majority1f_init(kf_signal_majority1f* filter, int* window, int window_size,
                              int initial_state);

/** @brief Update boolean majority-vote filter.
 *
 * @return 0 on success, -1 on invalid input.
 */
int kf_signal_majority1f_update(kf_signal_majority1f* filter, int input_state, int* output_state);

/** @brief Initialize stateful scalar jerk limiter.
 *
 * @return 0 on success, -1 on invalid input.
 */
int kf_signal_jerk_limiter1f_init(kf_signal_jerk_limiter1f* filter, float max_rate,
                                  float max_accel, float max_jerk, float dt, float initial_value,
                                  float initial_rate, float initial_accel);

/** @brief Update stateful scalar jerk limiter.
 *
 * @return 0 on success, -1 on invalid input.
 */
int kf_signal_jerk_limiter1f_update(kf_signal_jerk_limiter1f* filter, float target, float* y,
                                    float* rate, float* accel);

/** @brief Initialize dropout decay filter.
 *
 * decay_alpha must be in [0, 1].
 *
 * @return 0 on success, -1 on invalid input.
 */
int kf_signal_dropout_decay1f_init(kf_signal_dropout_decay1f* filter, float initial_value,
                                   float fallback, float decay_alpha);

/** @brief Update dropout decay filter.
 *
 * If valid, output becomes x. Otherwise output moves toward fallback by decay_alpha.
 *
 * @return 1 if current sample is valid, 0 if decayed, -1 on invalid input.
 */
int kf_signal_dropout_decay1f_update(kf_signal_dropout_decay1f* filter, float x, int valid,
                                     float* y);

/** @brief Compute squared Mahalanobis distance residual' * covariance^-1 * residual.
 *
 * covariance_work must point to an n*n buffer. residual_work must point to an
 * n-element buffer. Both are overwritten.
 *
 * @return 0 on success, -1 on invalid input or non-positive-definite covariance.
 */
int kf_signal_mahalanobis_distance_sq(const float* residual, const float* covariance,
                                      float* covariance_work, float* residual_work, int n,
                                      float* distance_sq);

/** @brief Mahalanobis threshold gate.
 *
 * @return 1 if accepted, 0 if rejected, -1 on invalid input.
 */
int kf_signal_mahalanobis_gate(const float* residual, const float* covariance,
                               float* covariance_work, float* residual_work, int n,
                               float threshold_sq, float* distance_sq);

/** @brief Compute the Euclidean norm of a vector.
 *
 * @return 0 on success, -1 on invalid input.
 */
int kf_signal_euclidean_norm(const float* x, int n, float* norm);

/** @brief Compute the Euclidean distance between two vectors.
 *
 * @return 0 on success, -1 on invalid input.
 */
int kf_signal_euclidean_distance(const float* a, const float* b, int n, float* distance);

/** @brief Distance-gated Euclidean vector filter.
 *
 * Computes distance = ||sample - state||. If max_distance is positive and the
 * distance is greater than max_distance, state is left unchanged and 0 is
 * returned. Otherwise state is updated with a low-pass step:
 *
 * state[i] = state[i] + alpha * (sample[i] - state[i])
 *
 * Use alpha = 1.0f to accept the sample directly.
 *
 * @return 1 if accepted, 0 if rejected, -1 on invalid input.
 */
int kf_signal_euclidean_filter(float* state, const float* sample, int n, float max_distance,
                               float alpha, float* distance);

#ifdef __cplusplus
}
#endif

#endif /* SIGNAL_FILTERS_H */

/* @} */
