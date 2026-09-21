#ifndef KFCORE_ESN_H
#define KFCORE_ESN_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum kfcore_esn_status
{
    KFCORE_ESN_OK = 0,
    KFCORE_ESN_INVALID_ARGUMENT = -1,
    KFCORE_ESN_NUMERICAL_FAILURE = -2
} kfcore_esn_status;

/** Non-owning ESN model view.
 *
 * All matrices use column-major storage. The caller owns every referenced
 * buffer and must keep it alive for each operation.
 */
typedef struct kfcore_esn_model
{
    int input_size;
    int reservoir_size;
    int output_size;
    float leak_rate;
    const float* input_weights;     /**< reservoir_size x input_size */
    const float* reservoir_weights; /**< reservoir_size x reservoir_size */
    const float* reservoir_bias;    /**< reservoir_size */
    const float* output_weights;    /**< output_size x reservoir_size */
    const float* output_bias;       /**< output_size */
} kfcore_esn_model;

/** Advance the leaky reservoir by one sample.
 *
 * @param model ESN model view. Reservoir fields must be valid.
 * @param input Input vector with model->input_size elements.
 * @param state In/out reservoir state with model->reservoir_size elements.
 * @param workspace Scratch vector with model->reservoir_size elements. It must
 *                  not overlap input, state, or model-owned weight/bias buffers.
 */
kfcore_esn_status kfcore_esn_step(const kfcore_esn_model* model, const float* input,
                                  float* state, float* workspace);

/** Apply the linear readout to a reservoir state.
 *
 * output must not overlap state or the model's output weight/bias buffers.
 */
kfcore_esn_status kfcore_esn_predict(const kfcore_esn_model* model, const float* state,
                                     float* output);

/** Advance the reservoir and then apply the readout.
 *
 * The non-overlap requirements of kfcore_esn_step and kfcore_esn_predict both
 * apply.
 */
kfcore_esn_status kfcore_esn_step_predict(const kfcore_esn_model* model, const float* input,
                                          float* state, float* workspace, float* output);

/** Fit linear readout weights with ridge regression.
 *
 * states is reservoir_size x sample_count, targets is output_size x
 * sample_count, and output_weights is output_size x reservoir_size. All are
 * column-major. gram_workspace must contain reservoir_size * reservoir_size
 * floats. states, targets, output_weights, and gram_workspace must be mutually
 * non-overlapping.
 *
 * The implementation solves the regularized normal equations with Cholesky
 * factorization. It does not form an inverse and does not fall back to a
 * different solver when factorization fails.
 */
kfcore_esn_status kfcore_esn_fit_ridge(const float* states, const float* targets,
                                       int reservoir_size, int output_size, int sample_count,
                                       float lambda, float* output_weights,
                                       float* gram_workspace);

/** Deterministically initialize dense reservoir weights with Bernoulli sparsity.
 *
 * reservoir_weights is a column-major reservoir_size x reservoir_size matrix.
 * Active candidate weights are generated from a stable SplitMix64-derived
 * sequence in [-1, 1); masked entries are exactly zero. density must be finite
 * and in (0, 1]. The function has no hidden allocation or global RNG state.
 */
kfcore_esn_status kfcore_esn_init_reservoir_weights(float* reservoir_weights,
                                                     int reservoir_size, uint64_t seed,
                                                     float density);

/** Estimate spectral radius magnitude with bounded deterministic power growth.
 *
 * The estimate uses repeated matrix-vector products and the geometric mean of
 * per-iteration growth factors. This is an iterative estimate, not a full
 * eigendecomposition. workspace must contain 2 * reservoir_size floats and
 * must not overlap reservoir_weights or spectral_radius.
 */
kfcore_esn_status kfcore_esn_estimate_spectral_radius(const float* reservoir_weights,
                                                       int reservoir_size, int iterations,
                                                       float* workspace,
                                                       float* spectral_radius);

/** Scale a dense reservoir in place to a requested spectral-radius estimate.
 *
 * target_radius must be finite and positive. The same bounded estimator used by
 * kfcore_esn_estimate_spectral_radius is applied before scaling. workspace must
 * contain 2 * reservoir_size floats and must not overlap reservoir_weights.
 * Zero or non-finite estimated radius returns KFCORE_ESN_NUMERICAL_FAILURE; no
 * seed/density change or fallback eigensolver is attempted.
 */
kfcore_esn_status kfcore_esn_scale_spectral_radius(float* reservoir_weights,
                                                    int reservoir_size, float target_radius,
                                                    int iterations, float* workspace);

/** Initialize inverse-correlation state for online recursive least squares.
 *
 * inverse_correlation is a caller-owned reservoir_size x reservoir_size
 * column-major matrix. On success it is replaced with (1 / delta) * I.
 * delta must be finite and strictly positive.
 */
kfcore_esn_status kfcore_esn_rls_init(float* inverse_correlation, int reservoir_size,
                                      float delta);

/** Perform one allocation-free forgetting-factor RLS readout update.
 *
 * state has reservoir_size elements, target has output_size elements,
 * output_weights is an output_size x reservoir_size column-major matrix, and
 * inverse_correlation is a reservoir_size x reservoir_size column-major matrix.
 * forgetting_factor must be finite and in (0, 1]. workspace must contain
 * reservoir_size + output_size floats and must not overlap any input or state
 * buffer.
 *
 * The function computes k = P*x/(lambda + x'*P*x), e = y - W*x,
 * W <- W + e*k' and P <- (P - d*k*k')/lambda. The complete update is validated
 * before output_weights or inverse_correlation is mutated. Invalid/non-finite
 * or non-positive denominator/update state returns an error without partial
 * mutation. No automatic reset, retry, or fallback is attempted.
 */
kfcore_esn_status kfcore_esn_rls_update(const float* state, const float* target,
                                        int reservoir_size, int output_size,
                                        float forgetting_factor, float* output_weights,
                                        float* inverse_correlation, float* workspace);

#ifdef __cplusplus
}
#endif

#endif
