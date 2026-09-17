#ifndef KFCORE_ESN_H
#define KFCORE_ESN_H

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
 * @param workspace Scratch vector with model->reservoir_size elements.
 */
kfcore_esn_status kfcore_esn_step(const kfcore_esn_model* model, const float* input,
                                  float* state, float* workspace);

/** Apply the linear readout to a reservoir state. */
kfcore_esn_status kfcore_esn_predict(const kfcore_esn_model* model, const float* state,
                                     float* output);

/** Advance the reservoir and then apply the readout. */
kfcore_esn_status kfcore_esn_step_predict(const kfcore_esn_model* model, const float* input,
                                          float* state, float* workspace, float* output);

/** Fit linear readout weights with ridge regression.
 *
 * states is reservoir_size x sample_count, targets is output_size x
 * sample_count, and output_weights is output_size x reservoir_size. All are
 * column-major. gram_workspace must contain reservoir_size * reservoir_size
 * floats and must not overlap the other buffers.
 *
 * The implementation solves the regularized normal equations with Cholesky
 * factorization. It does not form an inverse and does not fall back to a
 * different solver when factorization fails.
 */
kfcore_esn_status kfcore_esn_fit_ridge(const float* states, const float* targets,
                                       int reservoir_size, int output_size, int sample_count,
                                       float lambda, float* output_weights,
                                       float* gram_workspace);

#ifdef __cplusplus
}
#endif

#endif
