#ifndef KFCORE_ESN_DEEP_H
#define KFCORE_ESN_DEEP_H

#include "esn.h"

#ifdef __cplusplus
extern "C" {
#endif

/** Non-owning deep ESN composition view.
 *
 * Layer 0 consumes the external input. Every later layer consumes the newly
 * computed state of the immediately preceding layer from the same timestep.
 * Layer states are concatenated in declaration/depth order. The caller owns
 * the layer array and every buffer referenced by each kfcore_esn_model.
 */
typedef struct kfcore_esn_deep_model
{
    int layer_count;
    const kfcore_esn_model* layers;
} kfcore_esn_deep_model;

/** Query deep state and workspace requirements.
 *
 * state_size and workspace_size are required outputs. On success, state_size
 * is the sum of all layer reservoir sizes and workspace_size is state_size plus
 * the largest layer reservoir size. Readout fields are not required.
 */
kfcore_esn_status kfcore_esn_deep_layout(const kfcore_esn_deep_model* deep,
                                         int* state_size, int* workspace_size);

/** Advance a dense deep reservoir chain with atomic caller-state commit.
 *
 * input contains the external input for layer 0. state contains all layer
 * states concatenated in declaration order. workspace must contain at least
 * the number of floats reported by kfcore_esn_deep_layout. input, state,
 * workspace, and every layer weight or bias buffer must be mutually
 * non-overlapping. No sparse fallback or hidden allocation is performed. If
 * validation or any delegated layer step fails, caller state is left
 * unchanged; workspace remains scratch and may change after staging begins.
 */
kfcore_esn_status kfcore_esn_deep_step(const kfcore_esn_deep_model* deep,
                                       const float* input, float* state,
                                       float* workspace);

#ifdef __cplusplus
}
#endif

#endif
