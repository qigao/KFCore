#ifndef KFCORE_ESN_GROUP_DEEP_H
#define KFCORE_ESN_GROUP_DEEP_H

#include "esn_deep.h"

#ifdef __cplusplus
extern "C" {
#endif

/** Non-owning grouped-deep ESN composition view.
 *
 * Every deep group consumes the same external input vector. Each group's state
 * keeps the existing dESN depth order, and complete group states are
 * concatenated in group declaration order. The caller owns the group array,
 * every group's layer array, and all buffers referenced by those layers.
 */
typedef struct kfcore_esn_grouped_deep_model
{
    int group_count;
    const kfcore_esn_deep_model* groups;
} kfcore_esn_grouped_deep_model;

/** Query grouped-deep state and workspace requirements.
 *
 * state_size and workspace_size are required outputs. On success, state_size
 * is the sum of all deep-group state sizes and workspace_size is state_size
 * plus the largest deep-group workspace requirement. All groups must expose
 * the same positive external input_size through their first layer. Readout
 * fields are not required. No hidden allocation or sparse fallback is used.
 */
kfcore_esn_status kfcore_esn_grouped_deep_layout(
    const kfcore_esn_grouped_deep_model* grouped_deep,
    int* state_size,
    int* workspace_size);

/** Advance every dense deep group from the same input with atomic state commit.
 *
 * state contains group states concatenated in group-major, depth-minor order.
 * workspace must contain at least the number of floats reported by
 * kfcore_esn_grouped_deep_layout. Complete grouped validation occurs before
 * outer staging. input, state, workspace, and every layer weight/bias buffer
 * must be mutually non-overlapping. If validation or any delegated deep step
 * fails, caller state is unchanged; workspace is scratch and may change after
 * staging begins. No retry, group skip, sparse fallback, or hidden allocation
 * is performed.
 */
kfcore_esn_status kfcore_esn_grouped_deep_step(
    const kfcore_esn_grouped_deep_model* grouped_deep,
    const float* input,
    float* state,
    float* workspace);

#ifdef __cplusplus
}
#endif

#endif
