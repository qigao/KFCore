#ifndef KFCORE_ESN_GROUP_H
#define KFCORE_ESN_GROUP_H

#include "esn.h"

#ifdef __cplusplus
extern "C" {
#endif

/** Non-owning grouped ESN composition view.
 *
 * Every group consumes the same input vector and must expose the same positive
 * input_size. Group states are concatenated in declaration order. The caller
 * owns the group array and every buffer referenced by each kfcore_esn_model.
 */
typedef struct kfcore_esn_grouped_model
{
    int group_count;
    const kfcore_esn_model* groups;
} kfcore_esn_grouped_model;

/** Query grouped state and workspace requirements.
 *
 * state_size and workspace_size are required outputs. On success, state_size
 * is the sum of all group reservoir sizes and workspace_size is state_size plus
 * the largest group reservoir size. Readout fields are not required.
 */
kfcore_esn_status kfcore_esn_grouped_layout(const kfcore_esn_grouped_model* grouped,
                                            int* state_size, int* workspace_size);

/** Advance every dense reservoir from the same input with atomic state commit.
 *
 * state contains the concatenated group state in declaration order. workspace
 * must contain at least the number of floats reported by
 * kfcore_esn_grouped_layout. input, state, workspace, and every group weight or
 * bias buffer must be mutually non-overlapping. No sparse fallback or hidden
 * allocation is performed. If validation or any delegated group step fails,
 * caller state is left unchanged; workspace remains scratch and may change.
 */
kfcore_esn_status kfcore_esn_grouped_step(const kfcore_esn_grouped_model* grouped,
                                          const float* input, float* state,
                                          float* workspace);

#ifdef __cplusplus
}
#endif

#endif
