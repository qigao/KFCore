#ifndef KFCORE_ESN_SPARSE_H
#define KFCORE_ESN_SPARSE_H

#include "esn.h"

#ifdef __cplusplus
extern "C" {
#endif

/** Non-owning canonical CSR view of a recurrent ESN reservoir.
 *
 * row_offsets contains reservoir_size + 1 entries, begins at zero, is
 * non-decreasing, and ends at nonzero_count. Within each row, column_indices
 * are strictly increasing and in [0, reservoir_size). values contains finite
 * float32 weights. nonzero_count == 0 is valid; in that case column_indices
 * and values may be NULL.
 */
typedef struct kfcore_esn_sparse_reservoir
{
    int reservoir_size;
    int nonzero_count;
    const int* row_offsets;
    const int* column_indices;
    const float* values;
} kfcore_esn_sparse_reservoir;

/** Count exact non-zero entries in a dense column-major reservoir matrix. */
kfcore_esn_status kfcore_esn_sparse_count_nonzero(const float* reservoir_weights,
                                                  int reservoir_size,
                                                  int* nonzero_count);

/** Convert a dense column-major reservoir matrix to canonical CSR.
 *
 * row_offsets must hold reservoir_size + 1 integers. column_indices and values
 * must each hold capacity entries when capacity > 0. The function first
 * validates/counts the complete source matrix; insufficient capacity or any
 * invalid input returns an error without partially modifying caller outputs.
 */
kfcore_esn_status kfcore_esn_sparse_from_dense(const float* reservoir_weights,
                                               int reservoir_size, int capacity,
                                               int* row_offsets, int* column_indices,
                                               float* values, int* nonzero_count);

/** Advance a leaky ESN reservoir using CSR recurrence.
 *
 * The model supplies input weights, reservoir bias, dimensions, and leak rate;
 * model->reservoir_weights is not read and may be NULL. sparse->reservoir_size
 * must equal model->reservoir_size. input/state/workspace follow the same
 * ownership and non-overlap rules as kfcore_esn_step.
 *
 * No dense fallback is attempted. Rejected sparse input or numerical failure
 * leaves state unchanged; workspace is scratch and may be modified.
 */
kfcore_esn_status kfcore_esn_step_sparse(const kfcore_esn_model* model,
                                         const kfcore_esn_sparse_reservoir* sparse,
                                         const float* input, float* state,
                                         float* workspace);

#ifdef __cplusplus
}
#endif

#endif
