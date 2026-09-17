#include <math.h>
#include <string.h>

#include "esn.h"
#include "esn_sparse.h"
#define TINYTEST_NO_MAIN
#include "tinytest.h"

#define ESN_SPARSE_EPSILON 1.0e-5f

spec("kfcore esn sparse")
{
    it("converts a known column-major reservoir to canonical CSR")
    {
        const float dense[9] = {
            1.0f, 0.0f, 4.0f,
            0.0f, 3.0f, 0.0f,
            2.0f, 0.0f, 5.0f
        };
        int row_offsets[4] = { -1, -1, -1, -1 };
        int column_indices[5] = { -1, -1, -1, -1, -1 };
        float values[5] = { 0.0f };
        int nonzero_count = -1;

        check_equal(kfcore_esn_sparse_count_nonzero(dense, 3, &nonzero_count), KFCORE_ESN_OK);
        check_equal(nonzero_count, 5);
        check_equal(kfcore_esn_sparse_from_dense(dense, 3, 5, row_offsets, column_indices,
                                                  values, &nonzero_count),
                    KFCORE_ESN_OK);

        const int expected_offsets[4] = { 0, 2, 3, 5 };
        const int expected_columns[5] = { 0, 2, 1, 0, 2 };
        const float expected_values[5] = { 1.0f, 2.0f, 3.0f, 4.0f, 5.0f };
        check_equal(memcmp(row_offsets, expected_offsets, sizeof(row_offsets)), 0);
        check_equal(memcmp(column_indices, expected_columns, sizeof(column_indices)), 0);
        for (int i = 0; i < 5; ++i)
        {
            check_within(values[i], expected_values[i], ESN_SPARSE_EPSILON);
        }
    }

    it("accepts a zero-edge sparse reservoir without dense recurrence")
    {
        const float dense[4] = { 0.0f, 0.0f, 0.0f, 0.0f };
        int row_offsets[3] = { -1, -1, -1 };
        int nonzero_count = -1;

        check_equal(kfcore_esn_sparse_from_dense(dense, 2, 0, row_offsets, NULL, NULL,
                                                  &nonzero_count),
                    KFCORE_ESN_OK);
        check_equal(nonzero_count, 0);
        check_equal(row_offsets[0], 0);
        check_equal(row_offsets[1], 0);
        check_equal(row_offsets[2], 0);

        const float input_weights[2] = { 1.0f, -1.0f };
        const float bias[2] = { 0.0f, 0.0f };
        const kfcore_esn_model model = {
            1, 2, 0, 1.0f, input_weights, NULL, bias, NULL, NULL
        };
        const kfcore_esn_sparse_reservoir sparse = { 2, 0, row_offsets, NULL, NULL };
        const float input[1] = { 0.5f };
        float state[2] = { 0.25f, -0.25f };
        float workspace[2] = { 0.0f, 0.0f };

        check_equal(kfcore_esn_step_sparse(&model, &sparse, input, state, workspace),
                    KFCORE_ESN_OK);
        check_within(state[0], tanhf(0.5f), ESN_SPARSE_EPSILON);
        check_within(state[1], tanhf(-0.5f), ESN_SPARSE_EPSILON);
    }

    it("matches dense reservoir stepping for the same recurrence")
    {
        const float input_weights[2] = { 0.75f, -0.5f };
        const float dense[4] = {
            0.5f, -0.25f,
            0.0f, 0.75f
        };
        const float bias[2] = { 0.1f, -0.2f };
        const kfcore_esn_model dense_model = {
            1, 2, 0, 0.6f, input_weights, dense, bias, NULL, NULL
        };
        const kfcore_esn_model sparse_model = {
            1, 2, 0, 0.6f, input_weights, NULL, bias, NULL, NULL
        };
        int row_offsets[3] = { 0, 0, 0 };
        int column_indices[3] = { 0, 0, 0 };
        float values[3] = { 0.0f, 0.0f, 0.0f };
        int nonzero_count = 0;
        const float input[1] = { 0.3f };
        float dense_state[2] = { 0.2f, -0.4f };
        float sparse_state[2] = { 0.2f, -0.4f };
        float dense_workspace[2] = { 0.0f, 0.0f };
        float sparse_workspace[2] = { 0.0f, 0.0f };

        check_equal(kfcore_esn_sparse_from_dense(dense, 2, 3, row_offsets, column_indices,
                                                  values, &nonzero_count),
                    KFCORE_ESN_OK);
        const kfcore_esn_sparse_reservoir sparse = {
            2, nonzero_count, row_offsets, column_indices, values
        };

        check_equal(kfcore_esn_step(&dense_model, input, dense_state, dense_workspace),
                    KFCORE_ESN_OK);
        check_equal(kfcore_esn_step_sparse(&sparse_model, &sparse, input, sparse_state,
                                           sparse_workspace),
                    KFCORE_ESN_OK);
        check_within(sparse_state[0], dense_state[0], ESN_SPARSE_EPSILON);
        check_within(sparse_state[1], dense_state[1], ESN_SPARSE_EPSILON);
    }

    it("rejects malformed CSR without advancing state")
    {
        const float input_weights[2] = { 0.0f, 0.0f };
        const float bias[2] = { 0.0f, 0.0f };
        const kfcore_esn_model model = {
            1, 2, 0, 0.5f, input_weights, NULL, bias, NULL, NULL
        };
        const int row_offsets[3] = { 0, 2, 1 };
        const int column_indices[2] = { 0, 1 };
        const float values[2] = { 1.0f, 1.0f };
        const kfcore_esn_sparse_reservoir sparse = {
            2, 1, row_offsets, column_indices, values
        };
        const float input[1] = { 0.0f };
        float state[2] = { 0.25f, -0.5f };
        const float original_state[2] = { 0.25f, -0.5f };
        float workspace[2] = { 0.0f, 0.0f };

        check_equal(kfcore_esn_step_sparse(&model, &sparse, input, state, workspace),
                    KFCORE_ESN_INVALID_ARGUMENT);
        check_equal(memcmp(state, original_state, sizeof(state)), 0);
    }

    it("rejects insufficient conversion capacity without partial output")
    {
        const float dense[4] = {
            1.0f, 2.0f,
            0.0f, 3.0f
        };
        int row_offsets[3] = { 91, 92, 93 };
        int column_indices[2] = { 81, 82 };
        float values[2] = { 71.0f, 72.0f };
        int nonzero_count = 61;
        const int original_offsets[3] = { 91, 92, 93 };
        const int original_columns[2] = { 81, 82 };
        const float original_values[2] = { 71.0f, 72.0f };

        check_equal(kfcore_esn_sparse_from_dense(dense, 2, 2, row_offsets, column_indices,
                                                  values, &nonzero_count),
                    KFCORE_ESN_INVALID_ARGUMENT);
        check_equal(nonzero_count, 61);
        check_equal(memcmp(row_offsets, original_offsets, sizeof(row_offsets)), 0);
        check_equal(memcmp(column_indices, original_columns, sizeof(column_indices)), 0);
        check_equal(memcmp(values, original_values, sizeof(values)), 0);
    }

    it("rejects non-finite dense and sparse weights")
    {
        const float dense[1] = { NAN };
        int nonzero_count = 123;
        check_equal(kfcore_esn_sparse_count_nonzero(dense, 1, &nonzero_count),
                    KFCORE_ESN_INVALID_ARGUMENT);
        check_equal(nonzero_count, 123);

        const float input_weights[1] = { 0.0f };
        const float bias[1] = { 0.0f };
        const kfcore_esn_model model = {
            1, 1, 0, 1.0f, input_weights, NULL, bias, NULL, NULL
        };
        const int row_offsets[2] = { 0, 1 };
        const int column_indices[1] = { 0 };
        const float values[1] = { NAN };
        const kfcore_esn_sparse_reservoir sparse = {
            1, 1, row_offsets, column_indices, values
        };
        const float input[1] = { 0.0f };
        float state[1] = { 0.25f };
        float workspace[1] = { 0.0f };

        check_equal(kfcore_esn_step_sparse(&model, &sparse, input, state, workspace),
                    KFCORE_ESN_INVALID_ARGUMENT);
        check_within(state[0], 0.25f, ESN_SPARSE_EPSILON);
    }
}
