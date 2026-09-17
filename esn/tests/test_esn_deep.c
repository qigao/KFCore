#include <limits.h>
#include <math.h>
#include <stddef.h>
#include <string.h>

#include "esn.h"
#include "esn_deep.h"
#define TINYTEST_NO_MAIN
#include "tinytest.h"

#define ESN_DEEP_TEST_EPSILON 1.0e-5f

spec("kfcore esn deep")
{
    it("reports concatenated state and staged workspace sizes")
    {
        static const float input_weights_2x2[4] = { 1.0f, 0.0f, 0.0f, 1.0f };
        static const float reservoir_weights_2[4] = { 0.0f, 0.0f, 0.0f, 0.0f };
        static const float reservoir_bias_2[2] = { 0.0f, 0.0f };
        static const float input_weights_3x2[6] = { 1.0f, 0.0f, 0.0f, 1.0f, 1.0f, 1.0f };
        static const float reservoir_weights_3[9] = {
            0.0f, 0.0f, 0.0f,
            0.0f, 0.0f, 0.0f,
            0.0f, 0.0f, 0.0f
        };
        static const float reservoir_bias_3[3] = { 0.0f, 0.0f, 0.0f };
        static const float input_weights_1x3[3] = { 1.0f, 1.0f, 1.0f };
        static const float reservoir_weights_1[1] = { 0.0f };
        static const float reservoir_bias_1[1] = { 0.0f };
        const kfcore_esn_model layers[3] = {
            { .input_size = 2, .reservoir_size = 2, .output_size = 0, .leak_rate = 1.0f,
              .input_weights = input_weights_2x2, .reservoir_weights = reservoir_weights_2,
              .reservoir_bias = reservoir_bias_2, .output_weights = NULL, .output_bias = NULL },
            { .input_size = 2, .reservoir_size = 3, .output_size = 0, .leak_rate = 1.0f,
              .input_weights = input_weights_3x2, .reservoir_weights = reservoir_weights_3,
              .reservoir_bias = reservoir_bias_3, .output_weights = NULL, .output_bias = NULL },
            { .input_size = 3, .reservoir_size = 1, .output_size = 0, .leak_rate = 1.0f,
              .input_weights = input_weights_1x3, .reservoir_weights = reservoir_weights_1,
              .reservoir_bias = reservoir_bias_1, .output_weights = NULL, .output_bias = NULL }
        };
        const kfcore_esn_deep_model deep = { 3, layers };
        int state_size = -1;
        int workspace_size = -1;

        check_equal(kfcore_esn_deep_layout(&deep, &state_size, &workspace_size), KFCORE_ESN_OK);
        check_equal(state_size, 6);
        check_equal(workspace_size, 9);
    }

    it("rejects a layer-chain dimension mismatch")
    {
        static const float scalar = 0.0f;
        const kfcore_esn_model layers[2] = {
            { .input_size = 1, .reservoir_size = 2, .output_size = 0, .leak_rate = 1.0f,
              .input_weights = &scalar, .reservoir_weights = &scalar, .reservoir_bias = &scalar },
            { .input_size = 3, .reservoir_size = 1, .output_size = 0, .leak_rate = 1.0f,
              .input_weights = &scalar, .reservoir_weights = &scalar, .reservoir_bias = &scalar }
        };
        const kfcore_esn_deep_model deep = { 2, layers };
        int state_size = -1;
        int workspace_size = -1;

        check_equal(kfcore_esn_deep_layout(&deep, &state_size, &workspace_size),
                    KFCORE_ESN_INVALID_ARGUMENT);
    }

    it("rejects required null layout outputs")
    {
        static const float scalar = 0.0f;
        const kfcore_esn_model layer = {
            .input_size = 1, .reservoir_size = 1, .output_size = 0, .leak_rate = 1.0f,
            .input_weights = &scalar, .reservoir_weights = &scalar, .reservoir_bias = &scalar
        };
        const kfcore_esn_deep_model deep = { 1, &layer };
        int size = 0;

        check_equal(kfcore_esn_deep_layout(&deep, NULL, &size), KFCORE_ESN_INVALID_ARGUMENT);
        check_equal(kfcore_esn_deep_layout(&deep, &size, NULL), KFCORE_ESN_INVALID_ARGUMENT);
    }

    it("rejects invalid layer containers")
    {
        static const float scalar = 0.0f;
        const kfcore_esn_model layer = {
            .input_size = 1, .reservoir_size = 1, .output_size = 0, .leak_rate = 1.0f,
            .input_weights = &scalar, .reservoir_weights = &scalar, .reservoir_bias = &scalar
        };
        const kfcore_esn_deep_model zero_layers = { 0, &layer };
        const kfcore_esn_deep_model null_layers = { 1, NULL };
        int state_size = -1;
        int workspace_size = -1;

        check_equal(kfcore_esn_deep_layout(&zero_layers, &state_size, &workspace_size),
                    KFCORE_ESN_INVALID_ARGUMENT);
        check_equal(kfcore_esn_deep_layout(&null_layers, &state_size, &workspace_size),
                    KFCORE_ESN_INVALID_ARGUMENT);
    }

    it("rejects checked state-size overflow after valid chain dimensions")
    {
        static const float scalar = 0.0f;
        const kfcore_esn_model layers[2] = {
            { .input_size = 1, .reservoir_size = INT_MAX, .output_size = 0, .leak_rate = 1.0f,
              .input_weights = &scalar, .reservoir_weights = &scalar, .reservoir_bias = &scalar },
            { .input_size = INT_MAX, .reservoir_size = 1, .output_size = 0, .leak_rate = 1.0f,
              .input_weights = &scalar, .reservoir_weights = &scalar, .reservoir_bias = &scalar }
        };
        const kfcore_esn_deep_model deep = { 2, layers };
        int state_size = -1;
        int workspace_size = -1;

        check_equal(kfcore_esn_deep_layout(&deep, &state_size, &workspace_size),
                    KFCORE_ESN_INVALID_ARGUMENT);
    }

    it("rejects checked workspace-size overflow")
    {
        static const float scalar = 0.0f;
        const kfcore_esn_model layer = {
            .input_size = 1, .reservoir_size = INT_MAX, .output_size = 0, .leak_rate = 1.0f,
            .input_weights = &scalar, .reservoir_weights = &scalar, .reservoir_bias = &scalar
        };
        const kfcore_esn_deep_model deep = { 1, &layer };
        int state_size = -1;
        int workspace_size = -1;

        check_equal(kfcore_esn_deep_layout(&deep, &state_size, &workspace_size),
                    KFCORE_ESN_INVALID_ARGUMENT);
    }

    it("matches ordinary stepping for a single layer")
    {
        static const float input_weights[2] = { 1.0f, -1.0f };
        static const float reservoir_weights[4] = {
            0.25f, 0.0f,
            0.0f, -0.25f
        };
        static const float reservoir_bias[2] = { 0.1f, -0.1f };
        const kfcore_esn_model layer = {
            .input_size = 1, .reservoir_size = 2, .output_size = 0, .leak_rate = 0.5f,
            .input_weights = input_weights, .reservoir_weights = reservoir_weights,
            .reservoir_bias = reservoir_bias
        };
        const kfcore_esn_deep_model deep = { 1, &layer };
        const float input[1] = { 0.75f };
        float ordinary_state[2] = { 0.25f, -0.5f };
        float deep_state[2] = { 0.25f, -0.5f };
        float ordinary_workspace[2] = { 0.0f, 0.0f };
        float deep_workspace[4] = { 0.0f, 0.0f, 0.0f, 0.0f };

        check_equal(kfcore_esn_step(&layer, input, ordinary_state, ordinary_workspace), KFCORE_ESN_OK);
        check_equal(kfcore_esn_deep_step(&deep, input, deep_state, deep_workspace), KFCORE_ESN_OK);
        check_within(deep_state[0], ordinary_state[0], ESN_DEEP_TEST_EPSILON);
        check_within(deep_state[1], ordinary_state[1], ESN_DEEP_TEST_EPSILON);
    }

    it("feeds each later layer the freshly computed same-timestep state")
    {
        static const float one = 1.0f;
        static const float zero = 0.0f;
        const kfcore_esn_model layers[2] = {
            { .input_size = 1, .reservoir_size = 1, .output_size = 0, .leak_rate = 1.0f,
              .input_weights = &one, .reservoir_weights = &zero, .reservoir_bias = &zero },
            { .input_size = 1, .reservoir_size = 1, .output_size = 0, .leak_rate = 1.0f,
              .input_weights = &one, .reservoir_weights = &zero, .reservoir_bias = &zero }
        };
        const kfcore_esn_deep_model deep = { 2, layers };
        const float input[1] = { 1.0f };
        float state[2] = { -0.5f, 0.0f };
        float workspace[3] = { 0.0f, 0.0f, 0.0f };
        const float expected_layer0 = tanhf(1.0f);
        const float expected_layer1 = tanhf(expected_layer0);

        check_equal(kfcore_esn_deep_step(&deep, input, state, workspace), KFCORE_ESN_OK);
        check_within(state[0], expected_layer0, ESN_DEEP_TEST_EPSILON);
        check_within(state[1], expected_layer1, ESN_DEEP_TEST_EPSILON);
        check_equal(state[1] > 0.0f, 1);
    }

    it("preserves heterogeneous layer state ordering")
    {
        static const float layer0_input_weights[2] = { 1.0f, -1.0f };
        static const float layer0_reservoir_weights[4] = { 0.0f, 0.0f, 0.0f, 0.0f };
        static const float layer0_bias[2] = { 0.0f, 0.0f };
        static const float layer1_input_weights[2] = { 1.0f, 0.5f };
        static const float layer1_reservoir_weights[1] = { 0.0f };
        static const float layer1_bias[1] = { 0.0f };
        const kfcore_esn_model layers[2] = {
            { .input_size = 1, .reservoir_size = 2, .output_size = 0, .leak_rate = 1.0f,
              .input_weights = layer0_input_weights,
              .reservoir_weights = layer0_reservoir_weights,
              .reservoir_bias = layer0_bias },
            { .input_size = 2, .reservoir_size = 1, .output_size = 0, .leak_rate = 1.0f,
              .input_weights = layer1_input_weights,
              .reservoir_weights = layer1_reservoir_weights,
              .reservoir_bias = layer1_bias }
        };
        const kfcore_esn_deep_model deep = { 2, layers };
        const float input[1] = { 1.0f };
        float state[3] = { 0.0f, 0.0f, 0.0f };
        float workspace[5] = { 0.0f, 0.0f, 0.0f, 0.0f, 0.0f };
        const float expected0 = tanhf(1.0f);
        const float expected1 = -expected0;
        const float expected2 = tanhf(expected0 + 0.5f * expected1);

        check_equal(kfcore_esn_deep_step(&deep, input, state, workspace), KFCORE_ESN_OK);
        check_within(state[0], expected0, ESN_DEEP_TEST_EPSILON);
        check_within(state[1], expected1, ESN_DEEP_TEST_EPSILON);
        check_within(state[2], expected2, ESN_DEEP_TEST_EPSILON);
    }

    it("rejects invalid later-layer metadata without advancing caller state")
    {
        static const float one = 1.0f;
        static const float zero = 0.0f;
        const kfcore_esn_model layers[2] = {
            { .input_size = 1, .reservoir_size = 1, .output_size = 0, .leak_rate = 1.0f,
              .input_weights = &one, .reservoir_weights = &zero, .reservoir_bias = &zero },
            { .input_size = 1, .reservoir_size = 1, .output_size = 0, .leak_rate = 1.0f,
              .input_weights = &one, .reservoir_weights = &zero, .reservoir_bias = NULL }
        };
        const kfcore_esn_deep_model deep = { 2, layers };
        const float input[1] = { 1.0f };
        float state[2] = { 0.25f, -0.75f };
        float before[2];
        float workspace[3] = { 0.0f, 0.0f, 0.0f };
        memcpy(before, state, sizeof(state));

        check_equal(kfcore_esn_deep_step(&deep, input, state, workspace),
                    KFCORE_ESN_INVALID_ARGUMENT);
        check_equal(memcmp(state, before, sizeof(state)), 0);
    }

    it("rejects a chain mismatch without advancing caller state")
    {
        static const float one = 1.0f;
        static const float zero = 0.0f;
        static const float two_inputs[2] = { 1.0f, 1.0f };
        const kfcore_esn_model layers[2] = {
            { .input_size = 1, .reservoir_size = 1, .output_size = 0, .leak_rate = 1.0f,
              .input_weights = &one, .reservoir_weights = &zero, .reservoir_bias = &zero },
            { .input_size = 2, .reservoir_size = 1, .output_size = 0, .leak_rate = 1.0f,
              .input_weights = two_inputs, .reservoir_weights = &zero, .reservoir_bias = &zero }
        };
        const kfcore_esn_deep_model deep = { 2, layers };
        const float input[1] = { 1.0f };
        float state[2] = { 0.25f, -0.75f };
        float before[2];
        float workspace[3] = { 0.0f, 0.0f, 0.0f };
        memcpy(before, state, sizeof(state));

        check_equal(kfcore_esn_deep_step(&deep, input, state, workspace),
                    KFCORE_ESN_INVALID_ARGUMENT);
        check_equal(memcmp(state, before, sizeof(state)), 0);
    }

    it("rejects null deep runtime arguments without mutation")
    {
        static const float one = 1.0f;
        static const float zero = 0.0f;
        const kfcore_esn_model layer = {
            .input_size = 1, .reservoir_size = 1, .output_size = 0, .leak_rate = 1.0f,
            .input_weights = &one, .reservoir_weights = &zero, .reservoir_bias = &zero
        };
        const kfcore_esn_deep_model deep = { 1, &layer };
        const float input[1] = { 1.0f };
        float state[1] = { 0.4f };
        const float before = state[0];
        float workspace[2] = { 0.0f, 0.0f };

        check_equal(kfcore_esn_deep_step(NULL, input, state, workspace),
                    KFCORE_ESN_INVALID_ARGUMENT);
        check_within(state[0], before, 0.0f);
        check_equal(kfcore_esn_deep_step(&deep, NULL, state, workspace),
                    KFCORE_ESN_INVALID_ARGUMENT);
        check_within(state[0], before, 0.0f);
        check_equal(kfcore_esn_deep_step(&deep, input, NULL, workspace),
                    KFCORE_ESN_INVALID_ARGUMENT);
        check_equal(kfcore_esn_deep_step(&deep, input, state, NULL),
                    KFCORE_ESN_INVALID_ARGUMENT);
        check_within(state[0], before, 0.0f);
    }
}
