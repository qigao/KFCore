#include <limits.h>
#include <math.h>
#include <string.h>

#include "esn.h"
#include "esn_group.h"
#define TINYTEST_NO_MAIN
#include "tinytest.h"

#define ESN_GROUP_EPSILON 1.0e-5f

spec("kfcore esn grouped")
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
        static const float input_weights_1x2[2] = { 1.0f, 1.0f };
        static const float reservoir_weights_1[1] = { 0.0f };
        static const float reservoir_bias_1[1] = { 0.0f };
        const kfcore_esn_model groups[3] = {
            { 2, 2, 0, 1.0f, input_weights_2x2, reservoir_weights_2, reservoir_bias_2, NULL, NULL },
            { 2, 3, 0, 1.0f, input_weights_3x2, reservoir_weights_3, reservoir_bias_3, NULL, NULL },
            { 2, 1, 0, 1.0f, input_weights_1x2, reservoir_weights_1, reservoir_bias_1, NULL, NULL }
        };
        const kfcore_esn_grouped_model grouped = { 3, groups };
        int state_size = -1;
        int workspace_size = -1;

        check_equal(kfcore_esn_grouped_layout(&grouped, &state_size, &workspace_size),
                    KFCORE_ESN_OK);
        check_equal(state_size, 6);
        check_equal(workspace_size, 9);
    }

    it("rejects mismatched group input sizes")
    {
        static const float scalar = 0.0f;
        static const float input_weights_2[2] = { 1.0f, 1.0f };
        const kfcore_esn_model groups[2] = {
            { 1, 1, 0, 1.0f, &scalar, &scalar, &scalar, NULL, NULL },
            { 2, 1, 0, 1.0f, input_weights_2, &scalar, &scalar, NULL, NULL }
        };
        const kfcore_esn_grouped_model grouped = { 2, groups };
        int state_size = -1;
        int workspace_size = -1;

        check_equal(kfcore_esn_grouped_layout(&grouped, &state_size, &workspace_size),
                    KFCORE_ESN_INVALID_ARGUMENT);
    }

    it("rejects required null layout outputs")
    {
        static const float scalar = 0.0f;
        const kfcore_esn_model group = {
            1, 1, 0, 1.0f, &scalar, &scalar, &scalar, NULL, NULL
        };
        const kfcore_esn_grouped_model grouped = { 1, &group };
        int size = 0;

        check_equal(kfcore_esn_grouped_layout(&grouped, NULL, &size),
                    KFCORE_ESN_INVALID_ARGUMENT);
        check_equal(kfcore_esn_grouped_layout(&grouped, &size, NULL),
                    KFCORE_ESN_INVALID_ARGUMENT);
    }

    it("rejects invalid group containers")
    {
        static const float scalar = 0.0f;
        const kfcore_esn_model group = {
            1, 1, 0, 1.0f, &scalar, &scalar, &scalar, NULL, NULL
        };
        const kfcore_esn_grouped_model zero_groups = { 0, &group };
        const kfcore_esn_grouped_model null_groups = { 1, NULL };
        int state_size = -1;
        int workspace_size = -1;

        check_equal(kfcore_esn_grouped_layout(&zero_groups, &state_size, &workspace_size),
                    KFCORE_ESN_INVALID_ARGUMENT);
        check_equal(kfcore_esn_grouped_layout(&null_groups, &state_size, &workspace_size),
                    KFCORE_ESN_INVALID_ARGUMENT);
    }

    it("rejects checked layout overflow")
    {
        static const float scalar = 0.0f;
        const kfcore_esn_model groups[2] = {
            { 1, INT_MAX, 0, 1.0f, &scalar, &scalar, &scalar, NULL, NULL },
            { 1, INT_MAX, 0, 1.0f, &scalar, &scalar, &scalar, NULL, NULL }
        };
        const kfcore_esn_grouped_model grouped = { 2, groups };
        int state_size = -1;
        int workspace_size = -1;

        check_equal(kfcore_esn_grouped_layout(&grouped, &state_size, &workspace_size),
                    KFCORE_ESN_INVALID_ARGUMENT);
    }

    it("matches ordinary stepping for a single group")
    {
        static const float input_weights[2] = { 1.0f, -0.5f };
        static const float reservoir_weights[4] = {
            0.25f, -0.1f,
            0.2f, 0.3f
        };
        static const float reservoir_bias[2] = { 0.05f, -0.1f };
        const kfcore_esn_model model = {
            1, 2, 0, 0.5f, input_weights, reservoir_weights, reservoir_bias, NULL, NULL
        };
        const kfcore_esn_grouped_model grouped = { 1, &model };
        const float input[1] = { 0.75f };
        float dense_state[2] = { 0.25f, -0.5f };
        float grouped_state[2] = { 0.25f, -0.5f };
        float dense_workspace[2] = { 0.0f, 0.0f };
        float grouped_workspace[4] = { 0.0f, 0.0f, 0.0f, 0.0f };

        check_equal(kfcore_esn_step(&model, input, dense_state, dense_workspace), KFCORE_ESN_OK);
        check_equal(kfcore_esn_grouped_step(&grouped, input, grouped_state, grouped_workspace),
                    KFCORE_ESN_OK);
        check_within(grouped_state[0], dense_state[0], ESN_GROUP_EPSILON);
        check_within(grouped_state[1], dense_state[1], ESN_GROUP_EPSILON);
    }

    it("concatenates heterogeneous group states in declaration order")
    {
        static const float group0_input_weights[1] = { 1.0f };
        static const float group0_reservoir_weights[1] = { 0.0f };
        static const float group0_bias[1] = { 0.0f };
        static const float group1_input_weights[2] = { 2.0f, -1.0f };
        static const float group1_reservoir_weights[4] = { 0.0f, 0.0f, 0.0f, 0.0f };
        static const float group1_bias[2] = { 0.5f, -0.25f };
        const kfcore_esn_model groups[2] = {
            { 1, 1, 0, 1.0f, group0_input_weights, group0_reservoir_weights, group0_bias, NULL, NULL },
            { 1, 2, 0, 1.0f, group1_input_weights, group1_reservoir_weights, group1_bias, NULL, NULL }
        };
        const kfcore_esn_grouped_model grouped = { 2, groups };
        const float input[1] = { 0.5f };
        float state[3] = { 0.0f, 0.0f, 0.0f };
        float workspace[5] = { 0.0f, 0.0f, 0.0f, 0.0f, 0.0f };

        check_equal(kfcore_esn_grouped_step(&grouped, input, state, workspace), KFCORE_ESN_OK);
        check_within(state[0], tanhf(0.5f), ESN_GROUP_EPSILON);
        check_within(state[1], tanhf(1.5f), ESN_GROUP_EPSILON);
        check_within(state[2], tanhf(-0.75f), ESN_GROUP_EPSILON);
    }

    it("rejects a later invalid group without advancing any state")
    {
        static const float scalar_input_weight[1] = { 1.0f };
        static const float scalar_reservoir_weight[1] = { 0.0f };
        static const float scalar_bias[1] = { 0.0f };
        const kfcore_esn_model groups[2] = {
            { 1, 1, 0, 1.0f, scalar_input_weight, scalar_reservoir_weight, scalar_bias, NULL, NULL },
            { 1, 1, 0, 1.0f, scalar_input_weight, scalar_reservoir_weight, NULL, NULL, NULL }
        };
        const kfcore_esn_grouped_model grouped = { 2, groups };
        const float input[1] = { 1.0f };
        float state[2] = { 0.25f, -0.75f };
        float before[2];
        float workspace[3] = { 0.0f, 0.0f, 0.0f };
        memcpy(before, state, sizeof(state));

        check_equal(kfcore_esn_grouped_step(&grouped, input, state, workspace),
                    KFCORE_ESN_INVALID_ARGUMENT);
        check_equal(memcmp(state, before, sizeof(state)), 0);
    }

    it("rejects null grouped runtime arguments without mutation")
    {
        static const float input_weight[1] = { 1.0f };
        static const float reservoir_weight[1] = { 0.0f };
        static const float bias[1] = { 0.0f };
        const kfcore_esn_model group = {
            1, 1, 0, 1.0f, input_weight, reservoir_weight, bias, NULL, NULL
        };
        const kfcore_esn_grouped_model grouped = { 1, &group };
        const float input[1] = { 0.5f };
        float state[1] = { 0.25f };
        float before[1] = { 0.25f };
        float workspace[2] = { 0.0f, 0.0f };

        check_equal(kfcore_esn_grouped_step(NULL, input, state, workspace),
                    KFCORE_ESN_INVALID_ARGUMENT);
        check_equal(memcmp(state, before, sizeof(state)), 0);
        check_equal(kfcore_esn_grouped_step(&grouped, NULL, state, workspace),
                    KFCORE_ESN_INVALID_ARGUMENT);
        check_equal(memcmp(state, before, sizeof(state)), 0);
        check_equal(kfcore_esn_grouped_step(&grouped, input, NULL, workspace),
                    KFCORE_ESN_INVALID_ARGUMENT);
        check_equal(kfcore_esn_grouped_step(&grouped, input, state, NULL),
                    KFCORE_ESN_INVALID_ARGUMENT);
        check_equal(memcmp(state, before, sizeof(state)), 0);
    }
}
