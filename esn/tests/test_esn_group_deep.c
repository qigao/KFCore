#include <limits.h>
#include <math.h>
#include <stddef.h>
#include <string.h>

#include "esn.h"
#include "esn_deep.h"
#include "esn_group_deep.h"
#define TINYTEST_NO_MAIN
#include "tinytest.h"

#define ESN_GROUP_DEEP_TEST_EPSILON 1.0e-5f

spec("kfcore esn grouped deep")
{
    it("reports grouped-deep state and nested workspace sizes")
    {
        static const float scalar = 0.0f;
        const kfcore_esn_model group0_layers[2] = {
            { .input_size = 2, .reservoir_size = 2, .output_size = 0, .leak_rate = 1.0f,
              .input_weights = &scalar, .reservoir_weights = &scalar,
              .reservoir_bias = &scalar },
            { .input_size = 2, .reservoir_size = 1, .output_size = 0, .leak_rate = 1.0f,
              .input_weights = &scalar, .reservoir_weights = &scalar,
              .reservoir_bias = &scalar }
        };
        const kfcore_esn_model group1_layers[2] = {
            { .input_size = 2, .reservoir_size = 3, .output_size = 0, .leak_rate = 1.0f,
              .input_weights = &scalar, .reservoir_weights = &scalar,
              .reservoir_bias = &scalar },
            { .input_size = 3, .reservoir_size = 2, .output_size = 0, .leak_rate = 1.0f,
              .input_weights = &scalar, .reservoir_weights = &scalar,
              .reservoir_bias = &scalar }
        };
        const kfcore_esn_deep_model groups[2] = {
            { 2, group0_layers },
            { 2, group1_layers }
        };
        const kfcore_esn_grouped_deep_model grouped_deep = { 2, groups };
        int state_size = -1;
        int workspace_size = -1;

        check_equal(kfcore_esn_grouped_deep_layout(&grouped_deep, &state_size,
                                                   &workspace_size),
                    KFCORE_ESN_OK);
        check_equal(state_size, 8);
        check_equal(workspace_size, 16);
    }

    it("rejects required null layout outputs")
    {
        static const float scalar = 0.0f;
        const kfcore_esn_model layer = {
            .input_size = 1, .reservoir_size = 1, .output_size = 0, .leak_rate = 1.0f,
            .input_weights = &scalar, .reservoir_weights = &scalar,
            .reservoir_bias = &scalar
        };
        const kfcore_esn_deep_model group = { 1, &layer };
        const kfcore_esn_grouped_deep_model grouped_deep = { 1, &group };
        int size = 0;

        check_equal(kfcore_esn_grouped_deep_layout(&grouped_deep, NULL, &size),
                    KFCORE_ESN_INVALID_ARGUMENT);
        check_equal(kfcore_esn_grouped_deep_layout(&grouped_deep, &size, NULL),
                    KFCORE_ESN_INVALID_ARGUMENT);
    }

    it("rejects invalid grouped-deep containers without partial outputs")
    {
        static const float scalar = 0.0f;
        const kfcore_esn_model layer = {
            .input_size = 1, .reservoir_size = 1, .output_size = 0, .leak_rate = 1.0f,
            .input_weights = &scalar, .reservoir_weights = &scalar,
            .reservoir_bias = &scalar
        };
        const kfcore_esn_deep_model group = { 1, &layer };
        const kfcore_esn_grouped_deep_model zero_groups = { 0, &group };
        const kfcore_esn_grouped_deep_model null_groups = { 1, NULL };
        int state_size = -7;
        int workspace_size = -9;

        check_equal(kfcore_esn_grouped_deep_layout(NULL, &state_size, &workspace_size),
                    KFCORE_ESN_INVALID_ARGUMENT);
        check_equal(state_size, -7);
        check_equal(workspace_size, -9);

        check_equal(kfcore_esn_grouped_deep_layout(&zero_groups, &state_size,
                                                   &workspace_size),
                    KFCORE_ESN_INVALID_ARGUMENT);
        check_equal(state_size, -7);
        check_equal(workspace_size, -9);

        check_equal(kfcore_esn_grouped_deep_layout(&null_groups, &state_size,
                                                   &workspace_size),
                    KFCORE_ESN_INVALID_ARGUMENT);
        check_equal(state_size, -7);
        check_equal(workspace_size, -9);
    }

    it("rejects mismatched external input sizes across valid deep groups")
    {
        static const float scalar = 0.0f;
        const kfcore_esn_model group0_layer = {
            .input_size = 1, .reservoir_size = 1, .output_size = 0, .leak_rate = 1.0f,
            .input_weights = &scalar, .reservoir_weights = &scalar,
            .reservoir_bias = &scalar
        };
        const kfcore_esn_model group1_layer = {
            .input_size = 2, .reservoir_size = 1, .output_size = 0, .leak_rate = 1.0f,
            .input_weights = &scalar, .reservoir_weights = &scalar,
            .reservoir_bias = &scalar
        };
        const kfcore_esn_deep_model groups[2] = {
            { 1, &group0_layer },
            { 1, &group1_layer }
        };
        const kfcore_esn_grouped_deep_model grouped_deep = { 2, groups };
        int state_size = -7;
        int workspace_size = -9;

        check_equal(kfcore_esn_grouped_deep_layout(&grouped_deep, &state_size,
                                                   &workspace_size),
                    KFCORE_ESN_INVALID_ARGUMENT);
        check_equal(state_size, -7);
        check_equal(workspace_size, -9);
    }

    it("rejects invalid later deep-group metadata before producing layout outputs")
    {
        static const float scalar = 0.0f;
        const kfcore_esn_model group0_layer = {
            .input_size = 1, .reservoir_size = 1, .output_size = 0, .leak_rate = 1.0f,
            .input_weights = &scalar, .reservoir_weights = &scalar,
            .reservoir_bias = &scalar
        };
        const kfcore_esn_model group1_layer = {
            .input_size = 1, .reservoir_size = 1, .output_size = 0, .leak_rate = 1.0f,
            .input_weights = &scalar, .reservoir_weights = &scalar,
            .reservoir_bias = NULL
        };
        const kfcore_esn_deep_model groups[2] = {
            { 1, &group0_layer },
            { 1, &group1_layer }
        };
        const kfcore_esn_grouped_deep_model grouped_deep = { 2, groups };
        int state_size = -7;
        int workspace_size = -9;

        check_equal(kfcore_esn_grouped_deep_layout(&grouped_deep, &state_size,
                                                   &workspace_size),
                    KFCORE_ESN_INVALID_ARGUMENT);
        check_equal(state_size, -7);
        check_equal(workspace_size, -9);
    }

    it("rejects grouped state-size overflow from individually valid deep groups")
    {
        static const float scalar = 0.0f;
        const int large = INT_MAX / 2;
        const kfcore_esn_model layers[3] = {
            { .input_size = 1, .reservoir_size = large, .output_size = 0, .leak_rate = 1.0f,
              .input_weights = &scalar, .reservoir_weights = &scalar,
              .reservoir_bias = &scalar },
            { .input_size = 1, .reservoir_size = large, .output_size = 0, .leak_rate = 1.0f,
              .input_weights = &scalar, .reservoir_weights = &scalar,
              .reservoir_bias = &scalar },
            { .input_size = 1, .reservoir_size = large, .output_size = 0, .leak_rate = 1.0f,
              .input_weights = &scalar, .reservoir_weights = &scalar,
              .reservoir_bias = &scalar }
        };
        const kfcore_esn_deep_model groups[3] = {
            { 1, &layers[0] },
            { 1, &layers[1] },
            { 1, &layers[2] }
        };
        const kfcore_esn_grouped_deep_model grouped_deep = { 3, groups };
        int state_size = -7;
        int workspace_size = -9;

        check_equal(kfcore_esn_grouped_deep_layout(&grouped_deep, &state_size,
                                                   &workspace_size),
                    KFCORE_ESN_INVALID_ARGUMENT);
        check_equal(state_size, -7);
        check_equal(workspace_size, -9);
    }

    it("rejects grouped workspace-size overflow from valid deep groups")
    {
        static const float scalar = 0.0f;
        const int large = INT_MAX / 3;
        const kfcore_esn_model layers[2] = {
            { .input_size = 1, .reservoir_size = large, .output_size = 0, .leak_rate = 1.0f,
              .input_weights = &scalar, .reservoir_weights = &scalar,
              .reservoir_bias = &scalar },
            { .input_size = 1, .reservoir_size = large, .output_size = 0, .leak_rate = 1.0f,
              .input_weights = &scalar, .reservoir_weights = &scalar,
              .reservoir_bias = &scalar }
        };
        const kfcore_esn_deep_model groups[2] = {
            { 1, &layers[0] },
            { 1, &layers[1] }
        };
        const kfcore_esn_grouped_deep_model grouped_deep = { 2, groups };
        int state_size = -7;
        int workspace_size = -9;

        check_equal(kfcore_esn_grouped_deep_layout(&grouped_deep, &state_size,
                                                   &workspace_size),
                    KFCORE_ESN_INVALID_ARGUMENT);
        check_equal(state_size, -7);
        check_equal(workspace_size, -9);
    }

    it("matches ordinary deep stepping for one group")
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
        const kfcore_esn_deep_model group = { 2, layers };
        const kfcore_esn_grouped_deep_model grouped_deep = { 1, &group };
        const float input[1] = { 0.75f };
        float ordinary_state[3] = { 0.1f, -0.2f, 0.3f };
        float grouped_state[3] = { 0.1f, -0.2f, 0.3f };
        float ordinary_workspace[5] = { 0.0f };
        float grouped_workspace[8] = { 0.0f };

        check_equal(kfcore_esn_deep_step(&group, input, ordinary_state, ordinary_workspace),
                    KFCORE_ESN_OK);
        check_equal(kfcore_esn_grouped_deep_step(&grouped_deep, input, grouped_state,
                                                 grouped_workspace),
                    KFCORE_ESN_OK);
        check_within(grouped_state[0], ordinary_state[0], ESN_GROUP_DEEP_TEST_EPSILON);
        check_within(grouped_state[1], ordinary_state[1], ESN_GROUP_DEEP_TEST_EPSILON);
        check_within(grouped_state[2], ordinary_state[2], ESN_GROUP_DEEP_TEST_EPSILON);
    }

    it("preserves group-major depth-minor state ordering")
    {
        static const float group0_layer0_input_weights[2] = { 1.0f, -1.0f };
        static const float group0_layer0_reservoir_weights[4] = { 0.0f, 0.0f, 0.0f, 0.0f };
        static const float group0_layer0_bias[2] = { 0.0f, 0.0f };
        static const float group0_layer1_input_weights[2] = { 1.0f, 0.5f };
        static const float scalar_zero = 0.0f;
        static const float group1_layer0_input_weight = 0.5f;
        static const float group1_layer1_input_weights[2] = { 1.0f, -1.0f };
        static const float group1_layer1_reservoir_weights[4] = { 0.0f, 0.0f, 0.0f, 0.0f };
        static const float group1_layer1_bias[2] = { 0.0f, 0.0f };
        const kfcore_esn_model group0_layers[2] = {
            { .input_size = 1, .reservoir_size = 2, .output_size = 0, .leak_rate = 1.0f,
              .input_weights = group0_layer0_input_weights,
              .reservoir_weights = group0_layer0_reservoir_weights,
              .reservoir_bias = group0_layer0_bias },
            { .input_size = 2, .reservoir_size = 1, .output_size = 0, .leak_rate = 1.0f,
              .input_weights = group0_layer1_input_weights,
              .reservoir_weights = &scalar_zero,
              .reservoir_bias = &scalar_zero }
        };
        const kfcore_esn_model group1_layers[2] = {
            { .input_size = 1, .reservoir_size = 1, .output_size = 0, .leak_rate = 1.0f,
              .input_weights = &group1_layer0_input_weight,
              .reservoir_weights = &scalar_zero,
              .reservoir_bias = &scalar_zero },
            { .input_size = 1, .reservoir_size = 2, .output_size = 0, .leak_rate = 1.0f,
              .input_weights = group1_layer1_input_weights,
              .reservoir_weights = group1_layer1_reservoir_weights,
              .reservoir_bias = group1_layer1_bias }
        };
        const kfcore_esn_deep_model groups[2] = {
            { 2, group0_layers },
            { 2, group1_layers }
        };
        const kfcore_esn_grouped_deep_model grouped_deep = { 2, groups };
        const float input[1] = { 1.0f };
        float state[6] = { 0.0f };
        float workspace[11] = { 0.0f };
        const float group0_0 = tanhf(1.0f);
        const float group0_1 = -group0_0;
        const float group0_2 = tanhf(group0_0 + 0.5f * group0_1);
        const float group1_0 = tanhf(0.5f);
        const float group1_1 = tanhf(group1_0);
        const float group1_2 = -group1_1;

        check_equal(kfcore_esn_grouped_deep_step(&grouped_deep, input, state, workspace),
                    KFCORE_ESN_OK);
        check_within(state[0], group0_0, ESN_GROUP_DEEP_TEST_EPSILON);
        check_within(state[1], group0_1, ESN_GROUP_DEEP_TEST_EPSILON);
        check_within(state[2], group0_2, ESN_GROUP_DEEP_TEST_EPSILON);
        check_within(state[3], group1_0, ESN_GROUP_DEEP_TEST_EPSILON);
        check_within(state[4], group1_1, ESN_GROUP_DEEP_TEST_EPSILON);
        check_within(state[5], group1_2, ESN_GROUP_DEEP_TEST_EPSILON);
    }

    it("preserves same-timestep propagation independently inside each group")
    {
        static const float one = 1.0f;
        static const float half = 0.5f;
        static const float zero = 0.0f;
        const kfcore_esn_model group0_layers[2] = {
            { .input_size = 1, .reservoir_size = 1, .output_size = 0, .leak_rate = 1.0f,
              .input_weights = &one, .reservoir_weights = &zero,
              .reservoir_bias = &zero },
            { .input_size = 1, .reservoir_size = 1, .output_size = 0, .leak_rate = 1.0f,
              .input_weights = &one, .reservoir_weights = &zero,
              .reservoir_bias = &zero }
        };
        const kfcore_esn_model group1_layers[2] = {
            { .input_size = 1, .reservoir_size = 1, .output_size = 0, .leak_rate = 1.0f,
              .input_weights = &half, .reservoir_weights = &zero,
              .reservoir_bias = &zero },
            { .input_size = 1, .reservoir_size = 1, .output_size = 0, .leak_rate = 1.0f,
              .input_weights = &one, .reservoir_weights = &zero,
              .reservoir_bias = &zero }
        };
        const kfcore_esn_deep_model groups[2] = {
            { 2, group0_layers },
            { 2, group1_layers }
        };
        const kfcore_esn_grouped_deep_model grouped_deep = { 2, groups };
        const float input[1] = { 1.0f };
        float state[4] = { -0.5f, 0.0f, -0.25f, 0.0f };
        float workspace[7] = { 0.0f };
        const float group0_l0 = tanhf(1.0f);
        const float group0_l1 = tanhf(group0_l0);
        const float group1_l0 = tanhf(0.5f);
        const float group1_l1 = tanhf(group1_l0);

        check_equal(kfcore_esn_grouped_deep_step(&grouped_deep, input, state, workspace),
                    KFCORE_ESN_OK);
        check_within(state[0], group0_l0, ESN_GROUP_DEEP_TEST_EPSILON);
        check_within(state[1], group0_l1, ESN_GROUP_DEEP_TEST_EPSILON);
        check_within(state[2], group1_l0, ESN_GROUP_DEEP_TEST_EPSILON);
        check_within(state[3], group1_l1, ESN_GROUP_DEEP_TEST_EPSILON);
        check_equal(state[1] > 0.0f, 1);
        check_equal(state[3] > 0.0f, 1);
    }

    it("rejects mismatched group inputs before advancing caller state")
    {
        static const float one = 1.0f;
        static const float zero = 0.0f;
        static const float two_inputs[2] = { 1.0f, 1.0f };
        const kfcore_esn_model layers[2] = {
            { .input_size = 1, .reservoir_size = 1, .output_size = 0, .leak_rate = 1.0f,
              .input_weights = &one, .reservoir_weights = &zero,
              .reservoir_bias = &zero },
            { .input_size = 2, .reservoir_size = 1, .output_size = 0, .leak_rate = 1.0f,
              .input_weights = two_inputs, .reservoir_weights = &zero,
              .reservoir_bias = &zero }
        };
        const kfcore_esn_deep_model groups[2] = {
            { 1, &layers[0] },
            { 1, &layers[1] }
        };
        const kfcore_esn_grouped_deep_model grouped_deep = { 2, groups };
        const float input[1] = { 1.0f };
        float state[2] = { 0.25f, -0.75f };
        float before[2];
        float workspace[6] = { 0.0f };
        memcpy(before, state, sizeof(state));

        check_equal(kfcore_esn_grouped_deep_step(&grouped_deep, input, state, workspace),
                    KFCORE_ESN_INVALID_ARGUMENT);
        check_equal(memcmp(state, before, sizeof(state)), 0);
    }

    it("rejects invalid later group metadata before advancing caller state")
    {
        static const float one = 1.0f;
        static const float zero = 0.0f;
        const kfcore_esn_model layers[2] = {
            { .input_size = 1, .reservoir_size = 1, .output_size = 0, .leak_rate = 1.0f,
              .input_weights = &one, .reservoir_weights = &zero,
              .reservoir_bias = &zero },
            { .input_size = 1, .reservoir_size = 1, .output_size = 0, .leak_rate = 1.0f,
              .input_weights = &one, .reservoir_weights = &zero,
              .reservoir_bias = NULL }
        };
        const kfcore_esn_deep_model groups[2] = {
            { 1, &layers[0] },
            { 1, &layers[1] }
        };
        const kfcore_esn_grouped_deep_model grouped_deep = { 2, groups };
        const float input[1] = { 1.0f };
        float state[2] = { 0.25f, -0.75f };
        float before[2];
        float workspace[6] = { 0.0f };
        memcpy(before, state, sizeof(state));

        check_equal(kfcore_esn_grouped_deep_step(&grouped_deep, input, state, workspace),
                    KFCORE_ESN_INVALID_ARGUMENT);
        check_equal(memcmp(state, before, sizeof(state)), 0);
    }

    it("rejects null grouped-deep runtime arguments without mutation")
    {
        static const float one = 1.0f;
        static const float zero = 0.0f;
        const kfcore_esn_model layer = {
            .input_size = 1, .reservoir_size = 1, .output_size = 0, .leak_rate = 1.0f,
            .input_weights = &one, .reservoir_weights = &zero,
            .reservoir_bias = &zero
        };
        const kfcore_esn_deep_model group = { 1, &layer };
        const kfcore_esn_grouped_deep_model grouped_deep = { 1, &group };
        const float input[1] = { 1.0f };
        float state[1] = { 0.25f };
        const float before = state[0];
        float workspace[3] = { 0.0f };

        check_equal(kfcore_esn_grouped_deep_step(NULL, input, state, workspace),
                    KFCORE_ESN_INVALID_ARGUMENT);
        check_within(state[0], before, ESN_GROUP_DEEP_TEST_EPSILON);

        check_equal(kfcore_esn_grouped_deep_step(&grouped_deep, NULL, state, workspace),
                    KFCORE_ESN_INVALID_ARGUMENT);
        check_within(state[0], before, ESN_GROUP_DEEP_TEST_EPSILON);

        check_equal(kfcore_esn_grouped_deep_step(&grouped_deep, input, NULL, workspace),
                    KFCORE_ESN_INVALID_ARGUMENT);

        check_equal(kfcore_esn_grouped_deep_step(&grouped_deep, input, state, NULL),
                    KFCORE_ESN_INVALID_ARGUMENT);
        check_within(state[0], before, ESN_GROUP_DEEP_TEST_EPSILON);
    }
}
