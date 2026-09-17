#include <limits.h>

#include "esn.h"
#include "esn_group.h"
#define TINYTEST_NO_MAIN
#include "tinytest.h"

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
}
