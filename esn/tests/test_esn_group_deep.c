#include <limits.h>
#include <stddef.h>

#include "esn.h"
#include "esn_deep.h"
#include "esn_group_deep.h"
#define TINYTEST_NO_MAIN
#include "tinytest.h"

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
}
