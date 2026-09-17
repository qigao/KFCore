#include <limits.h>
#include <stddef.h>

#include "esn.h"
#include "esn_deep.h"
#define TINYTEST_NO_MAIN
#include "tinytest.h"

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
}
