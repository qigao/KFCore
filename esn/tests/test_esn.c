#include <float.h>
#include <math.h>
#include <string.h>

#include "esn.h"
#include "tinytest.h"

#define ESN_TEST_EPSILON 1.0e-5f

static kfcore_esn_model make_test_model(void)
{
    static const float input_weights[2] = { 1.0f, -1.0f };
    static const float reservoir_weights[4] = { 0.0f, 0.0f, 0.0f, 0.0f };
    static const float reservoir_bias[2] = { 0.0f, 0.0f };
    static const float output_weights[2] = { 2.0f, -1.0f };
    static const float output_bias[1] = { 0.25f };

    const kfcore_esn_model model = {
        1, 2, 1, 0.5f, input_weights, reservoir_weights, reservoir_bias, output_weights, output_bias
    };
    return model;
}

spec("kfcore esn")
{
    it("advances a leaky reservoir deterministically")
    {
        const kfcore_esn_model model = make_test_model();
        const float input[1] = { 1.0f };
        float state[2] = { 0.0f, 0.0f };
        float workspace[2] = { 0.0f, 0.0f };
        const float expected = 0.5f * tanhf(1.0f);

        check_equal(kfcore_esn_step(&model, input, state, workspace), KFCORE_ESN_OK);
        check_within(state[0], expected, ESN_TEST_EPSILON);
        check_within(state[1], -expected, ESN_TEST_EPSILON);
    }

    it("uses recurrent weights with column-major storage")
    {
        static const float input_weights[2] = { 0.0f, 0.0f };
        static const float reservoir_weights[4] = {
            1.0f, 3.0f,
            2.0f, 4.0f
        };
        static const float reservoir_bias[2] = { 0.0f, 0.0f };
        static const float output_weights[2] = { 0.0f, 0.0f };
        static const float output_bias[1] = { 0.0f };
        const kfcore_esn_model model = {
            1, 2, 1, 1.0f, input_weights, reservoir_weights, reservoir_bias, output_weights, output_bias
        };
        const float input[1] = { 0.0f };
        float state[2] = { 0.2f, -0.4f };
        float workspace[2] = { 0.0f, 0.0f };

        check_equal(kfcore_esn_step(&model, input, state, workspace), KFCORE_ESN_OK);
        check_within(state[0], tanhf(-0.6f), ESN_TEST_EPSILON);
        check_within(state[1], tanhf(-1.0f), ESN_TEST_EPSILON);
    }

    it("applies the configured linear readout")
    {
        const kfcore_esn_model model = make_test_model();
        const float state[2] = { 0.5f, -0.25f };
        float output[1] = { 0.0f };

        check_equal(kfcore_esn_predict(&model, state, output), KFCORE_ESN_OK);
        check_within(output[0], 1.5f, ESN_TEST_EPSILON);
    }

    it("does not advance state when step-predict readout validation fails")
    {
        static const float input_weights[1] = { 1.0f };
        static const float reservoir_weights[1] = { 0.0f };
        static const float reservoir_bias[1] = { 0.0f };
        static const float output_bias[1] = { 0.0f };
        const kfcore_esn_model model = {
            1, 1, 1, 1.0f, input_weights, reservoir_weights, reservoir_bias, NULL, output_bias
        };
        const float input[1] = { 1.0f };
        float state[1] = { 0.0f };
        float workspace[1] = { 0.0f };
        float output[1] = { 0.0f };

        check_equal(kfcore_esn_step_predict(&model, input, state, workspace, output),
                    KFCORE_ESN_INVALID_ARGUMENT);
        check_within(state[0], 0.0f, ESN_TEST_EPSILON);
    }

    it("fits a ridge readout without an explicit inverse")
    {
        const float states[3] = { 1.0f, 2.0f, 3.0f };
        const float targets[3] = { 2.0f, 4.0f, 6.0f };
        float output_weights[1] = { 0.0f };
        float gram[1] = { 0.0f };

        check_equal(kfcore_esn_fit_ridge(states, targets, 1, 1, 3, 1.0e-5f, output_weights, gram),
                    KFCORE_ESN_OK);
        check_within(output_weights[0], 2.0f, 1.0e-4f);
    }

    it("fits multi-output ridge weights with the documented matrix layout")
    {
        const float states[8] = {
            1.0f, 0.0f,
            0.0f, 1.0f,
            1.0f, 0.0f,
            0.0f, 1.0f
        };
        const float targets[8] = {
            2.0f, 0.5f,
            -1.0f, 3.0f,
            2.0f, 0.5f,
            -1.0f, 3.0f
        };
        float output_weights[4] = { 0.0f, 0.0f, 0.0f, 0.0f };
        float gram[4] = { 0.0f, 0.0f, 0.0f, 0.0f };

        check_equal(kfcore_esn_fit_ridge(states, targets, 2, 2, 4, 1.0e-5f, output_weights, gram),
                    KFCORE_ESN_OK);
        check_within(output_weights[0], 2.0f, 1.0e-4f);
        check_within(output_weights[1], 0.5f, 1.0e-4f);
        check_within(output_weights[2], -1.0f, 1.0e-4f);
        check_within(output_weights[3], 3.0f, 1.0e-4f);
    }

    it("initializes reservoir weights deterministically from an explicit seed")
    {
        float first[16] = { 0.0f };
        float repeat[16] = { 0.0f };
        float other[16] = { 0.0f };

        check_equal(kfcore_esn_init_reservoir_weights(first, 4, UINT64_C(42), 0.5f),
                    KFCORE_ESN_OK);
        check_equal(kfcore_esn_init_reservoir_weights(repeat, 4, UINT64_C(42), 0.5f),
                    KFCORE_ESN_OK);
        check_equal(kfcore_esn_init_reservoir_weights(other, 4, UINT64_C(43), 0.5f),
                    KFCORE_ESN_OK);
        check_equal(memcmp(first, repeat, sizeof(first)), 0);
        check(memcmp(first, other, sizeof(first)) != 0,
              "different reservoir seeds must produce different matrices");

        int zero_count = 0;
        int nonzero_count = 0;
        for (int i = 0; i < 16; ++i)
        {
            if (first[i] == 0.0f)
            {
                ++zero_count;
            }
            else
            {
                ++nonzero_count;
            }
        }
        check(zero_count > 0, "sparse reservoir should contain masked entries");
        check(nonzero_count > 0, "sparse reservoir should retain active entries");
    }

    it("uses a stable dense reservoir random sequence")
    {
        float weights[4] = { 0.0f, 0.0f, 0.0f, 0.0f };

        check_equal(kfcore_esn_init_reservoir_weights(weights, 2, UINT64_C(42), 1.0f),
                    KFCORE_ESN_OK);
        check_within(weights[0], -0.6801792383f, 1.0e-7f);
        check_within(weights[1], -0.3116186857f, 1.0e-7f);
        check_within(weights[2], 0.7364560366f, 1.0e-7f);
        check_within(weights[3], 0.6012636423f, 1.0e-7f);
    }

    it("estimates spectral radius for a rotation-scale matrix")
    {
        const float weights[4] = {
            0.0f, 2.0f,
            -2.0f, 0.0f
        };
        float workspace[4] = { 0.0f, 0.0f, 0.0f, 0.0f };
        float radius = 0.0f;

        check_equal(kfcore_esn_estimate_spectral_radius(weights, 2, 32, workspace, &radius),
                    KFCORE_ESN_OK);
        check_within(radius, 2.0f, 1.0e-5f);
    }

    it("scales a reservoir to the requested spectral radius estimate")
    {
        float weights[4] = {
            0.0f, 2.0f,
            -2.0f, 0.0f
        };
        float workspace[4] = { 0.0f, 0.0f, 0.0f, 0.0f };
        float radius = 0.0f;

        check_equal(kfcore_esn_scale_spectral_radius(weights, 2, 0.9f, 32, workspace),
                    KFCORE_ESN_OK);
        check_equal(kfcore_esn_estimate_spectral_radius(weights, 2, 32, workspace, &radius),
                    KFCORE_ESN_OK);
        check_within(radius, 0.9f, 1.0e-5f);
    }

    it("rejects invalid reservoir construction arguments and zero radius")
    {
        float weights[4] = { 0.0f, 0.0f, 0.0f, 0.0f };
        float workspace[4] = { 0.0f, 0.0f, 0.0f, 0.0f };
        float radius = 0.0f;

        check_equal(kfcore_esn_init_reservoir_weights(weights, 2, UINT64_C(1), 0.0f),
                    KFCORE_ESN_INVALID_ARGUMENT);
        check_equal(kfcore_esn_init_reservoir_weights(weights, 2, UINT64_C(1), 1.1f),
                    KFCORE_ESN_INVALID_ARGUMENT);
        check_equal(kfcore_esn_estimate_spectral_radius(weights, 2, 0, workspace, &radius),
                    KFCORE_ESN_INVALID_ARGUMENT);
        check_equal(kfcore_esn_estimate_spectral_radius(weights, 2, 8, workspace, &radius),
                    KFCORE_ESN_NUMERICAL_FAILURE);
        check_equal(kfcore_esn_scale_spectral_radius(weights, 2, 1.0f, 8, workspace),
                    KFCORE_ESN_NUMERICAL_FAILURE);
        check_equal(kfcore_esn_scale_spectral_radius(weights, 2, 0.0f, 8, workspace),
                    KFCORE_ESN_INVALID_ARGUMENT);
    }

    it("fails fast on invalid runtime and fit arguments")
    {
        kfcore_esn_model model = make_test_model();
        const float input[1] = { 1.0f };
        float state[2] = { 0.0f, 0.0f };
        float workspace[2] = { 0.0f, 0.0f };
        const float states[1] = { 1.0f };
        const float targets[1] = { 1.0f };
        const float invalid_states[1] = { NAN };
        float output_weights[1] = { 0.0f };
        float gram[1] = { 0.0f };

        model.leak_rate = 0.0f;
        check_equal(kfcore_esn_step(&model, input, state, workspace), KFCORE_ESN_INVALID_ARGUMENT);
        check_equal(kfcore_esn_step(NULL, input, state, workspace), KFCORE_ESN_INVALID_ARGUMENT);
        check_equal(kfcore_esn_fit_ridge(states, targets, 1, 1, 1, 0.0f, output_weights, gram),
                    KFCORE_ESN_INVALID_ARGUMENT);
        check_equal(kfcore_esn_fit_ridge(invalid_states, targets, 1, 1, 1, 1.0f, output_weights,
                                         gram),
                    KFCORE_ESN_INVALID_ARGUMENT);
    }

    it("reports numerical failure instead of changing solvers")
    {
        const float states[1] = { FLT_MAX };
        const float targets[1] = { 1.0f };
        float output_weights[1] = { 0.0f };
        float gram[1] = { 0.0f };

        check_equal(kfcore_esn_fit_ridge(states, targets, 1, 1, 1, 1.0f, output_weights, gram),
                    KFCORE_ESN_NUMERICAL_FAILURE);
    }
}
