#include <float.h>
#include <math.h>

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
