#include <float.h>
#include <math.h>
#include <string.h>

#include "esn.h"
#define TINYTEST_NO_MAIN
#include "tinytest.h"

#define ESN_RLS_EPSILON 1.0e-5f

spec("kfcore esn rls")
{
    it("commits finite weights when the float rank-one product would overflow")
    {
        const float state = 0.25f;
        const float target = 1.5e38f;
        float weight = -2.0e38f;
        float correlation = 16.0f;
        float workspace[2] = { 0.0f, 0.0f };
        check_equal(kfcore_esn_rls_update(&state, &target, 1, 1, 1.0f,
                                          &weight, &correlation, workspace), KFCORE_ESN_OK);
        check(isfinite(weight));
        check_within(weight / 2.0e38f, 1.0f, ESN_RLS_EPSILON);
        check_equal(correlation, 8.0f);
    }

    it("preserves both matrices when a later weight update overflows")
    {
        const float state = 0.25f;
        const float target[2] = { 1.0f, 2.0e38f };
        float weights[2] = { 0.0f, 0.0f };
        const float original[2] = { 0.0f, 0.0f };
        float correlation = 16.0f;
        float workspace[3] = { 0.0f };
        check_equal(kfcore_esn_rls_update(&state, target, 1, 2, 1.0f,
                                          weights, &correlation, workspace),
                    KFCORE_ESN_NUMERICAL_FAILURE);
        check_equal(memcmp(weights, original, sizeof(weights)), 0);
        check_equal(correlation, 16.0f);
    }

    it("initializes inverse correlation from delta")
    {
        float inverse_correlation[4] = { 7.0f, 7.0f, 7.0f, 7.0f };

        check_equal(kfcore_esn_rls_init(inverse_correlation, 2, 4.0f), KFCORE_ESN_OK);
        check_within(inverse_correlation[0], 0.25f, ESN_RLS_EPSILON);
        check_within(inverse_correlation[1], 0.0f, ESN_RLS_EPSILON);
        check_within(inverse_correlation[2], 0.0f, ESN_RLS_EPSILON);
        check_within(inverse_correlation[3], 0.25f, ESN_RLS_EPSILON);
    }

    it("matches an analytical scalar RLS update")
    {
        const float state[1] = { 2.0f };
        const float target[1] = { 4.0f };
        float output_weights[1] = { 0.0f };
        float inverse_correlation[1] = { 1.0f };
        float workspace[2] = { 0.0f, 0.0f };

        check_equal(kfcore_esn_rls_update(state, target, 1, 1, 1.0f, output_weights,
                                          inverse_correlation, workspace),
                    KFCORE_ESN_OK);
        check_within(output_weights[0], 1.6f, ESN_RLS_EPSILON);
        check_within(inverse_correlation[0], 0.2f, ESN_RLS_EPSILON);
    }

    it("updates multi-output weights in column-major layout")
    {
        const float state[2] = { 1.0f, 2.0f };
        const float target[2] = { 3.0f, -1.0f };
        float output_weights[4] = { 0.0f, 0.0f, 0.0f, 0.0f };
        float inverse_correlation[4] = {
            1.0f, 0.0f,
            0.0f, 1.0f
        };
        float workspace[4] = { 0.0f, 0.0f, 0.0f, 0.0f };

        check_equal(kfcore_esn_rls_update(state, target, 2, 2, 1.0f, output_weights,
                                          inverse_correlation, workspace),
                    KFCORE_ESN_OK);

        check_within(output_weights[0], 0.5f, ESN_RLS_EPSILON);
        check_within(output_weights[1], -1.0f / 6.0f, ESN_RLS_EPSILON);
        check_within(output_weights[2], 1.0f, ESN_RLS_EPSILON);
        check_within(output_weights[3], -1.0f / 3.0f, ESN_RLS_EPSILON);

        check_within(inverse_correlation[0], 5.0f / 6.0f, ESN_RLS_EPSILON);
        check_within(inverse_correlation[1], -1.0f / 3.0f, ESN_RLS_EPSILON);
        check_within(inverse_correlation[2], -1.0f / 3.0f, ESN_RLS_EPSILON);
        check_within(inverse_correlation[3], 1.0f / 3.0f, ESN_RLS_EPSILON);
    }

    it("applies the configured forgetting factor")
    {
        const float state[1] = { 1.0f };
        const float target[1] = { 0.0f };
        float output_weights[1] = { 0.0f };
        float inverse_correlation[1] = { 1.0f };
        float workspace[2] = { 0.0f, 0.0f };

        check_equal(kfcore_esn_rls_update(state, target, 1, 1, 0.5f, output_weights,
                                          inverse_correlation, workspace),
                    KFCORE_ESN_OK);
        check_within(output_weights[0], 0.0f, ESN_RLS_EPSILON);
        check_within(inverse_correlation[0], 2.0f / 3.0f, ESN_RLS_EPSILON);
    }

    it("keeps a mathematically zero correlation finite for tiny forgetting")
    {
        const float state[1] = { 1.0f };
        const float target[1] = { 0.0f };
        float output_weights[1] = { 0.0f };
        float inverse_correlation[1] = { 0.0f };
        float workspace[2] = { 0.0f, 0.0f };
        const float tiny_forgetting = nextafterf(0.0f, 1.0f);

        check(tiny_forgetting > 0.0f, "test requires a positive subnormal float");
        check_equal(kfcore_esn_rls_update(state, target, 1, 1, tiny_forgetting,
                                          output_weights, inverse_correlation, workspace),
                    KFCORE_ESN_OK);
        check(isfinite(inverse_correlation[0]), "tiny forgetting must not create NaN/Inf");
        check_within(inverse_correlation[0], 0.0f, ESN_RLS_EPSILON);
    }

    it("converges online toward a known scalar mapping")
    {
        float output_weights[1] = { 0.0f };
        float inverse_correlation[1] = { 0.0f };
        float workspace[2] = { 0.0f, 0.0f };

        check_equal(kfcore_esn_rls_init(inverse_correlation, 1, 1.0f), KFCORE_ESN_OK);
        for (int pass = 0; pass < 8; ++pass)
        {
            for (int sample = 1; sample <= 4; ++sample)
            {
                const float state[1] = { (float)sample };
                const float target[1] = { 2.0f * (float)sample };
                check_equal(kfcore_esn_rls_update(state, target, 1, 1, 1.0f, output_weights,
                                                  inverse_correlation, workspace),
                            KFCORE_ESN_OK);
                check(isfinite(output_weights[0]), "RLS weight must remain finite");
                check(isfinite(inverse_correlation[0]), "RLS state must remain finite");
            }
        }
        check_within(output_weights[0], 2.0f, 2.0e-2f);
    }

    it("rejects invalid RLS updates without partial mutation")
    {
        const float state[1] = { 1.0f };
        const float target[1] = { 2.0f };
        const float invalid_target[1] = { NAN };
        float output_weights[1] = { 0.25f };
        float inverse_correlation[1] = { -1.0f };
        float workspace[2] = { 0.0f, 0.0f };
        const float original_weight = output_weights[0];
        const float original_correlation = inverse_correlation[0];

        check_equal(kfcore_esn_rls_update(state, target, 1, 1, 1.0f, output_weights,
                                          inverse_correlation, workspace),
                    KFCORE_ESN_NUMERICAL_FAILURE);
        check_within(output_weights[0], original_weight, ESN_RLS_EPSILON);
        check_within(inverse_correlation[0], original_correlation, ESN_RLS_EPSILON);

        check_equal(kfcore_esn_rls_update(state, invalid_target, 1, 1, 1.0f, output_weights,
                                          inverse_correlation, workspace),
                    KFCORE_ESN_INVALID_ARGUMENT);
        check_within(output_weights[0], original_weight, ESN_RLS_EPSILON);
        check_within(inverse_correlation[0], original_correlation, ESN_RLS_EPSILON);

        check_equal(kfcore_esn_rls_update(state, target, 1, 1, 0.0f, output_weights,
                                          inverse_correlation, workspace),
                    KFCORE_ESN_INVALID_ARGUMENT);
        check_equal(kfcore_esn_rls_init(inverse_correlation, 1, 0.0f),
                    KFCORE_ESN_INVALID_ARGUMENT);
    }
}
