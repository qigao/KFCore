/** @file kalman_ekf.c
 * KFCore
 * @author Jan Zwiener (jan@zwiener.org)
 *
 * @brief Extended Kalman Filter helpers
 * @{ */

#include <math.h>
#include <string.h>

#include "kalman_ekf.h"
#include "kalman_takasu.h"
#include "kalman_udu.h"
#include "kalman_workspace_internal.h"
#include "linalg.h"

static kfcore_kalman_status add_workspace(size_t a, size_t b, size_t* total)
{
    return kfcore_kalman_checked_add(a, b, total);
}

kfcore_kalman_status kalman_ekf_takasu_predict_workspace_floats(size_t n, size_t r,
                                                                size_t* required)
{
    size_t inner;
    size_t nn;
    size_t total;
    kfcore_kalman_status status;

    if (!required)
    {
        return KFCORE_KALMAN_INVALID_ARGUMENT;
    }
    status = kalman_predict_workspace_floats(n, r, &inner);
    if (status != KFCORE_KALMAN_OK)
    {
        return status;
    }
    if ((status = kfcore_kalman_checked_mul(n, n, &nn)) != KFCORE_KALMAN_OK ||
        (status = add_workspace(n, nn, &total)) != KFCORE_KALMAN_OK ||
        (status = add_workspace(total, inner, required)) != KFCORE_KALMAN_OK)
    {
        return status;
    }
    return KFCORE_KALMAN_OK;
}

kfcore_kalman_status kalman_ekf_takasu_predict(
    float* x, float* P, kalman_ekf_transition_fn transition,
    const float* G, const float* Q, size_t n, size_t r, void* user,
    float* workspace, size_t workspace_floats)
{
    size_t required;
    size_t nn;
    size_t inner_required;
    kfcore_kalman_status status =
        kalman_ekf_takasu_predict_workspace_floats(n, r, &required);

    if (status != KFCORE_KALMAN_OK)
    {
        return status;
    }
    if (!x || !P || !transition || (r > 0U && (!G || !Q)))
    {
        return KFCORE_KALMAN_INVALID_ARGUMENT;
    }
    status = kfcore_kalman_require_workspace(workspace, workspace_floats, required);
    if (status != KFCORE_KALMAN_OK)
    {
        return status;
    }

    (void)kfcore_kalman_checked_mul(n, n, &nn);
    (void)kalman_predict_workspace_floats(n, r, &inner_required);

    float* x_pred = workspace;
    float* Phi = x_pred + n;
    float* inner = Phi + nn;

    if (transition(x_pred, Phi, x, (int)n, user) != 0)
    {
        return KFCORE_KALMAN_CALLBACK_FAILURE;
    }

    status = kalman_predict(NULL, P, Phi, G, Q, n, r, inner, inner_required);
    if (status != KFCORE_KALMAN_OK)
    {
        return status;
    }

    memcpy(x, x_pred, sizeof(float) * n);
    return KFCORE_KALMAN_OK;
}

kfcore_kalman_status kalman_ekf_takasu_update_workspace_floats(size_t n, size_t m,
                                                               size_t* required)
{
    size_t inner;
    size_t nm;
    size_t total;
    kfcore_kalman_status status;

    if (!required)
    {
        return KFCORE_KALMAN_INVALID_ARGUMENT;
    }
    status = kalman_takasu_workspace_floats(n, m, &inner);
    if (status != KFCORE_KALMAN_OK)
    {
        return status;
    }
    if ((status = kfcore_kalman_checked_mul(n, m, &nm)) != KFCORE_KALMAN_OK ||
        (status = add_workspace(m, m, &total)) != KFCORE_KALMAN_OK ||
        (status = add_workspace(total, nm, &total)) != KFCORE_KALMAN_OK ||
        (status = add_workspace(total, inner, required)) != KFCORE_KALMAN_OK)
    {
        return status;
    }
    return KFCORE_KALMAN_OK;
}

kfcore_kalman_status kalman_ekf_takasu_update(
    float* x, float* P, const float* z, const float* R,
    kalman_ekf_measurement_fn measurement, size_t n, size_t m,
    float chi2_threshold, float* chi2, void* user,
    float* workspace, size_t workspace_floats)
{
    size_t required;
    size_t nm;
    size_t inner_required;
    kfcore_kalman_status status =
        kalman_ekf_takasu_update_workspace_floats(n, m, &required);

    if (status != KFCORE_KALMAN_OK)
    {
        return status;
    }
    if (!x || !P || !z || !R || !measurement)
    {
        return KFCORE_KALMAN_INVALID_ARGUMENT;
    }
    status = kfcore_kalman_require_workspace(workspace, workspace_floats, required);
    if (status != KFCORE_KALMAN_OK)
    {
        return status;
    }

    (void)kfcore_kalman_checked_mul(n, m, &nm);
    (void)kalman_takasu_workspace_floats(n, m, &inner_required);

    float* z_pred = workspace;
    float* dz = z_pred + m;
    float* Ht = dz + m;
    float* inner = Ht + nm;

    if (measurement(z_pred, Ht, x, (int)n, (int)m, user) != 0)
    {
        return KFCORE_KALMAN_CALLBACK_FAILURE;
    }

    for (size_t i = 0U; i < m; ++i)
    {
        dz[i] = z[i] - z_pred[i];
    }

    return kalman_takasu(x, P, dz, R, Ht, n, m, chi2_threshold, chi2,
                         inner, inner_required);
}

kfcore_kalman_status kalman_ekf_udu_predict_workspace_floats(size_t n, size_t r,
                                                             size_t* required)
{
    size_t inner;
    size_t nn;
    size_t total;
    kfcore_kalman_status status;

    if (!required)
    {
        return KFCORE_KALMAN_INVALID_ARGUMENT;
    }
    status = kalman_udu_predict_workspace_floats(n, r, &inner);
    if (status != KFCORE_KALMAN_OK)
    {
        return status;
    }
    if ((status = kfcore_kalman_checked_mul(n, n, &nn)) != KFCORE_KALMAN_OK ||
        (status = add_workspace(n, nn, &total)) != KFCORE_KALMAN_OK ||
        (status = add_workspace(total, inner, required)) != KFCORE_KALMAN_OK)
    {
        return status;
    }
    return KFCORE_KALMAN_OK;
}

kfcore_kalman_status kalman_ekf_udu_predict(
    float* x, float* U, float* d, kalman_ekf_transition_fn transition,
    const float* G, const float* Q, size_t n, size_t r, void* user,
    float* workspace, size_t workspace_floats)
{
    size_t required;
    size_t nn;
    size_t inner_required;
    kfcore_kalman_status status =
        kalman_ekf_udu_predict_workspace_floats(n, r, &required);

    if (status != KFCORE_KALMAN_OK)
    {
        return status;
    }
    if (!x || !U || !d || !transition || (r > 0U && (!G || !Q)))
    {
        return KFCORE_KALMAN_INVALID_ARGUMENT;
    }
    status = kfcore_kalman_require_workspace(workspace, workspace_floats, required);
    if (status != KFCORE_KALMAN_OK)
    {
        return status;
    }

    (void)kfcore_kalman_checked_mul(n, n, &nn);
    (void)kalman_udu_predict_workspace_floats(n, r, &inner_required);

    float* x_pred = workspace;
    float* Phi = x_pred + n;
    float* inner = Phi + nn;

    if (transition(x_pred, Phi, x, (int)n, user) != 0)
    {
        return KFCORE_KALMAN_CALLBACK_FAILURE;
    }

    status = kalman_udu_predict(NULL, U, d, Phi, G, Q, n, r, inner, inner_required);
    if (status != KFCORE_KALMAN_OK)
    {
        return status;
    }

    memcpy(x, x_pred, sizeof(float) * n);
    return KFCORE_KALMAN_OK;
}

kfcore_kalman_status kalman_ekf_udu_update_workspace_floats(size_t n, size_t m,
                                                            size_t* required)
{
    size_t scalar;
    size_t nm;
    size_t mm;
    size_t total;
    kfcore_kalman_status status;

    if (!required)
    {
        return KFCORE_KALMAN_INVALID_ARGUMENT;
    }
    status = kalman_udu_scalar_workspace_floats(n, &scalar);
    if (status != KFCORE_KALMAN_OK)
    {
        return status;
    }
    if ((status = kfcore_kalman_check_dim(m, 0)) != KFCORE_KALMAN_OK ||
        (status = kfcore_kalman_checked_mul(n, m, &nm)) != KFCORE_KALMAN_OK ||
        (status = kfcore_kalman_checked_mul(m, m, &mm)) != KFCORE_KALMAN_OK ||
        (status = add_workspace(m, m, &total)) != KFCORE_KALMAN_OK ||
        (status = add_workspace(total, nm, &total)) != KFCORE_KALMAN_OK ||
        (status = add_workspace(total, mm, &total)) != KFCORE_KALMAN_OK ||
        (status = add_workspace(total, scalar, required)) != KFCORE_KALMAN_OK)
    {
        return status;
    }
    return KFCORE_KALMAN_OK;
}

kfcore_kalman_status kalman_ekf_udu_update(
    float* x, float* U, float* d, const float* z, const float* R,
    kalman_ekf_measurement_fn measurement, size_t n, size_t m,
    float chi2_threshold, int downweight_outlier, void* user,
    float* workspace, size_t workspace_floats)
{
    size_t required;
    size_t nm;
    size_t mm;
    size_t scalar_required;
    kfcore_kalman_status status =
        kalman_ekf_udu_update_workspace_floats(n, m, &required);

    if (status != KFCORE_KALMAN_OK)
    {
        return status;
    }
    if (!x || !U || !d || !z || !R || !measurement)
    {
        return KFCORE_KALMAN_INVALID_ARGUMENT;
    }
    status = kfcore_kalman_require_workspace(workspace, workspace_floats, required);
    if (status != KFCORE_KALMAN_OK)
    {
        return status;
    }

    (void)kfcore_kalman_checked_mul(n, m, &nm);
    (void)kfcore_kalman_checked_mul(m, m, &mm);
    (void)kalman_udu_scalar_workspace_floats(n, &scalar_required);

    float* z_pred = workspace;
    float* dz = z_pred + m;
    float* Ht = dz + m;
    float* R_work = Ht + nm;
    float* scalar_workspace = R_work + mm;

    if (measurement(z_pred, Ht, x, (int)n, (int)m, user) != 0)
    {
        return KFCORE_KALMAN_CALLBACK_FAILURE;
    }

    for (size_t i = 0U; i < m; ++i)
    {
        dz[i] = z[i] - z_pred[i];
    }

    memcpy(R_work, R, sizeof(float) * mm);
    status = decorrelate(dz, Ht, R_work, n, m);
    if (status != KFCORE_KALMAN_OK)
    {
        return status;
    }

    for (size_t i = 0U; i < m; ++i)
    {
        float Rv = 1.0f;
        const float* H_line = Ht + i * n;

        if (chi2_threshold > 0.0f)
        {
            float HPHT = 0.0f;
            matmul("N", "N", 1, (int)n, (int)n, 1.0f,
                   H_line, U, 0.0f, scalar_workspace);
            for (size_t j = 0U; j < n; ++j)
            {
                HPHT += scalar_workspace[j] * scalar_workspace[j] * d[j];
            }

            const float innovation_variance = HPHT + Rv;
            if (!(innovation_variance > 0.0f) || !isfinite(innovation_variance))
            {
                return KFCORE_KALMAN_NUMERICAL_FAILURE;
            }

            const float mahalanobis_distance_sq =
                dz[i] * dz[i] / innovation_variance;
            if (mahalanobis_distance_sq > chi2_threshold)
            {
                if (!downweight_outlier)
                {
                    continue;
                }
                const float factor = mahalanobis_distance_sq / chi2_threshold;
                Rv = (factor - 1.0f) * HPHT + factor * Rv;
            }
        }

        status = kalman_udu_scalar(x, U, d, dz[i], Rv, H_line, n,
                                   scalar_workspace, scalar_required);
        if (status != KFCORE_KALMAN_OK)
        {
            return status;
        }
    }

    return KFCORE_KALMAN_OK;
}

/* @} */
