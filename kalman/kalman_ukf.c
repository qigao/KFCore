/** @file kalman_ukf.c
 * KFCore
 * @author Jan Zwiener (jan@zwiener.org)
 *
 * @brief Unscented Kalman Filter helpers
 * @{ */

#include <math.h>
#include <string.h>

#include "kalman_ukf.h"
#include "kalman_workspace_internal.h"
#include "linalg.h"

static int ukf_weights(int n, const kalman_ukf_params* params, float* wm0, float* wc0,
                       float* wi, float* gamma)
{
    const float alpha = params ? params->alpha : 1.0f;
    const float beta  = params ? params->beta : 2.0f;
    const float kappa = params ? params->kappa : 0.0f;
    const float nf    = (float)n;
    const float c     = alpha * alpha * (nf + kappa);

    if (!isfinite(alpha) || !isfinite(beta) || !isfinite(kappa) ||
        alpha <= 0.0f || c <= 0.0f || !isfinite(c))
    {
        return -1;
    }

    const float lambda = c - nf;
    *wm0   = lambda / c;
    *wc0   = *wm0 + (1.0f - alpha * alpha + beta);
    *wi    = 0.5f / c;
    *gamma = SQRTF(c);
    return 0;
}

static int ukf_scaled_cholesky(float* S, const float* P, int n, float gamma)
{
    for (int col = 0; col < n; ++col)
    {
        for (int row = 0; row < n; ++row)
        {
            MAT_ELEM(S, row, col, n, n) =
                (row >= col) ? MAT_ELEM(P, col, row, n, n) : 0.0f;
        }
    }

    if (cholesky(S, n, 1) != 0)
    {
        return -1;
    }

    for (int col = 0; col < n; ++col)
    {
        for (int row = col; row < n; ++row)
        {
            MAT_ELEM(S, row, col, n, n) *= gamma;
        }
    }
    return 0;
}

static void symmetrize_from_upper(float* P, int n)
{
    for (int col = 0; col < n; ++col)
    {
        for (int row = col + 1; row < n; ++row)
        {
            MAT_ELEM(P, row, col, n, n) = MAT_ELEM(P, col, row, n, n);
        }
    }
}

static kfcore_kalman_status ukf_sigma_count(size_t n, size_t* count)
{
    size_t twice_n;
    kfcore_kalman_status status = kfcore_kalman_checked_mul(n, 2U, &twice_n);
    if (status != KFCORE_KALMAN_OK)
    {
        return status;
    }
    return kfcore_kalman_checked_add(twice_n, 1U, count);
}

kfcore_kalman_status kalman_ukf_predict_workspace_floats(size_t n, size_t* required)
{
    size_t nn;
    size_t sigma_count;
    size_t propagated;
    size_t total;
    kfcore_kalman_status status;

    if (!required)
    {
        return KFCORE_KALMAN_INVALID_ARGUMENT;
    }
    status = kfcore_kalman_check_dim(n, 0);
    if (status != KFCORE_KALMAN_OK)
    {
        return status;
    }
    if ((status = kfcore_kalman_checked_mul(n, n, &nn)) != KFCORE_KALMAN_OK ||
        (status = ukf_sigma_count(n, &sigma_count)) != KFCORE_KALMAN_OK ||
        (status = kfcore_kalman_checked_mul(sigma_count, n, &propagated)) !=
            KFCORE_KALMAN_OK ||
        (status = kfcore_kalman_checked_add(nn, n, &total)) != KFCORE_KALMAN_OK ||
        (status = kfcore_kalman_checked_add(total, propagated, required)) !=
            KFCORE_KALMAN_OK)
    {
        return status;
    }
    return KFCORE_KALMAN_OK;
}

kfcore_kalman_status kalman_ukf_predict(
    float* x, float* P, const float* Q, kalman_ukf_transition_fn transition,
    size_t n, const kalman_ukf_params* params, void* user,
    float* workspace, size_t workspace_floats)
{
    size_t required;
    size_t nn;
    size_t sigma_count;
    float wm0;
    float wc0;
    float wi;
    float gamma;
    kfcore_kalman_status status = kalman_ukf_predict_workspace_floats(n, &required);

    if (status != KFCORE_KALMAN_OK)
    {
        return status;
    }
    if (!x || !P || !transition)
    {
        return KFCORE_KALMAN_INVALID_ARGUMENT;
    }
    if (ukf_weights((int)n, params, &wm0, &wc0, &wi, &gamma) != 0)
    {
        return KFCORE_KALMAN_INVALID_ARGUMENT;
    }
    status = kfcore_kalman_require_workspace(workspace, workspace_floats, required);
    if (status != KFCORE_KALMAN_OK)
    {
        return status;
    }

    (void)kfcore_kalman_checked_mul(n, n, &nn);
    (void)ukf_sigma_count(n, &sigma_count);

    float* Sx = workspace;
    float* sigma_point = Sx + nn;
    float* propagated = sigma_point + n;
    if (ukf_scaled_cholesky(Sx, P, (int)n, gamma) != 0)
    {
        return KFCORE_KALMAN_NUMERICAL_FAILURE;
    }

    if (transition(propagated, x, (int)n, user) != 0)
    {
        return KFCORE_KALMAN_CALLBACK_FAILURE;
    }
    for (size_t k = 0U; k < n; ++k)
    {
        for (size_t row = 0U; row < n; ++row)
        {
            sigma_point[row] = x[row] + MAT_ELEM(Sx, row, k, n, n);
        }
        if (transition(propagated + (1U + k) * n, sigma_point, (int)n, user) != 0)
        {
            return KFCORE_KALMAN_CALLBACK_FAILURE;
        }

        for (size_t row = 0U; row < n; ++row)
        {
            sigma_point[row] = x[row] - MAT_ELEM(Sx, row, k, n, n);
        }
        if (transition(propagated + (1U + n + k) * n, sigma_point, (int)n, user) != 0)
        {
            return KFCORE_KALMAN_CALLBACK_FAILURE;
        }
    }

    for (size_t row = 0U; row < n; ++row)
    {
        x[row] = wm0 * propagated[row];
    }
    for (size_t k = 1U; k < sigma_count; ++k)
    {
        for (size_t row = 0U; row < n; ++row)
        {
            x[row] += wi * propagated[k * n + row];
        }
    }

    if (Q)
    {
        memcpy(P, Q, sizeof(float) * nn);
    }
    else
    {
        memset(P, 0, sizeof(float) * nn);
    }

    for (size_t k = 0U; k < sigma_count; ++k)
    {
        const float weight = (k == 0U) ? wc0 : wi;
        const float* y = propagated + k * n;

        for (size_t col = 0U; col < n; ++col)
        {
            const float dcol = y[col] - x[col];
            for (size_t row = 0U; row <= col; ++row)
            {
                MAT_ELEM(P, row, col, n, n) +=
                    weight * (y[row] - x[row]) * dcol;
            }
        }
    }
    symmetrize_from_upper(P, (int)n);
    return KFCORE_KALMAN_OK;
}

kfcore_kalman_status kalman_ukf_update_workspace_floats(size_t n, size_t m,
                                                        size_t* required)
{
    size_t nn;
    size_t nm;
    size_t mm;
    size_t sigma_count;
    size_t z_sigma;
    size_t total;
    kfcore_kalman_status status;

    if (!required)
    {
        return KFCORE_KALMAN_INVALID_ARGUMENT;
    }
    if ((status = kfcore_kalman_check_dim(n, 0)) != KFCORE_KALMAN_OK ||
        (status = kfcore_kalman_check_dim(m, 0)) != KFCORE_KALMAN_OK)
    {
        return status;
    }
    if ((status = kfcore_kalman_checked_mul(n, n, &nn)) != KFCORE_KALMAN_OK ||
        (status = kfcore_kalman_checked_mul(n, m, &nm)) != KFCORE_KALMAN_OK ||
        (status = kfcore_kalman_checked_mul(m, m, &mm)) != KFCORE_KALMAN_OK ||
        (status = ukf_sigma_count(n, &sigma_count)) != KFCORE_KALMAN_OK ||
        (status = kfcore_kalman_checked_mul(sigma_count, m, &z_sigma)) !=
            KFCORE_KALMAN_OK ||
        (status = kfcore_kalman_checked_add(nn, n, &total)) != KFCORE_KALMAN_OK ||
        (status = kfcore_kalman_checked_add(total, z_sigma, &total)) !=
            KFCORE_KALMAN_OK ||
        (status = kfcore_kalman_checked_add(total, m, &total)) != KFCORE_KALMAN_OK ||
        (status = kfcore_kalman_checked_add(total, m, &total)) != KFCORE_KALMAN_OK ||
        (status = kfcore_kalman_checked_add(total, mm, &total)) != KFCORE_KALMAN_OK ||
        (status = kfcore_kalman_checked_add(total, nm, &total)) != KFCORE_KALMAN_OK ||
        (status = kfcore_kalman_checked_add(total, m, required)) != KFCORE_KALMAN_OK)
    {
        return status;
    }
    return KFCORE_KALMAN_OK;
}

kfcore_kalman_status kalman_ukf_update(
    float* x, float* P, const float* z, const float* R,
    kalman_ukf_measurement_fn measurement, size_t n, size_t m,
    const kalman_ukf_params* params, float chi2_threshold, float* chi2,
    void* user, float* workspace, size_t workspace_floats)
{
    size_t required;
    size_t nn;
    size_t nm;
    size_t mm;
    size_t sigma_count;
    size_t z_sigma_count;
    float wm0;
    float wc0;
    float wi;
    float gamma;
    kfcore_kalman_status status = kalman_ukf_update_workspace_floats(n, m, &required);

    if (status != KFCORE_KALMAN_OK)
    {
        return status;
    }
    if (!x || !P || !z || !R || !measurement)
    {
        return KFCORE_KALMAN_INVALID_ARGUMENT;
    }
    if (ukf_weights((int)n, params, &wm0, &wc0, &wi, &gamma) != 0)
    {
        return KFCORE_KALMAN_INVALID_ARGUMENT;
    }
    status = kfcore_kalman_require_workspace(workspace, workspace_floats, required);
    if (status != KFCORE_KALMAN_OK)
    {
        return status;
    }

    (void)kfcore_kalman_checked_mul(n, n, &nn);
    (void)kfcore_kalman_checked_mul(n, m, &nm);
    (void)kfcore_kalman_checked_mul(m, m, &mm);
    (void)ukf_sigma_count(n, &sigma_count);
    (void)kfcore_kalman_checked_mul(sigma_count, m, &z_sigma_count);

    float* Sx = workspace;
    float* sigma_point = Sx + nn;
    float* z_sigma = sigma_point + n;
    float* z_pred = z_sigma + z_sigma_count;
    float* dz = z_pred + m;
    float* S = dz + m;
    float* Pxz = S + mm;
    float* y = Pxz + nm;
    if (ukf_scaled_cholesky(Sx, P, (int)n, gamma) != 0)
    {
        return KFCORE_KALMAN_NUMERICAL_FAILURE;
    }

    if (measurement(z_sigma, x, (int)n, (int)m, user) != 0)
    {
        return KFCORE_KALMAN_CALLBACK_FAILURE;
    }
    for (size_t k = 0U; k < n; ++k)
    {
        for (size_t row = 0U; row < n; ++row)
        {
            sigma_point[row] = x[row] + MAT_ELEM(Sx, row, k, n, n);
        }
        if (measurement(z_sigma + (1U + k) * m, sigma_point, (int)n, (int)m, user) != 0)
        {
            return KFCORE_KALMAN_CALLBACK_FAILURE;
        }

        for (size_t row = 0U; row < n; ++row)
        {
            sigma_point[row] = x[row] - MAT_ELEM(Sx, row, k, n, n);
        }
        if (measurement(z_sigma + (1U + n + k) * m, sigma_point,
                        (int)n, (int)m, user) != 0)
        {
            return KFCORE_KALMAN_CALLBACK_FAILURE;
        }
    }

    for (size_t row = 0U; row < m; ++row)
    {
        z_pred[row] = wm0 * z_sigma[row];
    }
    for (size_t k = 1U; k < sigma_count; ++k)
    {
        for (size_t row = 0U; row < m; ++row)
        {
            z_pred[row] += wi * z_sigma[k * m + row];
        }
    }
    for (size_t row = 0U; row < m; ++row)
    {
        dz[row] = z[row] - z_pred[row];
    }

    memcpy(S, R, sizeof(float) * mm);
    memset(Pxz, 0, sizeof(float) * nm);

    for (size_t k = 0U; k < sigma_count; ++k)
    {
        const float weight = (k == 0U) ? wc0 : wi;
        const float* zs = z_sigma + k * m;

        for (size_t col = 0U; col < m; ++col)
        {
            const float dz_col = zs[col] - z_pred[col];
            for (size_t row = 0U; row <= col; ++row)
            {
                MAT_ELEM(S, row, col, m, m) +=
                    weight * (zs[row] - z_pred[row]) * dz_col;
            }

            if (k > 0U)
            {
                const size_t sigma_index = k - 1U;
                const size_t chol_col = sigma_index % n;
                const float sign = (sigma_index < n) ? 1.0f : -1.0f;

                for (size_t row = chol_col; row < n; ++row)
                {
                    MAT_ELEM(Pxz, row, col, n, m) +=
                        weight * sign * MAT_ELEM(Sx, row, chol_col, n, n) * dz_col;
                }
            }
        }
    }
    symmetrize_from_upper(S, (int)m);

    if (cholesky(S, (int)m, 1) != 0)
    {
        return KFCORE_KALMAN_NUMERICAL_FAILURE;
    }

    if (chi2 || chi2_threshold > 0.0f)
    {
        float chi2sum = 0.0f;
        memcpy(y, dz, sizeof(float) * m);
        trisolve(S, y, (int)m, 1, "N");
        for (size_t i = 0U; i < m; ++i)
        {
            chi2sum += y[i] * y[i];
        }
        chi2sum /= (float)m;

        if (chi2)
        {
            *chi2 = chi2sum;
        }
        if (chi2_threshold > 0.0f && chi2sum > chi2_threshold)
        {
            return KFCORE_KALMAN_REJECTED;
        }
    }

    trisolveright(S, Pxz, (int)m, (int)n, "T");
    symmetricrankupdate(P, Pxz, (int)n, (int)m);
    trisolveright(S, Pxz, (int)m, (int)n, "N");
    matmul("N", "N", (int)n, 1, (int)m, 1.0f, Pxz, dz, 1.0f, x);
    symmetrize_from_upper(P, (int)n);

    return KFCORE_KALMAN_OK;
}

/* @} */
