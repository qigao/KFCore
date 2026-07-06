/** @file kalman_ukf.c
 * KFCore
 * @author Jan Zwiener (jan@zwiener.org)
 *
 * @brief Unscented Kalman Filter helpers
 * @{ */

/******************************************************************************
 * SYSTEM INCLUDE FILES
 ******************************************************************************/

#include <assert.h>
#include <math.h>
#include <string.h>

/******************************************************************************
 * PROJECT INCLUDE FILES
 ******************************************************************************/

#include "kalman_ukf.h"
#include "linalg.h"

/******************************************************************************
 * DEFINES
 ******************************************************************************/

#ifndef KALMAN_MAX_STATE_SIZE
#define KALMAN_MAX_STATE_SIZE 32
#endif

#ifndef KALMAN_MAX_MEASUREMENTS
#define KALMAN_MAX_MEASUREMENTS 3
#endif

#define KALMAN_MAX_SIGMA_POINTS (2 * KALMAN_MAX_STATE_SIZE + 1)

/******************************************************************************
 * LOCAL FUNCTION PROTOTYPES
 ******************************************************************************/

static int  ukf_weights(int n, const kalman_ukf_params* params, float* wm0, float* wc0, float* wi,
                        float* gamma);
static int  ukf_scaled_cholesky(float* S, const float* P, int n, float gamma);
static void symmetrize_from_upper(float* P, int n);

/******************************************************************************
 * FUNCTION BODIES
 ******************************************************************************/

static int ukf_weights(int n, const kalman_ukf_params* params, float* wm0, float* wc0, float* wi,
                       float* gamma)
{
    const float alpha = params ? params->alpha : 1.0f;
    const float beta  = params ? params->beta : 2.0f;
    const float kappa = params ? params->kappa : 0.0f;
    const float nf    = (float)n;
    const float c     = alpha * alpha * (nf + kappa);

    if (alpha <= 0.0f || c <= 0.0f || !isfinite(c))
    {
        return -1;
    }

    {
        const float lambda = c - nf;
        *wm0               = lambda / c;
        *wc0               = *wm0 + (1.0f - alpha * alpha + beta);
        *wi                = 0.5f / c;
        *gamma             = SQRTF(c);
    }

    return 0;
}

static int ukf_scaled_cholesky(float* S, const float* P, int n, float gamma)
{
    for (int col = 0; col < n; ++col)
    {
        for (int row = 0; row < n; ++row)
        {
            MAT_ELEM(S, row, col, n, n) = (row >= col) ? MAT_ELEM(P, col, row, n, n) : 0.0f;
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

int kalman_ukf_predict(float* x, float* P, const float* Q, kalman_ukf_transition_fn transition,
                       int n, const kalman_ukf_params* params, void* user)
{
    float     Sx[KALMAN_MAX_STATE_SIZE * KALMAN_MAX_STATE_SIZE];
    float     sigma_point[KALMAN_MAX_STATE_SIZE];
    float     propagated[KALMAN_MAX_SIGMA_POINTS * KALMAN_MAX_STATE_SIZE];
    float     wm0;
    float     wc0;
    float     wi;
    float     gamma;
    const int sigma_count = 2 * n + 1;

    assert(n > 0 && n <= KALMAN_MAX_STATE_SIZE);

    if (!x || !P || !transition)
    {
        return -1;
    }
    if (ukf_weights(n, params, &wm0, &wc0, &wi, &gamma) != 0)
    {
        return -1;
    }
    if (ukf_scaled_cholesky(Sx, P, n, gamma) != 0)
    {
        return -1;
    }

    if (transition(propagated, x, n, user) != 0)
    {
        return -1;
    }
    for (int k = 0; k < n; ++k)
    {
        for (int row = 0; row < n; ++row)
        {
            sigma_point[row] = x[row] + MAT_ELEM(Sx, row, k, n, n);
        }
        if (transition(propagated + (1 + k) * n, sigma_point, n, user) != 0)
        {
            return -1;
        }

        for (int row = 0; row < n; ++row)
        {
            sigma_point[row] = x[row] - MAT_ELEM(Sx, row, k, n, n);
        }
        if (transition(propagated + (1 + n + k) * n, sigma_point, n, user) != 0)
        {
            return -1;
        }
    }

    for (int row = 0; row < n; ++row)
    {
        x[row] = wm0 * propagated[row];
    }
    for (int k = 1; k < sigma_count; ++k)
    {
        for (int row = 0; row < n; ++row)
        {
            x[row] += wi * propagated[k * n + row];
        }
    }

    if (Q)
    {
        memcpy(P, Q, sizeof(P[0]) * n * n);
    }
    else
    {
        memset(P, 0, sizeof(P[0]) * n * n);
    }

    for (int k = 0; k < sigma_count; ++k)
    {
        const float  weight = (k == 0) ? wc0 : wi;
        const float* y      = propagated + k * n;

        for (int col = 0; col < n; ++col)
        {
            const float dcol = y[col] - x[col];
            for (int row = 0; row <= col; ++row)
            {
                MAT_ELEM(P, row, col, n, n) += weight * (y[row] - x[row]) * dcol;
            }
        }
    }
    symmetrize_from_upper(P, n);

    return 0;
}

int kalman_ukf_update(float* x, float* P, const float* z, const float* R,
                      kalman_ukf_measurement_fn measurement, int n, int m,
                      const kalman_ukf_params* params, float chi2_threshold, float* chi2,
                      void* user)
{
    float     Sx[KALMAN_MAX_STATE_SIZE * KALMAN_MAX_STATE_SIZE];
    float     sigma_point[KALMAN_MAX_STATE_SIZE];
    float     z_sigma[KALMAN_MAX_SIGMA_POINTS * KALMAN_MAX_MEASUREMENTS];
    float     z_pred[KALMAN_MAX_MEASUREMENTS];
    float     dz[KALMAN_MAX_MEASUREMENTS];
    float     S[KALMAN_MAX_MEASUREMENTS * KALMAN_MAX_MEASUREMENTS];
    float     Pxz[KALMAN_MAX_STATE_SIZE * KALMAN_MAX_MEASUREMENTS];
    float     wm0;
    float     wc0;
    float     wi;
    float     gamma;
    const int sigma_count = 2 * n + 1;

    assert(n > 0 && n <= KALMAN_MAX_STATE_SIZE);
    assert(m > 0 && m <= KALMAN_MAX_MEASUREMENTS);

    if (!x || !P || !z || !R || !measurement)
    {
        return -1;
    }
    if (ukf_weights(n, params, &wm0, &wc0, &wi, &gamma) != 0)
    {
        return -1;
    }
    if (ukf_scaled_cholesky(Sx, P, n, gamma) != 0)
    {
        return -1;
    }

    if (measurement(z_sigma, x, n, m, user) != 0)
    {
        return -1;
    }
    for (int k = 0; k < n; ++k)
    {
        for (int row = 0; row < n; ++row)
        {
            sigma_point[row] = x[row] + MAT_ELEM(Sx, row, k, n, n);
        }
        if (measurement(z_sigma + (1 + k) * m, sigma_point, n, m, user) != 0)
        {
            return -1;
        }

        for (int row = 0; row < n; ++row)
        {
            sigma_point[row] = x[row] - MAT_ELEM(Sx, row, k, n, n);
        }
        if (measurement(z_sigma + (1 + n + k) * m, sigma_point, n, m, user) != 0)
        {
            return -1;
        }
    }

    for (int row = 0; row < m; ++row)
    {
        z_pred[row] = wm0 * z_sigma[row];
    }
    for (int k = 1; k < sigma_count; ++k)
    {
        for (int row = 0; row < m; ++row)
        {
            z_pred[row] += wi * z_sigma[k * m + row];
        }
    }
    for (int row = 0; row < m; ++row)
    {
        dz[row] = z[row] - z_pred[row];
    }

    memcpy(S, R, sizeof(S[0]) * m * m);
    memset(Pxz, 0, sizeof(Pxz[0]) * n * m);

    for (int k = 0; k < sigma_count; ++k)
    {
        const float  weight = (k == 0) ? wc0 : wi;
        const float* zs     = z_sigma + k * m;

        for (int col = 0; col < m; ++col)
        {
            const float dz_col = zs[col] - z_pred[col];
            for (int row = 0; row <= col; ++row)
            {
                MAT_ELEM(S, row, col, m, m) += weight * (zs[row] - z_pred[row]) * dz_col;
            }
            if (k > 0)
            {
                const int   sigma_index = k - 1;
                const int   chol_col    = sigma_index % n;
                const float sign        = (sigma_index < n) ? 1.0f : -1.0f;

                for (int row = chol_col; row < n; ++row)
                {
                    MAT_ELEM(Pxz, row, col, n, m) +=
                        weight * sign * MAT_ELEM(Sx, row, chol_col, n, n) * dz_col;
                }
            }
        }
    }
    symmetrize_from_upper(S, m);

    if (cholesky(S, m, 1) != 0)
    {
        return -1;
    }

    if (chi2 || (chi2_threshold > 0.0f))
    {
        float y[KALMAN_MAX_MEASUREMENTS];
        float chi2sum = 0.0f;

        memcpy(y, dz, sizeof(y[0]) * m);
        trisolve(S, y, m, 1, "N");
        for (int i = 0; i < m; ++i)
        {
            chi2sum += y[i] * y[i];
        }
        chi2sum /= (float)m;

        if (chi2)
        {
            *chi2 = chi2sum;
        }
        if ((chi2_threshold > 0.0f) && (chi2sum > chi2_threshold))
        {
            return -2;
        }
    }

    trisolveright(S, Pxz, m, n, "T");
    symmetricrankupdate(P, Pxz, n, m);
    trisolveright(S, Pxz, m, n, "N");
    matmul("N", "N", n, 1, m, 1.0f, Pxz, dz, 1.0f, x);
    symmetrize_from_upper(P, n);

    return 0;
}

/* @} */
