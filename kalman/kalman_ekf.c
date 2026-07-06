/** @file kalman_ekf.c
 * KFCore
 * @author Jan Zwiener (jan@zwiener.org)
 *
 * @brief Extended Kalman Filter helpers
 * @{ */

/******************************************************************************
 * SYSTEM INCLUDE FILES
 ******************************************************************************/

#include <assert.h>
#include <string.h>

/******************************************************************************
 * PROJECT INCLUDE FILES
 ******************************************************************************/

#include "kalman_ekf.h"
#include "kalman_takasu.h"
#include "kalman_udu.h"
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

/******************************************************************************
 * FUNCTION BODIES
 ******************************************************************************/

int kalman_ekf_takasu_predict(float* x, float* P, kalman_ekf_transition_fn transition,
                              const float* G, const float* Q, int n, int r, void* user)
{
    float x_pred[KALMAN_MAX_STATE_SIZE];
    float Phi[KALMAN_MAX_STATE_SIZE * KALMAN_MAX_STATE_SIZE];

    assert(n > 0 && n <= KALMAN_MAX_STATE_SIZE);
    assert(r >= 0 && r <= KALMAN_MAX_STATE_SIZE);

    if (!x || !P || !transition)
    {
        return -1;
    }

    if (transition(x_pred, Phi, x, n, user) != 0)
    {
        return -1;
    }

    kalman_predict(NULL, P, Phi, G, Q, n, r);
    memcpy(x, x_pred, sizeof(x[0]) * n);

    return 0;
}

int kalman_ekf_takasu_update(float* x, float* P, const float* z, const float* R,
                             kalman_ekf_measurement_fn measurement, int n, int m,
                             float chi2_threshold, float* chi2, void* user)
{
    float z_pred[KALMAN_MAX_MEASUREMENTS];
    float dz[KALMAN_MAX_MEASUREMENTS];
    float Ht[KALMAN_MAX_STATE_SIZE * KALMAN_MAX_MEASUREMENTS];

    assert(n > 0 && n <= KALMAN_MAX_STATE_SIZE);
    assert(m > 0 && m <= KALMAN_MAX_MEASUREMENTS);

    if (!x || !P || !z || !R || !measurement)
    {
        return -1;
    }

    if (measurement(z_pred, Ht, x, n, m, user) != 0)
    {
        return -1;
    }

    for (int i = 0; i < m; ++i)
    {
        dz[i] = z[i] - z_pred[i];
    }

    return kalman_takasu(x, P, dz, R, Ht, n, m, chi2_threshold, chi2);
}

int kalman_ekf_udu_predict(float* x, float* U, float* d, kalman_ekf_transition_fn transition,
                           const float* G, const float* Q, int n, int r, void* user)
{
    float x_pred[KALMAN_MAX_STATE_SIZE];
    float Phi[KALMAN_MAX_STATE_SIZE * KALMAN_MAX_STATE_SIZE];

    assert(n > 0 && n <= KALMAN_MAX_STATE_SIZE);
    assert(r >= 0 && r <= KALMAN_MAX_STATE_SIZE);

    if (!x || !U || !d || !transition)
    {
        return -1;
    }

    if (transition(x_pred, Phi, x, n, user) != 0)
    {
        return -1;
    }

    kalman_udu_predict(NULL, U, d, Phi, G, Q, n, r);
    memcpy(x, x_pred, sizeof(x[0]) * n);

    return 0;
}

int kalman_ekf_udu_update(float* x, float* U, float* d, const float* z, const float* R,
                          kalman_ekf_measurement_fn measurement, int n, int m, float chi2_threshold,
                          int downweight_outlier, void* user)
{
    int   retcode = 0;
    float z_pred[KALMAN_MAX_MEASUREMENTS];
    float dz[KALMAN_MAX_MEASUREMENTS];
    float Ht[KALMAN_MAX_STATE_SIZE * KALMAN_MAX_MEASUREMENTS];
    float R_work[KALMAN_MAX_MEASUREMENTS * KALMAN_MAX_MEASUREMENTS];
    float Reye[KALMAN_MAX_MEASUREMENTS * KALMAN_MAX_MEASUREMENTS];

    assert(n > 0 && n <= KALMAN_MAX_STATE_SIZE);
    assert(m > 0 && m <= KALMAN_MAX_MEASUREMENTS);

    if (!x || !U || !d || !z || !R || !measurement)
    {
        return -1;
    }

    if (measurement(z_pred, Ht, x, n, m, user) != 0)
    {
        return -1;
    }

    for (int i = 0; i < m; ++i)
    {
        dz[i] = z[i] - z_pred[i];
    }

    memcpy(R_work, R, sizeof(R_work[0]) * m * m);
    if (decorrelate(dz, Ht, R_work, n, m) != 0)
    {
        return -1;
    }
    mateye(Reye, m);

    for (int i = 0; i < m; ++i)
    {
        float        Rv     = MAT_ELEM(Reye, i, i, m, m);
        const float* H_line = Ht + i * n;

        if (chi2_threshold > 0.0f)
        {
            float tmp[KALMAN_MAX_STATE_SIZE];
            float HPHT = 0.0f;
            matmul("N", "N", 1, n, n, 1.0f, H_line, U, 0.0f, tmp);
            for (int j = 0; j < n; ++j)
            {
                HPHT += tmp[j] * tmp[j] * d[j];
            }

            const float s                   = HPHT + Rv;
            const float mahalanobis_dist_sq = dz[i] * dz[i] / s;
            if (mahalanobis_dist_sq > chi2_threshold)
            {
                if (!downweight_outlier)
                {
                    continue;
                }
                {
                    const float f = mahalanobis_dist_sq / chi2_threshold;
                    Rv            = (f - 1.0f) * HPHT + f * Rv;
                }
            }
        }

        if (kalman_udu_scalar(x, U, d, dz[i], Rv, H_line, n) != 0)
        {
            retcode = -1;
        }
    }

    return retcode;
}

/* @} */
