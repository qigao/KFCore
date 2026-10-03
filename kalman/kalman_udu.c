/** @file kalman_udu.c
 * KFCore
 * @author Jan Zwiener (jan@zwiener.org)
 *
 * @brief UDU Kalman Filter
 * @{ */

/******************************************************************************
 * SYSTEM INCLUDE FILES
 ******************************************************************************/

#include <math.h>
#include <string.h> /* memcpy */

/******************************************************************************
 * PROJECT INCLUDE FILES
 ******************************************************************************/

#include "linalg.h"
#include "miniblas.h" /* strmm_ */
#include "kalman_udu.h"
#include "kalman_workspace_internal.h"

/******************************************************************************
 * DEFINES
 ******************************************************************************/


/******************************************************************************
 * TYPEDEFS
 ******************************************************************************/

/******************************************************************************
 * LOCAL DATA DEFINITIONS
 ******************************************************************************/

/******************************************************************************
 * LOCAL FUNCTION PROTOTYPES
 ******************************************************************************/

/******************************************************************************
 * FUNCTION BODIES
 ******************************************************************************/

kfcore_kalman_status kalman_udu_scalar_workspace_floats(size_t n, size_t* required)
{
    kfcore_kalman_status status = kfcore_kalman_check_dim(n, 0);
    if (status != KFCORE_KALMAN_OK || !required)
    {
        return status == KFCORE_KALMAN_OK ? KFCORE_KALMAN_INVALID_ARGUMENT : status;
    }
    return kfcore_kalman_checked_mul(n, 2U, required);
}

kfcore_kalman_status kalman_udu_scalar(float* x, float* U, float* d, float dz, float R,
                                       const float* H_line, size_t n,
                                       float* workspace, size_t workspace_floats)
{
    size_t required;
    kfcore_kalman_status status = kalman_udu_scalar_workspace_floats(n, &required);
    if (status != KFCORE_KALMAN_OK)
    {
        return status;
    }
    if (!x || !U || !d || !H_line || !(R > 0.0f) || !isfinite(R))
    {
        return KFCORE_KALMAN_INVALID_ARGUMENT;
    }
    status = kfcore_kalman_require_workspace(workspace, workspace_floats, required);
    if (status != KFCORE_KALMAN_OK)
    {
        return status;
    }

    float* a = workspace;
    float* b = workspace + n;
    float alpha = R;
    float gamma = 1.0f / alpha;
    int ni = (int)n;
    int one = 1;
    float onef = 1.0f;

    memcpy(a, H_line, sizeof(float) * n);
    if (strmm_("L", "U", "T", "U", &ni, &one, &onef, U, &ni, a, &ni) != 0)
        return KFCORE_KALMAN_NUMERICAL_FAILURE;

    for (size_t j = 0U; j < n; ++j)
    {
        b[j] = d[j] * a[j];
    }

    for (size_t j = 0U; j < n; ++j)
    {
        float beta = alpha;
        alpha += a[j] * b[j];
        if (!(alpha > 0.0f) || !isfinite(alpha))
        {
            return KFCORE_KALMAN_NUMERICAL_FAILURE;
        }

        const float lambda = -a[j] * gamma;
        gamma = 1.0f / alpha;
        d[j] *= beta * gamma;

        for (size_t i = 0U; i < j; ++i)
        {
            beta = MAT_ELEM(U, i, j, n, n);
            MAT_ELEM(U, i, j, n, n) = beta + b[i] * lambda;
            b[i] += b[j] * beta;
        }
    }

    for (size_t j = 0U; j < n; ++j)
    {
        x[j] += gamma * dz * b[j];
    }
    return KFCORE_KALMAN_OK;
}

kfcore_kalman_status kalman_udu_workspace_floats(size_t n, size_t* required)
{
    return kalman_udu_scalar_workspace_floats(n, required);
}

kfcore_kalman_status kalman_udu(float* x, float* U, float* d, const float* z, const float* R,
                                const float* Ht, size_t n, size_t m,
                                float chi2_threshold, int downweight_outlier,
                                float* workspace, size_t workspace_floats)
{
    size_t required;
    kfcore_kalman_status status = kalman_udu_workspace_floats(n, &required);
    if (status != KFCORE_KALMAN_OK)
    {
        return status;
    }
    if (kfcore_kalman_check_dim(m, 0) != KFCORE_KALMAN_OK ||
        !x || !U || !d || !z || !R || !Ht)
    {
        return KFCORE_KALMAN_INVALID_ARGUMENT;
    }
    status = kfcore_kalman_require_workspace(workspace, workspace_floats, required);
    if (status != KFCORE_KALMAN_OK)
    {
        return status;
    }

    for (size_t i = 0U; i < m; ++i)
    {
        const float* h = Ht + i * n;
        float Rv = MAT_ELEM(R, i, i, m, m);
        float dz = z[i];

        if (matmul("N", "N", 1, 1, (int)n, -1.0f, h, x, 1.0f, &dz) != 0)
            return KFCORE_KALMAN_NUMERICAL_FAILURE;

        if (chi2_threshold > 0.0f)
        {
            float HPHT = 0.0f;
            if (matmul("N", "N", 1, (int)n, (int)n, 1.0f, h, U, 0.0f, workspace) != 0)
                return KFCORE_KALMAN_NUMERICAL_FAILURE;
            for (size_t j = 0U; j < n; ++j)
            {
                HPHT += workspace[j] * workspace[j] * d[j];
            }

            const float innovation_variance = HPHT + Rv;
            if (!(innovation_variance > 0.0f) || !isfinite(innovation_variance))
            {
                return KFCORE_KALMAN_NUMERICAL_FAILURE;
            }

            const float mahalanobis_distance_sq = dz * dz / innovation_variance;
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

        status = kalman_udu_scalar(x, U, d, dz, Rv, h, n, workspace, workspace_floats);
        if (status != KFCORE_KALMAN_OK)
        {
            return status;
        }
    }

    return KFCORE_KALMAN_OK;
}

kfcore_kalman_status decorrelate(float* z, float* Ht, float* R, size_t n, size_t m)
{
    if (kfcore_kalman_check_dim(n, 0) != KFCORE_KALMAN_OK ||
        kfcore_kalman_check_dim(m, 0) != KFCORE_KALMAN_OK ||
        !z || !Ht || !R)
    {
        return KFCORE_KALMAN_INVALID_ARGUMENT;
    }
    if (cholesky(R, (int)m, 0) != 0)
    {
        return KFCORE_KALMAN_NUMERICAL_FAILURE;
    }

    if (trisolveright(R, Ht, (int)m, (int)n, "T") != 0 ||
        trisolve(R, z, (int)m, 1, "N") != 0)
        return KFCORE_KALMAN_NUMERICAL_FAILURE;
    return KFCORE_KALMAN_OK;
}

kfcore_kalman_status kalman_udu_predict_workspace_floats(size_t n, size_t r, size_t* required)
{
    size_t nn;
    size_t nr;
    size_t total;
    kfcore_kalman_status status;

    if ((status = kfcore_kalman_check_dim(n, 0)) != KFCORE_KALMAN_OK ||
        (status = kfcore_kalman_check_dim(r, 1)) != KFCORE_KALMAN_OK ||
        !required)
    {
        return status == KFCORE_KALMAN_OK ? KFCORE_KALMAN_INVALID_ARGUMENT : status;
    }
    if ((status = kfcore_kalman_checked_mul(n, n, &nn)) != KFCORE_KALMAN_OK ||
        (status = kfcore_kalman_checked_mul(n, r, &nr)) != KFCORE_KALMAN_OK ||
        (status = kfcore_kalman_checked_add(nn, nr, &total)) != KFCORE_KALMAN_OK ||
        (status = kfcore_kalman_checked_add(total, n, &total)) != KFCORE_KALMAN_OK ||
        (status = kfcore_kalman_checked_add(total, n, required)) != KFCORE_KALMAN_OK)
    {
        return status;
    }
    return KFCORE_KALMAN_OK;
}

kfcore_kalman_status kalman_udu_predict(float* x, float* U, float* d, const float* Phi,
                                        const float* G, const float* Q, size_t n, size_t r,
                                        float* workspace, size_t workspace_floats)
{
    size_t required;
    size_t nn;
    size_t nr;
    kfcore_kalman_status status = kalman_udu_predict_workspace_floats(n, r, &required);
    if (status != KFCORE_KALMAN_OK)
    {
        return status;
    }
    if (!U || !d || !Phi || (r > 0U && (!G || !Q)))
    {
        return KFCORE_KALMAN_INVALID_ARGUMENT;
    }
    status = kfcore_kalman_require_workspace(workspace, workspace_floats, required);
    if (status != KFCORE_KALMAN_OK)
    {
        return status;
    }

    (void)kfcore_kalman_checked_mul(n, n, &nn);
    (void)kfcore_kalman_checked_mul(n, r, &nr);

    float* tmp = workspace;
    float* G_tmp = tmp + n;
    float* PhiU = G_tmp + nr;
    float* din = PhiU + nn;

    if (x)
    {
        memcpy(tmp, x, sizeof(float) * n);
        if (matmul("N", "N", (int)n, 1, (int)n, 1.0f, Phi, tmp, 0.0f, x) != 0)
            return KFCORE_KALMAN_NUMERICAL_FAILURE;
    }
    if (r > 0U)
    {
        memcpy(G_tmp, G, sizeof(float) * nr);
    }
    memcpy(PhiU, Phi, sizeof(float) * nn);

    int ni = (int)n;
    float one = 1.0f;
    if (strmm_("R", "U", "N", "U", &ni, &ni, &one, U, &ni, PhiU, &ni) != 0)
        return KFCORE_KALMAN_NUMERICAL_FAILURE;

    mateye(U, (int)n);
    memcpy(din, d, sizeof(float) * n);

    for (size_t i = n; i-- > 0U;)
    {
        float sigma = 0.0f;
        for (size_t j = 0U; j < n; ++j)
        {
            sigma += MAT_ELEM(PhiU, i, j, n, n) *
                     MAT_ELEM(PhiU, i, j, n, n) * din[j];
            if (j < r)
            {
                sigma += MAT_ELEM(G_tmp, i, j, n, r) *
                         MAT_ELEM(G_tmp, i, j, n, r) * Q[j];
            }
        }

        if (!(sigma > 0.0f) || !isfinite(sigma))
        {
            return KFCORE_KALMAN_NUMERICAL_FAILURE;
        }
        d[i] = sigma;

        for (size_t j = 0U; j < i; ++j)
        {
            sigma = 0.0f;
            for (size_t k = 0U; k < n; ++k)
            {
                sigma += MAT_ELEM(PhiU, i, k, n, n) *
                         din[k] * MAT_ELEM(PhiU, j, k, n, n);
            }
            for (size_t k = 0U; k < r; ++k)
            {
                sigma += MAT_ELEM(G_tmp, i, k, n, r) *
                         Q[k] * MAT_ELEM(G_tmp, j, k, n, r);
            }

            MAT_ELEM(U, j, i, n, n) = sigma / d[i];
            for (size_t k = 0U; k < n; ++k)
            {
                MAT_ELEM(PhiU, j, k, n, n) -=
                    MAT_ELEM(U, j, i, n, n) * MAT_ELEM(PhiU, i, k, n, n);
            }
            for (size_t k = 0U; k < r; ++k)
            {
                MAT_ELEM(G_tmp, j, k, n, r) -=
                    MAT_ELEM(U, j, i, n, n) * MAT_ELEM(G_tmp, i, k, n, r);
            }
        }
    }

    return KFCORE_KALMAN_OK;
}

/* @} */
