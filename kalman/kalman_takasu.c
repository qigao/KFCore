/** @file kalman_takasu.c
 * KFCore
 * @author Jan Zwiener (jan@zwiener.org)
 *
 * @brief Kalman Filter Implementation (Takasu Formulation)
 * @{ */

/******************************************************************************
 * SYSTEM INCLUDE FILES
 ******************************************************************************/

#include <math.h>
#include <assert.h>
#include <string.h> /* memcpy */

/******************************************************************************
 * PROJECT INCLUDE FILES
 ******************************************************************************/

#include "kalman_takasu.h"
#include "kalman_workspace_internal.h"
#include "linalg.h"
#include "miniblas.h"

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

kfcore_kalman_status kalman_takasu_workspace_floats(size_t n, size_t m, size_t* required)
{
    size_t nm, mm, total;
    kfcore_kalman_status status;
    if ((status = kfcore_kalman_check_dim(n, 0)) != KFCORE_KALMAN_OK ||
        (status = kfcore_kalman_check_dim(m, 0)) != KFCORE_KALMAN_OK || !required)
        return status == KFCORE_KALMAN_OK ? KFCORE_KALMAN_INVALID_ARGUMENT : status;
    if ((status = kfcore_kalman_checked_mul(n, m, &nm)) != KFCORE_KALMAN_OK ||
        (status = kfcore_kalman_checked_mul(m, m, &mm)) != KFCORE_KALMAN_OK ||
        (status = kfcore_kalman_checked_add(nm, mm, &total)) != KFCORE_KALMAN_OK ||
        (status = kfcore_kalman_checked_add(total, m, required)) != KFCORE_KALMAN_OK)
        return status;
    return KFCORE_KALMAN_OK;
}

kfcore_kalman_status kalman_takasu(float* x, float* P, const float* dz, const float* R,
                                   const float* Ht, size_t n, size_t m,
                                   float chi2_threshold, float* chi2,
                                   float* workspace, size_t workspace_floats)
{
    size_t required, nm, mm;
    kfcore_kalman_status status = kalman_takasu_workspace_floats(n, m, &required);
    if (status != KFCORE_KALMAN_OK) return status;
    if (!x || !P || !dz || !R || !Ht) return KFCORE_KALMAN_INVALID_ARGUMENT;
    if ((status = kfcore_kalman_require_workspace(workspace, workspace_floats, required)) != KFCORE_KALMAN_OK)
        return status;
    (void)kfcore_kalman_checked_mul(n, m, &nm);
    (void)kfcore_kalman_checked_mul(m, m, &mm);
    float* D = workspace;
    float* L = D + nm;
    float* y = L + mm;

    matmulsym(P, Ht, (int)n, (int)m, D);
    memcpy(L, R, sizeof(float) * mm);
    matmul("T", "N", (int)m, (int)m, (int)n, 1.0f, Ht, D, 1.0f, L);
    if (cholesky(L, (int)m, 1) != 0) return KFCORE_KALMAN_NUMERICAL_FAILURE;

    if (chi2 || chi2_threshold > 0.0f)
    {
        memcpy(y, dz, sizeof(float) * m);
        trisolve(L, y, (int)m, 1, "N");
        float chi2sum = 0.0f;
        for (size_t i = 0; i < m; ++i) chi2sum += y[i] * y[i];
        chi2sum /= (float)m;
        if (chi2) *chi2 = chi2sum;
        if (chi2_threshold > 0.0f && chi2sum > chi2_threshold) return KFCORE_KALMAN_REJECTED;
    }

    trisolveright(L, D, (int)m, (int)n, "T");
    symmetricrankupdate(P, D, (int)n, (int)m);
    trisolveright(L, D, (int)m, (int)n, "N");
    matmul("N", "N", (int)n, 1, (int)m, 1.0f, D, dz, 1.0f, x);
    for (size_t i = 0U; i < n; ++i)
    {
        if (!isfinite(x[i]))
        {
            return KFCORE_KALMAN_NUMERICAL_FAILURE;
        }
    }
    return KFCORE_KALMAN_OK;
}

kfcore_kalman_status kalman_predict_workspace_floats(size_t n, size_t r, size_t* required)
{
    size_t nn, nr, total;
    kfcore_kalman_status status;
    if ((status = kfcore_kalman_check_dim(n, 0)) != KFCORE_KALMAN_OK ||
        (status = kfcore_kalman_check_dim(r, 1)) != KFCORE_KALMAN_OK || !required)
        return status == KFCORE_KALMAN_OK ? KFCORE_KALMAN_INVALID_ARGUMENT : status;
    if ((status = kfcore_kalman_checked_mul(n, n, &nn)) != KFCORE_KALMAN_OK ||
        (status = kfcore_kalman_checked_mul(n, r, &nr)) != KFCORE_KALMAN_OK ||
        (status = kfcore_kalman_checked_add(n, nn, &total)) != KFCORE_KALMAN_OK ||
        (status = kfcore_kalman_checked_add(total, nr, required)) != KFCORE_KALMAN_OK)
        return status;
    return KFCORE_KALMAN_OK;
}

kfcore_kalman_status kalman_predict(float* x, float* P, const float* Phi, const float* G,
                                    const float* Q, size_t n, size_t r,
                                    float* workspace, size_t workspace_floats)
{
    size_t required, nn;
    kfcore_kalman_status status = kalman_predict_workspace_floats(n, r, &required);
    if (status != KFCORE_KALMAN_OK) return status;
    if ((!x && !P) || !Phi || (r > 0U && P && (!G || !Q))) return KFCORE_KALMAN_INVALID_ARGUMENT;
    if ((status = kfcore_kalman_require_workspace(workspace, workspace_floats, required)) != KFCORE_KALMAN_OK)
        return status;
    (void)kfcore_kalman_checked_mul(n, n, &nn);
    float* tmp = workspace;
    float* Phi_x_P = tmp + n;
    float* GQ = Phi_x_P + nn;

    if (x)
    {
        memcpy(tmp, x, sizeof(float) * n);
        matmul("N", "N", (int)n, 1, (int)n, 1.0f, Phi, tmp, 0.0f, x);
    }
    if (P)
    {
        int ni=(int)n;
        float alpha=1.0f, beta=0.0f;
        ssymm_("R","U",&ni,&ni,&alpha,P,&ni,(float*)Phi,&ni,&beta,Phi_x_P,&ni);
        if (r > 0U)
        {
            for (size_t j=0;j<r;++j)
                for (size_t i=0;i<n;++i)
                    MAT_ELEM(GQ,i,j,n,r)=Q[j]*MAT_ELEM(G,i,j,n,r);
            matmul("N","T",(int)n,(int)n,(int)r,1.0f,GQ,G,0.0f,P);
            matmul("N","T",(int)n,(int)n,(int)n,1.0f,Phi_x_P,Phi,1.0f,P);
        }
        else
            matmul("N","T",(int)n,(int)n,(int)n,1.0f,Phi_x_P,Phi,0.0f,P);
    }
    return KFCORE_KALMAN_OK;
}

/* @} */
