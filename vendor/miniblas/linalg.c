/** @file linalg.c
 * Embedded linear algebra backend
 * @author Jan Zwiener (jan@zwiener.org)
 *
 * @brief Math functions
 * @{ */

/******************************************************************************
 * SYSTEM INCLUDE FILES
 ******************************************************************************/

#include <math.h>
#include <string.h> /* memset */

/******************************************************************************
 * PROJECT INCLUDE FILES
 ******************************************************************************/

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

int matmul(const char* ta, const char* tb, int n, int k, int m, float alpha, const float* A,
           const float* B, float beta, float* C)
{
    int lda = lsame_(ta, "T") ? m : n;
    int ldb = lsame_(tb, "T") ? k : m;
    return sgemm_((char*)ta, (char*)tb, &n, &k, &m, &alpha,
                  (float*)A, &lda, (float*)B, &ldb, &beta, C, &n);
}

int matmulsym(const float* A_sym, const float* B, int n, int m, float* C)
{
    float alpha = 1.0f;
    float beta = 0.0f;
    return ssymm_("L" /* calculate C = A*B not C = B*A */,
                  "U" /* reference upper triangular part of A */, &n, /* rows of B/C */
                  &m,                                                 /* cols of B / C */
                  &alpha, (float*)A_sym, &n, (float*)B, &n, &beta, C, &n);
}

int matvec(const char* trans, int rows, int cols, float alpha, const float* A, const float* x,
           float beta, float* y)
{
    int inc = 1;
    return sgemv_(trans, &rows, &cols, &alpha, A, &rows, x, &inc, &beta, y, &inc);
}

int rank1update(float* A, const float* x, const float* y, int rows, int cols, float alpha)
{
    int inc = 1;
    return sger_(&rows, &cols, &alpha, x, &inc, y, &inc, A, &rows);
}

void mateye(float* A, int n)
{
    memset(A, 0, sizeof(float) * n * n);
    for (int i = 0; i < n; i++)
    {
        MAT_ELEM(A, i, i, n, n) = 1.0f;
    }
}

int cholesky(float* A, const int n, int onlyWriteLowerPart)
{
    /* in-place calculation of lower triangular matrix L*L' = A */

    /* set the upper triangular part to zero? */
    if (!onlyWriteLowerPart)
    {
        for (int i = 0; i < n - 1; i++) /* row */
        {
            for (int j = i + 1; j < n; j++) /* col */
            {
                MAT_ELEM(A, i, j, n, n) = 0.0f;
            }
        }
    }

    for (int j = 0; j < n; j++) /* main loop */
    {
        const float Ajj = MAT_ELEM(A, j, j, n, n);
        if (Ajj <= 0.0f || !isfinite(Ajj))
        {
            return -1;
        }
        MAT_ELEM(A, j, j, n, n) = SQRTF(Ajj);

        const float invLjj = 1.0f / MAT_ELEM(A, j, j, n, n);
        for (int i = j + 1; i < n; i++)
        {
            MAT_ELEM(A, i, j, n, n) *= invLjj;
        }

        for (int k = j + 1; k < n; k++)
        {
            for (int i = k; i < n; i++)
            {
                MAT_ELEM(A, i, k, n, n) -= MAT_ELEM(A, i, j, n, n) * MAT_ELEM(A, k, j, n, n);
            }
        }
    }
    return 0;
}

float vecdot(const float* x, const float* y, int n)
{
    int inc = 1;

    return sdot_(&n, x, &inc, y, &inc);
}

float vecnorm(const float* x, int n)
{
    int inc = 1;

    return snrm2_(&n, x, &inc);
}

float vecmean(const float* x, int n)
{
    int   inc = 1;
    float mean;
    int   result = svec_mean_(&n, x, &inc, &mean);

    assert(result == 0);

    return mean;
}

float vecvariance(const float* x, int n, int ddof)
{
    int   inc = 1;
    float variance;
    int   result = svec_variance_(&n, x, &inc, &ddof, &variance);

    assert(result == 0);

    return variance;
}

float vecrms(const float* x, int n)
{
    int   inc = 1;
    float rms;
    int   result = svec_rms_(&n, x, &inc, &rms);

    assert(result == 0);

    return rms;
}

int vecnormalize(float* x, int n, float eps, float* norm)
{
    int   inc = 1;
    float norm_local;
    int   result = svec_normalize_(&n, x, &inc, &eps, &norm_local);

    if (norm)
    {
        *norm = norm_local;
    }

    return result;
}

float vecdist_l1(const float* x, const float* y, int n)
{
    int   inc = 1;
    float distance;
    int   result = svec_l1_distance_(&n, x, &inc, y, &inc, &distance);

    assert(result == 0);

    return distance;
}

float vecdist_linf(const float* x, const float* y, int n)
{
    int   inc = 1;
    float distance;
    int   result = svec_linf_distance_(&n, x, &inc, y, &inc, &distance);

    assert(result == 0);

    return distance;
}

int veccosine(const float* x, const float* y, int n, float* cosine)
{
    int inc = 1;

    return svec_cosine_similarity_(&n, x, &inc, y, &inc, cosine);
}

float mat2det(const float* A)
{
    return smat2_det_(A);
}

int mat2inv(const float* A, float* Ainv, float eps)
{
    return smat2_inv_(A, Ainv, &eps);
}

float mat3det(const float* A)
{
    return smat3_det_(A);
}

int mat3inv(const float* A, float* Ainv, float eps)
{
    return smat3_inv_(A, Ainv, &eps);
}

int trisolve(const float* A, float* B, int n, int m, const char* tp)
{
    float alpha = 1.0f;
    return strsm_("L" /* left hand*/, "L" /* lower triangular matrix */, tp /* transpose L? */,
                  "N" /* L is not unit triangular */, &n, &m, &alpha, A, &n, B, &n);
}

int trisolveright(const float* L, float* A, int n, int m, const char* tp)
{
    float alpha = 1.0f;
    return strsm_("R" /* right hand*/, "L" /* lower triangular matrix */, tp /* transpose L? */,
                  "N" /* L is not unit triangular */, &m, &n, &alpha, L, &n, A, &m);
}

int symmetricrankupdate(float* P, const float* E, int n, int m)
{
    float alpha = -1.0f;
    float beta = 1.0f;
    return ssyrk_("U", "N", &n, &m, &alpha, (float*)E, &n, &beta, P, &n);
}

int udu(const float* A, float* U, float* d, const int m)
{
    if (!A || !U || !d || m <= 0)
    {
        return -1;
    }

    /*    A = U*diag(d)*U' decomposition
     *    Source:
     *      1. Golub, Gene H., and Charles F. Van Loan. "Matrix Computations." 4rd ed.,
     *         Johns Hopkins University Press, 2013.
     *      2. Grewal, Weill, Andrews. "Global positioning systems, inertial
     *         navigation, and integration". 1st ed. John Wiley & Sons, New York, 2001.
     *
     *    function [U, d] = udu(M)
     *      [m, ~] = size(M);
     *      U = zeros(m, m); d = zeros(m, 1);
     *
     *      for j = m:-1:1
     *        for i = j:-1:1
     *          sigma = M(i, j);
     *          for k = j + 1:m
     *              sigma = sigma - U(i, k) * d(k) * U(j, k);
     *          end
     *          if i == j
     *              d(j) = sigma;
     *              U(j, j) = 1; % U is a unit triangular matrix
     *          else
     *              U(i, j) = sigma / d(j); % off-diagonal elements of U
     *          end
     *        end
     *      end
     *    end
     */
    int   i, j, k;
    float sigma;

    memset(U, 0, sizeof(U[0]) * m * m);
    memset(d, 0, sizeof(d[0]) * m);

    for (j = m - 1; j >= 0; j--) /* UDU decomposition */
    {
        for (i = j; i >= 0; i--)
        {
            sigma = MAT_ELEM(A, i, j, m, m);
            for (k = j + 1; k < m; k++)
            {
                sigma -= MAT_ELEM(U, i, k, m, m) * d[k] * MAT_ELEM(U, j, k, m, m);
            }
            if (i == j)
            {
                if (!(sigma > 0.0f) || !isfinite(sigma))
                {
                    return -1;
                }
                d[j]                    = sigma;
                MAT_ELEM(U, j, j, m, m) = 1.0f;
            }
            else
            {
                MAT_ELEM(U, i, j, m, m) = sigma / d[j];
            }
        }
    }
    return 0;
}
