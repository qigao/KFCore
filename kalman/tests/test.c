/** @file test.c
 * KFCore
 * @author Jan Zwiener (jan@zwiener.org)
 *
 * @brief Unit Test File
 * @{ */

/******************************************************************************
 * SYSTEM INCLUDE FILES
 ******************************************************************************/

#include <stdio.h>
#include <string.h>
#include <math.h>

/******************************************************************************
 * PROJECT INCLUDE FILES
 ******************************************************************************/

#include "tinytest.h"
#include "linalg.h"
#include "miniblas.h"
#include "frame_transform.h"
#include "navtoolbox.h"
#include "nav_fusion2d.h"
#include "nav_fusion3d.h"
#include "kalman_takasu.h"
#include "kalman_udu.h"
#include "kalman_ekf.h"
#include "kalman_ukf.h"
#include "signal_filters.h"
#ifdef APRILTAG_HAVE_MINIBLAS
#include "apriltag_pose_miniblas.h"
#endif

/******************************************************************************
 * DEFINES
 ******************************************************************************/

#define TEST_FLOAT_WITHIN(delta, expected, actual, message)                                        \
    do                                                                                             \
    {                                                                                              \
        (void)(message);                                                                           \
        check_within((actual), (expected), (delta));                                               \
    } while (0)

#define TEST_KALMAN_WORKSPACE_FLOATS 4096U

static int test_kalman_takasu(float* x, float* P, const float* dz, const float* R,
                              const float* Ht, int n, int m, float chi2_threshold, float* chi2)
{
    float workspace[TEST_KALMAN_WORKSPACE_FLOATS];
    return (int)(kalman_takasu)(x, P, dz, R, Ht, (size_t)n, (size_t)m,
                               chi2_threshold, chi2, workspace,
                               TEST_KALMAN_WORKSPACE_FLOATS);
}

static void test_kalman_predict(float* x, float* P, const float* Phi, const float* G,
                                const float* Q, int n, int r)
{
    float workspace[TEST_KALMAN_WORKSPACE_FLOATS];
    (void)(kalman_predict)(x, P, Phi, G, Q, (size_t)n, (size_t)r,
                           workspace, TEST_KALMAN_WORKSPACE_FLOATS);
}

static int test_kalman_udu(float* x, float* U, float* d, const float* z, const float* R,
                           const float* Ht, int n, int m, float chi2_threshold,
                           int downweight_outlier)
{
    float workspace[TEST_KALMAN_WORKSPACE_FLOATS];
    return (int)(kalman_udu)(x, U, d, z, R, Ht, (size_t)n, (size_t)m,
                            chi2_threshold, downweight_outlier, workspace,
                            TEST_KALMAN_WORKSPACE_FLOATS);
}

static int test_kalman_udu_scalar(float* x, float* U, float* d, float dz, float R,
                                  const float* H_line, int n)
{
    float workspace[TEST_KALMAN_WORKSPACE_FLOATS];
    return (int)(kalman_udu_scalar)(x, U, d, dz, R, H_line, (size_t)n,
                                   workspace, TEST_KALMAN_WORKSPACE_FLOATS);
}

static void test_kalman_udu_predict(float* x, float* U, float* d, const float* Phi,
                                    const float* G, const float* Q, int n, int r)
{
    float workspace[TEST_KALMAN_WORKSPACE_FLOATS];
    (void)(kalman_udu_predict)(x, U, d, Phi, G, Q, (size_t)n, (size_t)r,
                               workspace, TEST_KALMAN_WORKSPACE_FLOATS);
}

static int test_kalman_ekf_takasu_predict(
    float* x, float* P, kalman_ekf_transition_fn transition,
    const float* G, const float* Q, int n, int r, void* user)
{
    float workspace[TEST_KALMAN_WORKSPACE_FLOATS];
    return (int)(kalman_ekf_takasu_predict)(
        x, P, transition, G, Q, (size_t)n, (size_t)r, user,
        workspace, TEST_KALMAN_WORKSPACE_FLOATS);
}

static int test_kalman_ekf_takasu_update(
    float* x, float* P, const float* z, const float* R,
    kalman_ekf_measurement_fn measurement, int n, int m,
    float chi2_threshold, float* chi2, void* user)
{
    float workspace[TEST_KALMAN_WORKSPACE_FLOATS];
    return (int)(kalman_ekf_takasu_update)(
        x, P, z, R, measurement, (size_t)n, (size_t)m,
        chi2_threshold, chi2, user,
        workspace, TEST_KALMAN_WORKSPACE_FLOATS);
}

static int test_kalman_ekf_udu_predict(
    float* x, float* U, float* d, kalman_ekf_transition_fn transition,
    const float* G, const float* Q, int n, int r, void* user)
{
    float workspace[TEST_KALMAN_WORKSPACE_FLOATS];
    return (int)(kalman_ekf_udu_predict)(
        x, U, d, transition, G, Q, (size_t)n, (size_t)r, user,
        workspace, TEST_KALMAN_WORKSPACE_FLOATS);
}

static int test_kalman_ekf_udu_update(
    float* x, float* U, float* d, const float* z, const float* R,
    kalman_ekf_measurement_fn measurement, int n, int m,
    float chi2_threshold, int downweight_outlier, void* user)
{
    float workspace[TEST_KALMAN_WORKSPACE_FLOATS];
    return (int)(kalman_ekf_udu_update)(
        x, U, d, z, R, measurement, (size_t)n, (size_t)m,
        chi2_threshold, downweight_outlier, user,
        workspace, TEST_KALMAN_WORKSPACE_FLOATS);
}

#define kalman_ekf_takasu_predict(...) test_kalman_ekf_takasu_predict(__VA_ARGS__)
#define kalman_ekf_takasu_update(...) test_kalman_ekf_takasu_update(__VA_ARGS__)
#define kalman_ekf_udu_predict(...) test_kalman_ekf_udu_predict(__VA_ARGS__)
#define kalman_ekf_udu_update(...) test_kalman_ekf_udu_update(__VA_ARGS__)

#define kalman_takasu(...) test_kalman_takasu(__VA_ARGS__)
#define kalman_predict(...) test_kalman_predict(__VA_ARGS__)
#define kalman_udu(...) test_kalman_udu(__VA_ARGS__)
#define kalman_udu_scalar(...) test_kalman_udu_scalar(__VA_ARGS__)
#define kalman_udu_predict(...) test_kalman_udu_predict(__VA_ARGS__)

/******************************************************************************
 * TYPEDEFS
 ******************************************************************************/

/******************************************************************************
 * LOCAL DATA DEFINITIONS
 ******************************************************************************/

static volatile float benchmark_sink;

/******************************************************************************
 * LOCAL FUNCTION PROTOTYPES
 ******************************************************************************/

/** @brief Fill array with a Hilbert matrix.
 * @param[out] H Output Hilbert matrix (n x n).
 * @param[in] n Dimension of H. */
static void hilbert(float* H, int n);

/** @brief Print a matrix to stdout
 * @param[in] R column-major n x m matrix
 * @param[in] n rows
 * @param[in] m cols
 * @param[in] fmt printf format string, e.g. "%.3f"
 * @param[in] name Pretty print with the name of the matrix, can be NULL */
static void matprint(const float* R, const int n, const int m, const char* fmt, const char* name);

static int ekf_test_transition(float* x_pred, float* Phi, const float* x, int n, void* user);
static int ekf_test_measurement(float* z_pred, float* Ht, const float* x, int n, int m, void* user);
static int ekf_benchmark_measurement(float* z_pred, float* Ht, const float* x, int n, int m,
                                     void* user);
static int ukf_square_transition(float* x_pred, const float* x, int n, void* user);
static int ukf_square_measurement(float* z_pred, const float* x, int n, int m, void* user);
static int ukf_benchmark_measurement(float* z_pred, const float* x, int n, int m, void* user);
static void testframetransform(void);
static void testsignalfilters(void);
#ifdef APRILTAG_HAVE_MINIBLAS
static void testapriltagadapter(void);
#endif
static void testekf(void);
static void testukf(void);
static void benchmark_core_routines(void);

/******************************************************************************
 * FUNCTION BODIES
 ******************************************************************************/

static void testlinalg(void)
{
    printf("Running linalg (linear algebra) tests...\n");
    /* Note: all matrices in column-major order */

    // Test BLAS1 vector routines
    {
        int         n    = 3;
        int         inc1 = 1;
        int         inc2 = 2;
        int         incn = -1;
        float       x[3] = { 1.0f, 2.0f, 3.0f };
        float       y[3] = { 0.0f, 0.0f, 0.0f };
        float       z[5] = { 1.0f, 99.0f, 2.0f, 99.0f, 3.0f };
        float       w[3] = { 0.0f, 0.0f, 0.0f };
        float       a    = 2.0f;
        const float threshold = 1.0e-06f;

        check_equal(scopy_(&n, x, &inc1, y, &inc1), 0);
        TEST_FLOAT_WITHIN(threshold, 1.0f, y[0], "scopy y[0] failed");
        TEST_FLOAT_WITHIN(threshold, 2.0f, y[1], "scopy y[1] failed");
        TEST_FLOAT_WITHIN(threshold, 3.0f, y[2], "scopy y[2] failed");

        check_equal(sscal_(&n, &a, y, &inc1), 0);
        TEST_FLOAT_WITHIN(threshold, 2.0f, y[0], "sscal y[0] failed");
        TEST_FLOAT_WITHIN(threshold, 4.0f, y[1], "sscal y[1] failed");
        TEST_FLOAT_WITHIN(threshold, 6.0f, y[2], "sscal y[2] failed");
        TEST_FLOAT_WITHIN(threshold, 28.0f, sdot_(&n, x, &inc1, y, &inc1), "sdot failed");
        TEST_FLOAT_WITHIN(threshold, 3.7416575f, snrm2_(&n, x, &inc1), "snrm2 failed");

        a = -0.5f;
        check_equal(saxpy_(&n, &a, y, &inc1, x, &inc1), 0);
        TEST_FLOAT_WITHIN(threshold, 0.0f, x[0], "saxpy x[0] failed");
        TEST_FLOAT_WITHIN(threshold, 0.0f, x[1], "saxpy x[1] failed");
        TEST_FLOAT_WITHIN(threshold, 0.0f, x[2], "saxpy x[2] failed");

        check_equal(scopy_(&n, z, &inc2, w, &inc1), 0);
        TEST_FLOAT_WITHIN(threshold, 1.0f, w[0], "scopy stride w[0] failed");
        TEST_FLOAT_WITHIN(threshold, 2.0f, w[1], "scopy stride w[1] failed");
        TEST_FLOAT_WITHIN(threshold, 3.0f, w[2], "scopy stride w[2] failed");

        check_equal(scopy_(&n, w, &incn, y, &inc1), 0);
        TEST_FLOAT_WITHIN(threshold, 3.0f, y[0], "scopy negative stride y[0] failed");
        TEST_FLOAT_WITHIN(threshold, 2.0f, y[1], "scopy negative stride y[1] failed");
        TEST_FLOAT_WITHIN(threshold, 1.0f, y[2], "scopy negative stride y[2] failed");

        check_equal(sswap_(&n, w, &inc1, y, &inc1), 0);
        TEST_FLOAT_WITHIN(threshold, 3.0f, w[0], "sswap w[0] failed");
        TEST_FLOAT_WITHIN(threshold, 2.0f, w[1], "sswap w[1] failed");
        TEST_FLOAT_WITHIN(threshold, 1.0f, w[2], "sswap w[2] failed");
        printf("[x] BLAS1 vector routines\n");
    }
    // Test BLAS2 vector/matrix routines
    {
        char        trans = 'N';
        int         m     = 2;
        int         n     = 3;
        int         lda   = 2;
        int         inc1  = 1;
        float       alpha = 1.0f;
        float       beta  = 0.0f;
        const float A[6]  = { 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f };
        const float x[3]  = { 1.0f, 2.0f, 3.0f };
        float       y[3]  = { 10.0f, 20.0f, 30.0f };
        const float threshold = 1.0e-06f;

        check_equal(sgemv_(&trans, &m, &n, &alpha, A, &lda, x, &inc1, &beta, y, &inc1), 0);
        TEST_FLOAT_WITHIN(threshold, 22.0f, y[0], "sgemv N y[0] failed");
        TEST_FLOAT_WITHIN(threshold, 28.0f, y[1], "sgemv N y[1] failed");

        {
            const float xt[2] = { 1.0f, 2.0f };

            trans = 'T';
            check_equal(sgemv_(&trans, &m, &n, &alpha, A, &lda, xt, &inc1, &beta, y, &inc1),
                         0);
        }
        TEST_FLOAT_WITHIN(threshold, 5.0f, y[0], "sgemv T y[0] failed");
        TEST_FLOAT_WITHIN(threshold, 11.0f, y[1], "sgemv T y[1] failed");
        TEST_FLOAT_WITHIN(threshold, 17.0f, y[2], "sgemv T y[2] failed");

        {
            const float gx[2] = { 1.0f, 2.0f };
            const float gy[3] = { 3.0f, 4.0f, 5.0f };
            float       G[6]  = { 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f };

            check_equal(sger_(&m, &n, &alpha, gx, &inc1, gy, &inc1, G, &lda), 0);
            TEST_FLOAT_WITHIN(threshold, 3.0f, G[0], "sger G[0] failed");
            TEST_FLOAT_WITHIN(threshold, 6.0f, G[1], "sger G[1] failed");
            TEST_FLOAT_WITHIN(threshold, 4.0f, G[2], "sger G[2] failed");
            TEST_FLOAT_WITHIN(threshold, 8.0f, G[3], "sger G[3] failed");
            TEST_FLOAT_WITHIN(threshold, 5.0f, G[4], "sger G[4] failed");
            TEST_FLOAT_WITHIN(threshold, 10.0f, G[5], "sger G[5] failed");
        }
        printf("[x] BLAS2 matrix-vector routines\n");
    }
    // Test vector statistics and distances
    {
        int         n4    = 4;
        int         n3    = 3;
        int         n2    = 2;
        int         inc1  = 1;
        int         ddof0 = 0;
        int         ddof1 = 1;
        float       value;
        float       eps = 1.0e-06f;
        float       v4[4] = { 1.0f, 2.0f, 3.0f, 4.0f };
        float       vn[2] = { 3.0f, 4.0f };
        const float a[3]  = { 1.0f, 2.0f, 3.0f };
        const float b[3]  = { 4.0f, 0.0f, 3.0f };
        const float threshold = 1.0e-06f;

        check_equal(svec_mean_(&n4, v4, &inc1, &value), 0);
        TEST_FLOAT_WITHIN(threshold, 2.5f, value, "svec_mean failed");
        check_equal(svec_variance_(&n4, v4, &inc1, &ddof0, &value), 0);
        TEST_FLOAT_WITHIN(threshold, 1.25f, value, "svec_variance population failed");
        check_equal(svec_variance_(&n4, v4, &inc1, &ddof1, &value), 0);
        TEST_FLOAT_WITHIN(threshold, 1.6666667f, value, "svec_variance sample failed");
        check_equal(svec_rms_(&n4, v4, &inc1, &value), 0);
        TEST_FLOAT_WITHIN(threshold, 2.7386128f, value, "svec_rms failed");

        check_equal(svec_normalize_(&n2, vn, &inc1, &eps, &value), 0);
        TEST_FLOAT_WITHIN(threshold, 5.0f, value, "svec_normalize norm failed");
        TEST_FLOAT_WITHIN(threshold, 0.6f, vn[0], "svec_normalize x[0] failed");
        TEST_FLOAT_WITHIN(threshold, 0.8f, vn[1], "svec_normalize x[1] failed");

        check_equal(svec_l1_distance_(&n3, a, &inc1, b, &inc1, &value), 0);
        TEST_FLOAT_WITHIN(threshold, 5.0f, value, "svec_l1_distance failed");
        check_equal(svec_linf_distance_(&n3, a, &inc1, b, &inc1, &value), 0);
        TEST_FLOAT_WITHIN(threshold, 3.0f, value, "svec_linf_distance failed");
        check_equal(svec_cosine_similarity_(&n3, a, &inc1, b, &inc1, &value), 0);
        TEST_FLOAT_WITHIN(threshold, 0.6948792f, value, "svec_cosine_similarity failed");
        printf("[x] Vector statistics and distances\n");
    }
    // Test small fixed-size matrix helpers
    {
        const float A2[4] = { 4.0f, 2.0f, 7.0f, 6.0f };
        float       I2[4];
        const float A3[9] = { 1.0f, 0.0f, 5.0f, 2.0f, 1.0f, 6.0f, 3.0f, 4.0f, 0.0f };
        float       I3[9];
        float       eps = 1.0e-06f;
        const float threshold = 1.0e-06f;
        const float I3exp[9] = { -24.0f, 20.0f, -5.0f, 18.0f, -15.0f,
                                  4.0f,   5.0f,  -4.0f, 1.0f };

        TEST_FLOAT_WITHIN(threshold, 10.0f, smat2_det_(A2), "smat2_det failed");
        check_equal(smat2_inv_(A2, I2, &eps), 0);
        TEST_FLOAT_WITHIN(threshold, 0.6f, I2[0], "smat2_inv I2[0] failed");
        TEST_FLOAT_WITHIN(threshold, -0.2f, I2[1], "smat2_inv I2[1] failed");
        TEST_FLOAT_WITHIN(threshold, -0.7f, I2[2], "smat2_inv I2[2] failed");
        TEST_FLOAT_WITHIN(threshold, 0.4f, I2[3], "smat2_inv I2[3] failed");

        TEST_FLOAT_WITHIN(threshold, 1.0f, smat3_det_(A3), "smat3_det failed");
        check_equal(smat3_inv_(A3, I3, &eps), 0);
        for (int i = 0; i < 9; ++i)
        {
            TEST_FLOAT_WITHIN(threshold, I3exp[i], I3[i], "smat3_inv failed");
        }
        printf("[x] Small fixed-size matrix helpers\n");
    }
    // Test linalg.h wrappers over miniblas helpers
    {
        const float A[6] = { 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f };
        const float x[3] = { 1.0f, 2.0f, 3.0f };
        const float v[4] = { 1.0f, 2.0f, 3.0f, 4.0f };
        float       y[2] = { 0.0f, 0.0f };
        float       G[6] = { 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f };
        float       vn[2] = { 3.0f, 4.0f };
        float       norm;
        const float A2[4] = { 4.0f, 2.0f, 7.0f, 6.0f };
        float       I2[4];
        const float threshold = 1.0e-06f;

        matvec("N", 2, 3, 1.0f, A, x, 0.0f, y);
        TEST_FLOAT_WITHIN(threshold, 22.0f, y[0], "matvec wrapper y[0] failed");
        TEST_FLOAT_WITHIN(threshold, 28.0f, y[1], "matvec wrapper y[1] failed");

        rank1update(G, y, x, 2, 3, 0.5f);
        TEST_FLOAT_WITHIN(threshold, 11.0f, G[0], "rank1update wrapper G[0] failed");
        TEST_FLOAT_WITHIN(threshold, 14.0f, G[1], "rank1update wrapper G[1] failed");
        TEST_FLOAT_WITHIN(threshold, 22.0f, G[2], "rank1update wrapper G[2] failed");
        TEST_FLOAT_WITHIN(threshold, 28.0f, G[3], "rank1update wrapper G[3] failed");

        TEST_FLOAT_WITHIN(threshold, 2.5f, vecmean(v, 4), "vecmean wrapper failed");
        TEST_FLOAT_WITHIN(threshold, 1.25f, vecvariance(v, 4, 0), "vecvariance wrapper failed");
        TEST_FLOAT_WITHIN(threshold, 3.7416575f, vecnorm(x, 3), "vecnorm wrapper failed");
        check_equal(vecnormalize(vn, 2, 1.0e-06f, &norm), 0);
        TEST_FLOAT_WITHIN(threshold, 5.0f, norm, "vecnormalize wrapper norm failed");
        TEST_FLOAT_WITHIN(threshold, 0.6f, vn[0], "vecnormalize wrapper x[0] failed");

        TEST_FLOAT_WITHIN(threshold, 10.0f, mat2det(A2), "mat2det wrapper failed");
        check_equal(mat2inv(A2, I2, 1.0e-06f), 0);
        TEST_FLOAT_WITHIN(threshold, 0.6f, I2[0], "mat2inv wrapper failed");
        printf("[x] linalg wrapper helpers\n");
    }

    // Test Matrix Multiplication
    {
        const float A[4]    = { 1, 4, 3, 2 };             // A = 2 rows x 2 columns
        const float B[6]    = { 2, 3, 5, 6, 3, 9 };       // B= 2 rows x 3 columns
        float       C[6]    = { 1, 1, 1, 1, 1, 1 };       // Output: A*B --> 2 rows x 3 columns
        const float Cexp[6] = { 35, 44, 71, 98, 92, 92 }; // C = 3*A*B + 2*C
        const float alpha   = 3.0f;
        const float beta    = 2.0f;
        matmul("N", "N", 2, 3, 2, alpha, A, B, beta, C);
        for (int i = 0; i < 6; i++)
        {
            TEST_FLOAT_WITHIN(1.0e-08f, Cexp[i], C[i], "Error in matrix multiplication");
        }
        printf("[x] test C = alpha*A*B + beta*C (matmul)\n");
    }
    {
        const float A[6]    = { 9, 6, -5, 10, -3, 9 }; // A = 2 rows x 3 columns
        float       B[4]    = { -1, -1, -1, -1 };      // Output: A*A' --> 2 rows x 2 columns
        const float Bexp[4] = { 115, -23, -23, 217 };  // B = 1*A*A' + 0*B
        const float alpha   = 1.0f;
        const float beta    = 0.0f;
        matmul("N", "T", 2, 2, 3, alpha, A, A, beta, B);
        for (int i = 0; i < 4; i++)
        {
            TEST_FLOAT_WITHIN(1.0e-08f, Bexp[i], B[i], "Error in matrix multiplication");
        }
        printf("[x] test C = A*A' (matmul)\n");
    }
    {
        const float A[4]    = { 1, 4, 3, 2 };       // A = 2 rows x 2 columns
        const float B[6]    = { 2, 3, 5, 6, 3, 9 }; // B= 2 rows x 3 columns
        float       C[6]    = { 2, 2, 2, 2, 2, 2 }; // Output: A*B --> 2 rows x 3 columns
        const float Cexp[6] = { 21, 18, 43.5f, 40.5f, 58.5f, 40.5f }; // C = 1.5*A'*B + 0*C
        const float alpha   = 1.5f;
        const float beta    = 0.0f;
        matmul("T", "N", 2, 3, 2, alpha, A, B, beta, C);
        for (int i = 0; i < 6; i++)
        {
            TEST_FLOAT_WITHIN(1.0e-08f, Cexp[i], C[i], "Error in matrix multiplication");
        }
        printf("[x] test C = alpha*A'*B (matmul)\n");
    }
    {
        const float A[]    = { 1, -10, 5, 3, -20, 7 }; // A = 3 rows x 2 columns
        const float B[]    = { -1, 4, -2, 5, -3, 6 };  // B= 2 rows x 3 columns
        float       C[]    = { 3, 3, 3, 3 };
        const float Cexp[] = { 4, 16, -16, -46 }; // C = 1*A'*B'
        const float alpha  = 1.0f;
        const float beta   = 0.0f;

        matmul("T", "T", 2, 2, 3, alpha, A, B, beta, C);
        for (int i = 0; i < 4; i++)
        {
            TEST_FLOAT_WITHIN(1.0e-08f, Cexp[i], C[i], "Error in matrix multiplication");
        }
        printf("[x] test C = A'*B' (matmul)\n");
    }
    // Test cholesky decomposition
    {
#define CHOLESKY_TEST_N 8
        const int n = CHOLESKY_TEST_N;
        float     L[CHOLESKY_TEST_N * CHOLESKY_TEST_N];
        hilbert(L, n); // L = hilbert(n) put hilbert matrix into L
        // matprint(L, n, n, "%10.8f", "H");
        const int result = cholesky(L, n, 0); // L = chol(L) inplace calc.
        check_equal(result, 0);
        // matprint(L, n, n, "%10.8f", "L (L*L'=H)");
        // test if L*L' actually is equal to H:
        float LLt[CHOLESKY_TEST_N * CHOLESKY_TEST_N];
        matmul("N", "T", n, n, n, 1.0f, L, L, 0.0, LLt); // LLt = L*L'
        float H[CHOLESKY_TEST_N * CHOLESKY_TEST_N];
        hilbert(H, n); // recreate expected result
        matprint(H, n, n, "%8.6f", "H");
        const float threshold = 1.5e-08f; // comparable error of independent MATLAB test
        for (int i = 0; i < n * n; i++)
        {
            TEST_FLOAT_WITHIN(threshold, H[i], LLt[i], "cholesky decomp. of Hilbert matrix failed");
        }
        printf("[x] Cholesky decomposition on close to singular matrix "
               "(cholesky)\n");
    }
    // Right-hand side triangular solve
    {
        const float L[]    = { 2, 3, 0, 1 };         // 2 x 2 matrix
        float       B[]    = { 8, 18, 28, 2, 4, 6 }; // 3 x 2 matrix
        float       Xexp[] = { 1, 3, 5, 2, 4, 6 };   // X*L = B
        trisolveright(L, B, 2, 3, "N");
        const float threshold = 1.0e-08f;
        for (int i = 0; i < 2 * 3; i++)
        {
            TEST_FLOAT_WITHIN(threshold, B[i], Xexp[i], "trisolveright failed");
        }
        printf("[x] Right-hand side triangular solve (trisolveright)\n");
    }
    {
        const float L[]    = { 2, 3, 0, 1 };            // 2 x 2 matrix
        float       B[]    = { 12, 10, 8, 21, 17, 13 }; // 3 x 2 matrix
        float       Xexp[] = { 6, 5, 4, 3, 2, 1 };      // X*L' = B
        trisolveright(L, B, 2, 3, "T");
        const float threshold = 1.0e-08f;
        for (int i = 0; i < 2 * 3; i++)
        {
            TEST_FLOAT_WITHIN(threshold, B[i], Xexp[i], "trisolveright with transpose failed");
        }
        printf("[x] Right-hand side triangular solve with transpose (trisolveright)\n");
    }
    {
        float L[3 * 3];
        float Xexp[3 * 3];
        float B[3 * 3];
        hilbert(L, 3);
        hilbert(Xexp, 3);
        cholesky(L, 3, 0);
        matmul("N", "N", 3, 3, 3, 1.0f, Xexp, L, 0.0f, B);
        // matprint(Xexp, 3, 3, "%8.6f", "X");
        // matprint(L, 3, 3, "%8.6f", "L");
        // matprint(B, 3, 3, "%8.6f", "B");
        trisolveright(L, B, 3, 3, "N");
        const float threshold = 1.0e-07f;
        for (int i = 0; i < 3 * 3; i++)
        {
            TEST_FLOAT_WITHIN(threshold, B[i], Xexp[i], "trisolveright with transpose failed");
        }
        printf("[x] Right-hand side triangular solve test case #2 (trisolveright)\n");
    }
    /* FIXME: add test for cases were trisolveright could fail */
    {
        const float A[3 * 3] = { 9, 6, 8, 6, 6, 7, 8, 7, 9 };
        float       U[3 * 3] = { -1, 0, 0, -1, -1, 0, -1, -1, -1 };
        float       d[3]     = { -1, -1, -1 };
        udu(A, U, d, 3);

        // matprint(A, 3, 3, "%6.4f", "A");
        // matprint(U, 3, 3, "%6.4f", "U");
        // matprint(d, 3, 1, "%6.4f", "d");

        const float Uexp[3 * 3] = { 1, 0, 0, -0.4f, 1, 0, 0.8889f, 0.7778f, 1 };
        const float dexp[3]     = { 1.8f, 0.5556f, 9 };
        const float threshold   = 1.0e-03f;
        for (int i = 0; i < 3 * 3; i++)
        {
            TEST_FLOAT_WITHIN(threshold, U[i], Uexp[i], "UDU: U test failed");
        }
        for (int i = 0; i < 3; i++)
        {
            TEST_FLOAT_WITHIN(threshold, d[i], dexp[i], "UDU: d test failed");
        }
    }
    {
        const float negative[1] = { -1.0f };
        const float zero[1]     = { 0.0f };
        const float nan_value[1] = { NAN };
        const float final_zero_pivot[4] = { 0.0f, 0.0f, 0.0f, 1.0f };
        float U[4];
        float d[2];

        check_equal(udu(negative, U, d, 1), -1);
        check_equal(udu(zero, U, d, 1), -1);
        check_equal(udu(nan_value, U, d, 1), -1);
        check_equal(udu(final_zero_pivot, U, d, 2), -1);
        check_equal(udu(NULL, U, d, 1), -1);
        check_equal(udu(negative, NULL, d, 1), -1);
        check_equal(udu(negative, U, NULL, 1), -1);
        check_equal(udu(negative, U, d, 0), -1);
        printf("[x] UDU rejects invalid/final non-positive pivots\n");
    }
    // Test magnetometer yaw
    {
        const float roll_rad         = DEG2RAD(45.0f);
        const float pitch_rad        = DEG2RAD(70.0f);
        const float yaw_expected_rad = DEG2RAD(170.0f);
        const float mb[3]            = { 40.2481f, -27.63536f, -22.7238f };
        const float yaw_rad          = nav_mag_heading(mb, roll_rad, pitch_rad);
        const float threshold        = 0.001f;

        TEST_FLOAT_WITHIN(threshold, yaw_rad, yaw_expected_rad,
                          "Magnetometer heading test failed (nav_mag_heading)");
        printf("[x] Yaw from magnetometer (nav_mag_heading)\n");
    }
}

static void testkalmanbounds(void)
{
    float x[1]   = { 7.0f };
    float P[1]   = { 9.0f };
    float U[1]   = { 1.0f };
    float d[1]   = { 2.0f };
    float z[1]   = { 0.0f };
    float R[1]   = { 1.0f };
    float Ht[1]  = { 1.0f };
    float Phi[1] = { 1.0f };

    check_equal(kalman_takasu(x, P, z, R, Ht, 0, 1, 0.0f, NULL), KFCORE_KALMAN_INVALID_ARGUMENT);
    check_equal(kalman_takasu(NULL, P, z, R, Ht, 1, 1, 0.0f, NULL), KFCORE_KALMAN_INVALID_ARGUMENT);

    check_equal(kalman_udu_scalar(x, U, d, 0.0f, NAN, Ht, 1), KFCORE_KALMAN_INVALID_ARGUMENT);
    check_equal(kalman_udu(x, U, d, z, R, Ht, 1, 0, 0.0f, 0), KFCORE_KALMAN_INVALID_ARGUMENT);
    check_equal(decorrelate(z, Ht, R, 0, 1), KFCORE_KALMAN_INVALID_ARGUMENT);

    check_equal(kalman_ekf_takasu_predict(x, P, NULL, NULL, NULL, 33, 0, NULL), -1);
    check_equal(kalman_ekf_takasu_update(x, P, z, R, NULL, 1, 4, 0.0f, NULL, NULL), -1);
    check_equal(kalman_ekf_udu_predict(x, U, d, NULL, NULL, NULL, 33, 0, NULL), -1);
    check_equal(kalman_ekf_udu_update(x, U, d, z, R, NULL, 1, 4, 0.0f, 0, NULL), -1);

    check_equal(kalman_ukf_predict(x, P, NULL, NULL, 33, NULL, NULL), -1);
    check_equal(kalman_ukf_update(x, P, z, R, NULL, 1, 4, NULL, 0.0f, NULL, NULL), -1);

    (void)Phi;
    printf("[x] Kalman validation and remaining fixed-wrapper guards\n");
}

static void testnavtoolbox(void)
{
    printf("Running navtoolbox tests...\n");

    // Test Roll Pitch From Accelerometer, body2nav
    {
        const float f_body[3] = { 0.0f, 0.0f, -0.01f }; /* close to free fall */
        float       roll_rad, pitch_rad;
        nav_roll_pitch_from_accelerometer(f_body, &roll_rad, &pitch_rad);
        TEST_FLOAT_WITHIN(DEG2RAD(1.0e-06f), DEG2RAD(0.0f), roll_rad,
                          "Roll angle calculation incorrect");
        TEST_FLOAT_WITHIN(DEG2RAD(1.0e-06f), DEG2RAD(0.0f), pitch_rad,
                          "Pitch angle calculation incorrect");
    }
    {
        const float f_nav[3] = { 0.0f, 0.0f, -GRAVITY };
        float       R[9];
        float       f_body[3];
        nav_matrix_body2nav(DEG2RAD(10.0f), DEG2RAD(20.0f), 0.0f, R);
        // matprint(R, 3, 3, "%6.3f", "R");
        matmul("T", "N", 3, 1, 3, 1.0f, R, f_nav, 0.0f, f_body);
        // printf("f_body = %.2f %.2f %.2f\n", f_body[0], f_body[1], f_body[2]);
        float roll_rad, pitch_rad;
        nav_roll_pitch_from_accelerometer(f_body, &roll_rad, &pitch_rad);
        // printf("%.1f %.1f\n", RAD2DEG(roll_rad), RAD2DEG(pitch_rad));
        TEST_FLOAT_WITHIN(DEG2RAD(1.0e-06f), DEG2RAD(10.0f), roll_rad,
                          "Roll angle calculation incorrect");
        TEST_FLOAT_WITHIN(DEG2RAD(1.0e-06f), DEG2RAD(20.0f), pitch_rad,
                          "Pitch angle calculation incorrect");
        printf("[x] Body to navigation frame transformation "
               "(nav_matrix_body2nav)\n");
        printf("[x] Initial alignment from accelerometer "
               "(nav_roll_pitch_from_accelerometer)\n");
    }
    // Navigation/IMU helper filters
    {
        TEST_FLOAT_WITHIN(1.0e-06f, -PI_FLOAT, nav_wrap_pi(PI_FLOAT),
                          "nav_wrap_pi upper wrap failed");
        TEST_FLOAT_WITHIN(1.0e-06f, 0.0f, nav_wrap_pi(2.0f * PI_FLOAT),
                          "nav_wrap_pi full rotation failed");
        printf("[x] Navigation angle wrapping\n");
    }
    {
        float q[4];
        float roll;
        float pitch;
        float yaw;

        check_equal(nav_quat_from_euler(0.1f, 0.2f, 0.3f, q), 0);
        check_equal(nav_quat_to_euler(q, &roll, &pitch, &yaw), 0);
        TEST_FLOAT_WITHIN(1.0e-06f, 0.1f, roll, "Quaternion roll round-trip failed");
        TEST_FLOAT_WITHIN(1.0e-06f, 0.2f, pitch, "Quaternion pitch round-trip failed");
        TEST_FLOAT_WITHIN(1.0e-06f, 0.3f, yaw, "Quaternion yaw round-trip failed");
        printf("[x] Quaternion Euler conversion\n");
    }
    {
        float q[4];
        float R_from_euler[9];
        float R_from_q[9];

        nav_matrix_body2nav(0.1f, 0.2f, 0.3f, R_from_euler);
        check_equal(nav_quat_from_matrix_body2nav(R_from_euler, q), 0);
        check_equal(nav_quat_to_matrix_body2nav(q, R_from_q), 0);
        for (int i = 0; i < 9; ++i)
        {
            TEST_FLOAT_WITHIN(1.0e-06f, R_from_euler[i], R_from_q[i],
                              "Quaternion matrix round-trip failed");
        }
        printf("[x] Quaternion matrix conversion\n");
    }
    {
        float       q[4];
        const float body[3] = { 1.0f, 0.0f, 0.0f };
        float       nav[3];
        float       body_roundtrip[3];

        check_equal(nav_quat_from_euler(0.0f, 0.0f, DEG2RAD(90.0f), q), 0);
        check_equal(nav_quat_rotate_body_to_nav(q, body, nav), 0);
        TEST_FLOAT_WITHIN(1.0e-06f, 0.0f, nav[0], "Quaternion body-to-nav x failed");
        TEST_FLOAT_WITHIN(1.0e-06f, 1.0f, nav[1], "Quaternion body-to-nav y failed");
        check_equal(nav_quat_rotate_nav_to_body(q, nav, body_roundtrip), 0);
        TEST_FLOAT_WITHIN(1.0e-06f, 1.0f, body_roundtrip[0],
                          "Quaternion nav-to-body x failed");
        TEST_FLOAT_WITHIN(1.0e-06f, 0.0f, body_roundtrip[1],
                          "Quaternion nav-to-body y failed");
        printf("[x] Quaternion vector rotation\n");
    }
    {
        float       q[4];
        const float gyro[3] = { 0.0f, 0.0f, PI_FLOAT };
        float       yaw;

        nav_quat_identity(q);
        check_equal(nav_quat_integrate_gyro(q, gyro, 0.5f), 0);
        check_equal(nav_quat_to_euler(q, NULL, NULL, &yaw), 0);
        TEST_FLOAT_WITHIN(1.0e-06f, PI_FLOAT * 0.5f, yaw,
                          "Quaternion gyro integration failed");
        printf("[x] Quaternion gyro integration\n");
    }
    {
        float       q[4];
        const float gyro[3]  = { 0.1f, 0.0f, 0.0f };
        const float accel[3] = { 0.0f, 0.0f, -GRAVITY };
        float       roll;

        nav_quat_identity(q);
        check_equal(nav_quat_complementary_imu(q, gyro, accel, 1.0f, 0.5f), 0);
        check_equal(nav_quat_to_euler(q, &roll, NULL, NULL), 0);
        check(roll > 0.0f && roll < 0.1f);
        printf("[x] Quaternion IMU complementary filter\n");
    }
    {
        float       q[4];
        const float gyro[3]        = { 0.0f, 0.0f, 0.2f };
        const float accel[3]       = { 0.0f, 0.0f, -GRAVITY };
        const float mag_body[3]    = { 1.0f, 0.0f, 0.0f };
        const float mag_ref_nav[3] = { 1.0f, 0.0f, 0.0f };
        float       yaw;

        nav_quat_identity(q);
        check_equal(nav_quat_complementary_marg(q, gyro, accel, mag_body, mag_ref_nav, 1.0f,
                                                 0.5f, 0.5f),
                     0);
        check_equal(nav_quat_to_euler(q, NULL, NULL, &yaw), 0);
        check(yaw > 0.0f && yaw < 0.2f);
        printf("[x] Quaternion MARG complementary filter\n");
    }
    {
        float       q[4];
        const float accel[3] = { 0.0f, 0.0f, -GRAVITY };
        float       linear[3];

        nav_quat_identity(q);
        check_equal(nav_remove_gravity_body_quat(accel, q, linear), 0);
        TEST_FLOAT_WITHIN(1.0e-06f, 0.0f, linear[0], "Quaternion gravity removal x failed");
        TEST_FLOAT_WITHIN(1.0e-06f, 0.0f, linear[1], "Quaternion gravity removal y failed");
        TEST_FLOAT_WITHIN(1.0e-06f, 0.0f, linear[2], "Quaternion gravity removal z failed");
        printf("[x] Quaternion gravity removal\n");
    }
    {
        float       roll  = 0.0f;
        float       pitch = 0.0f;
        float       yaw   = 0.0f;
        const float gyro[3] = { 0.1f, 0.2f, 0.3f };

        check_equal(nav_euler_integrate_gyro(&roll, &pitch, &yaw, gyro, 1.0f), 0);
        TEST_FLOAT_WITHIN(1.0e-06f, 0.1f, roll, "Gyro roll integration failed");
        TEST_FLOAT_WITHIN(1.0e-06f, 0.2f, pitch, "Gyro pitch integration failed");
        TEST_FLOAT_WITHIN(1.0e-06f, 0.3f, yaw, "Gyro yaw integration failed");
        printf("[x] Euler gyro integration\n");
    }
    {
        float       roll     = 0.0f;
        float       pitch    = 0.0f;
        const float gyro[3]  = { 0.1f, 0.2f, 0.0f };
        const float accel[3] = { 0.0f, 0.0f, -GRAVITY };

        check_equal(nav_complementary_roll_pitch(&roll, &pitch, gyro, accel, 1.0f, 0.5f), 0);
        TEST_FLOAT_WITHIN(1.0e-03f, 0.05f, roll, "Complementary roll failed");
        TEST_FLOAT_WITHIN(1.0e-03f, 0.1f, pitch, "Complementary pitch failed");
        printf("[x] Complementary roll/pitch filter\n");
    }
    {
        float       yaw     = 0.0f;
        const float mag[3]  = { 1.0f, 0.0f, 0.0f };

        check_equal(nav_complementary_yaw(&yaw, 0.2f, mag, 0.0f, 0.0f, 1.0f, 0.5f), 0);
        TEST_FLOAT_WITHIN(1.0e-03f, 0.1f, yaw, "Complementary yaw failed");
        printf("[x] Complementary yaw filter\n");
    }
    {
        const float accel_ok[3]  = { 0.0f, 0.0f, -GRAVITY };
        const float accel_bad[3] = { 0.0f, 0.0f, -20.0f };
        float       norm;

        check_equal(nav_accel_gravity_gate(accel_ok, 0.1f, &norm), 1);
        TEST_FLOAT_WITHIN(1.0e-06f, GRAVITY, norm, "Gravity gate norm failed");
        check_equal(nav_accel_gravity_gate(accel_bad, 0.1f, NULL), 0);
        printf("[x] Accelerometer gravity gate\n");
    }
    {
        const float accel[3] = { 0.0f, 0.0f, -GRAVITY };
        float       linear[3];

        nav_remove_gravity_body(accel, 0.0f, 0.0f, 0.0f, linear);
        TEST_FLOAT_WITHIN(1.0e-06f, 0.0f, linear[0], "Gravity removal x failed");
        TEST_FLOAT_WITHIN(1.0e-06f, 0.0f, linear[1], "Gravity removal y failed");
        TEST_FLOAT_WITHIN(1.0e-06f, 0.0f, linear[2], "Gravity removal z failed");
        printf("[x] Body-frame gravity removal\n");
    }
    {
        const float accel[3] = { 0.0f, 0.0f, -GRAVITY };
        const float gyro[3]  = { 0.01f, 0.02f, 0.01f };
        float       accel_norm;
        float       gyro_norm;

        check_equal(nav_zupt_static_gate(accel, gyro, 0.1f, 0.05f, &accel_norm, &gyro_norm), 1);
        TEST_FLOAT_WITHIN(1.0e-06f, GRAVITY, accel_norm, "ZUPT accel norm failed");
        check(gyro_norm > 0.0f);
        printf("[x] ZUPT/static detector\n");
    }
    {
        nav_fusion2d fusion;
        const float  initial_state[NAV_FUSION2D_STATE_SIZE] = { 0.0f };
        const float  initial_std[NAV_FUSION2D_STATE_SIZE]   = { 10.0f, 10.0f, 1.0f, 1.0f,
                                                                 1.0f,  0.1f,  0.1f, 0.1f };
        const float  accel_body[2] = { 1.0f, 0.0f };
        const float  q[NAV_FUSION2D_STATE_SIZE] = { 0.01f, 0.01f, 0.1f, 0.1f,
                                                     0.01f, 0.0f,  0.0f, 0.0f };
        const float  Rpos[4]       = { 1.0f, 0.0f, 0.0f, 1.0f };
        const float  Rvel[4]       = { 0.25f, 0.0f, 0.0f, 0.25f };
        const float  position[2]   = { 1.2f, 0.1f };
        const float  body_vel[2]   = { 1.0f, 0.0f };

        check_equal(nav_fusion2d_init(&fusion, initial_state, initial_std), 0);
        check_equal(nav_fusion2d_predict_imu(&fusion, accel_body, 0.1f, 1.0f, q), 0);
        TEST_FLOAT_WITHIN(1.0e-06f, 0.5f, fusion.x[NAV_FUSION2D_X],
                          "Fusion2D predict x failed");
        TEST_FLOAT_WITHIN(1.0e-06f, 1.0f, fusion.x[NAV_FUSION2D_VX],
                          "Fusion2D predict vx failed");
        TEST_FLOAT_WITHIN(1.0e-06f, 0.1f, fusion.x[NAV_FUSION2D_YAW],
                          "Fusion2D predict yaw failed");

        check_equal(nav_fusion2d_update_position(&fusion, position, Rpos, 0.0f, NULL), 0);
        check(fusion.x[NAV_FUSION2D_X] > 0.5f);
        check_equal(nav_fusion2d_update_velocity_body(&fusion, body_vel, Rvel, 0.0f, NULL), 0);
        check_equal(nav_fusion2d_update_yaw(&fusion, 0.0f, 0.1f, 0.0f, NULL), 0);
        check(fusion.x[NAV_FUSION2D_YAW] >= -PI_FLOAT && fusion.x[NAV_FUSION2D_YAW] < PI_FLOAT);
        check_equal(nav_fusion2d_update_zupt(&fusion, Rvel, 0.0f, NULL), 0);
        printf("[x] 2D navigation fusion system layer\n");
    }
    {
        nav_fusion3d fusion;
        float        q0[4];
        float        q_meas[4];
        const float  p0[3]     = { 0.0f, 0.0f, 0.0f };
        const float  v0[3]     = { 0.0f, 0.0f, 0.0f };
        const float  ba[3]     = { 0.0f, 0.0f, 0.0f };
        const float  bg[3]     = { 0.0f, 0.0f, 0.0f };
        const float  accel[3]  = { 0.0f, 0.0f, -GRAVITY };
        const float  gyro[3]   = { 0.0f, 0.0f, 0.0f };
        const float  pos[3]    = { 1.0f, -2.0f, 0.5f };
        const float  zero[3]   = { 0.0f, 0.0f, 0.0f };
        const float  std15[NAV_FUSION3D_ERROR_SIZE] = { 10.0f, 10.0f, 10.0f,
                                                         1.0f,  1.0f,  1.0f,
                                                         1.0f,  1.0f,  1.0f,
                                                         0.1f,  0.1f,  0.1f,
                                                         0.1f,  0.1f,  0.1f };
        const float  qproc[NAV_FUSION3D_ERROR_SIZE] = { 0.01f, 0.01f, 0.01f,
                                                         0.01f, 0.01f, 0.01f,
                                                         0.01f, 0.01f, 0.01f,
                                                         0.0f,  0.0f,  0.0f,
                                                         0.0f,  0.0f,  0.0f };
        const float  R3[9] = { 0.25f, 0.0f, 0.0f, 0.0f, 0.25f,
                               0.0f,  0.0f, 0.0f, 0.25f };
        float        yaw_before;
        float        yaw_after;

        nav_quat_identity(q0);
        check_equal(nav_fusion3d_init(&fusion, p0, v0, q0, ba, bg, std15), 0);
        check_equal(nav_fusion3d_predict_imu(&fusion, accel, gyro, 1.0f, qproc), 0);
        TEST_FLOAT_WITHIN(1.0e-06f, 0.0f, fusion.p[0], "Fusion3D stationary p x failed");
        TEST_FLOAT_WITHIN(1.0e-06f, 0.0f, fusion.v[2], "Fusion3D stationary v z failed");

        check_equal(nav_fusion3d_update_position(&fusion, pos, R3, 0.0f, NULL), 0);
        check(fusion.p[0] > 0.0f);
        check(fusion.p[1] < 0.0f);

        check_equal(nav_quat_from_euler(0.0f, 0.0f, 0.2f, fusion.q), 0);
        check_equal(nav_quat_from_euler(0.0f, 0.0f, 0.0f, q_meas), 0);
        check_equal(nav_quat_to_euler(fusion.q, NULL, NULL, &yaw_before), 0);
        check_equal(nav_fusion3d_update_attitude(&fusion, q_meas, R3, 0.0f, NULL), 0);
        check_equal(nav_quat_to_euler(fusion.q, NULL, NULL, &yaw_after), 0);
        check(fabsf(yaw_after) < fabsf(yaw_before));

        check_equal(nav_fusion3d_update_velocity(&fusion, zero, R3, 0.0f, NULL), 0);
        check_equal(nav_fusion3d_update_zupt(&fusion, R3, 0.0f, NULL), 0);
        printf("[x] 3D quaternion error-state fusion system layer\n");
    }
    // Kalman Filter Tests
    {
        const float R[3 * 3]  = { 0.25f, 0, 0, 0, 0.25f, 0, 0, 0, 0.25f };
        const float dz[3]     = { 0.2688f, 0.9169f, -1.1294f };
        const float Ht[4 * 3] = { 8, 1, 6, 1, 3, 5, 7, 2, 4, 9, 2, 3 };
        float       x[4]      = { 1, 1, 1, 1 };
        float       P[4 * 4]  = { 0.04f, 0, 0, 0, 0, 0.04f, 0, 0, 0, 0, 0.04f, 0, 0, 0, 0, 0.04f };
        int         result    = kalman_takasu(x, P, dz, R, Ht, 4, 3, 0.0f, NULL);
        check_equal(result, 0);
        const float xexp[4]     = { 0.9064f, 0.9046f, 1.2017f, 0.9768f };
        const float threshold   = 1.0e-04f;
        const float Pexp[4 * 4] = {
            0.0081f,  0.0000f,  0.0000f, 0.0000f, -0.0006f, 0.0063f,  0.0000f,  0.0000f,
            -0.0056f, -0.0006f, 0.0081f, 0.0000f, -0.0021f, -0.0102f, -0.0021f, 0.0367f
        }; /* upper triangular part is valid */
        // matprint(x, 4, 1, "%6.3f", "x");
        // matprint(P, 4, 4, "%6.3f", "P");
        for (int i = 0; i < 4; i++)
        {
            TEST_FLOAT_WITHIN(threshold, x[i], xexp[i],
                              "nav_kalman state vector calculation failed");
        }
        for (int i = 0; i < 4 * 4; i++)
        {
            TEST_FLOAT_WITHIN(threshold, P[i], Pexp[i],
                              "nav_kalman covariance matrix calculation failed");
        }
        printf("[x] Kalman Filter Update (nav_kalman)\n");
    }
    // Test same kalman filter but now with an outlier
    {
        const float R[3 * 3] = { 0.25f, 0, 0, 0, 0.25f, 0, 0, 0, 0.25f };
        const float dz[3] = { 0.2688f, 0.9169f, -100.1294f }; // adding outlier to 3rd measurement
        const float Ht[4 * 3] = { 8, 1, 6, 1, 3, 5, 7, 2, 4, 9, 2, 3 };
        float       x[4]      = { 1, 1, 1, 1 };
        float       P[4 * 4]  = { 0.04f, 0, 0, 0, 0, 0.04f, 0, 0, 0, 0, 0.04f, 0, 0, 0, 0, 0.04f };
        float       chi2;
        int         result = kalman_takasu(x, P, dz, R, Ht, 4, 3, 7.8147f, &chi2);
        check_equal(result, -2);
        const float xexp[4]   = { 1, 1, 1, 1 };
        float Pexp[4 * 4]     = { 0.04f, 0, 0, 0, 0, 0.04f, 0, 0, 0, 0, 0.04f, 0, 0, 0, 0, 0.04f };
        const float threshold = 1.0e-04f;
        const float chi2exp   = 1622.8f; // from testcasegen.m kalman_takasu_robust()
        TEST_FLOAT_WITHIN(0.1f, chi2, chi2exp, "outlier test chi2 result incorrect");
        for (int i = 0; i < 4; i++)
        {
            TEST_FLOAT_WITHIN(threshold, x[i], xexp[i], "nav_kalman failed to reject outlier");
        }
        for (int i = 0; i < 4 * 4; i++)
        {
            TEST_FLOAT_WITHIN(threshold, P[i], Pexp[i],
                              "nav_kalman outlier modified covariance matrix");
        }
        printf("[x] Kalman Filter Outlier Test (nav_kalman)\n");
    }
    {
        const float R[3 * 3]  = { 0.25f, 0, 0, 0, 0.25f, 0, 0, 0, 0.25f };
        const float z[3]      = { 16.2688f, 17.9169f, 16.8706f };
        const float Ht[4 * 3] = { 8, 1, 6, 1, 3, 5, 7, 2, 4, 9, 2, 3 };
        float       x[4]      = { 1, 1, 1, 1 };
        float       P[4 * 4]  = { 0.04f, 0, 0, 0, 0, 0.04f, 0, 0, 0, 0, 0.04f, 0, 0, 0, 0, 0.04f };
        float       U[4 * 4];
        float       d[4];
        int         result;

        result = udu(P, U, d, 4);
        check_equal(result, 0);

        result = kalman_udu(x, U, d, z, R, Ht, 4, 3, 0.0f, 0);
        check_equal(result, 0);

        const float xexp[4]     = { 0.906426012f, 0.904562052f, 1.201702724f, 0.976775052f };
        const float threshold   = 1.0e-06f;
        const float Uexp[4 * 4] = { 1.000000000000000f,
                                    0.0f,
                                    0.0f,
                                    0.0f,
                                    -0.619422572178478f,
                                    1.000000000000000f,
                                    0.0f,
                                    0.0f,
                                    -0.717109934386682f,
                                    -0.147377605926585f,
                                    1.000000000000000f,
                                    0.0f,
                                    -0.055997010835721f,
                                    -0.277195167517748f,
                                    -0.055997010835721f,
                                    1.000000000000000f };
        const float dexp[4]     = { 2.62467e-03f, 3.259279e-03f, 7.977724e-03f, 3.67391e-02f };
        // matprint(x, 4, 1, "%6.3f", "x");
        // matprint(U, 4, 4, "%6.3f", "U");
        // matprint(d, 4, 1, "%6.3f", "d");
        for (int i = 0; i < 4; i++)
        {
            TEST_FLOAT_WITHIN(threshold, x[i], xexp[i],
                              "kalman_udu state vector calculation failed");
            TEST_FLOAT_WITHIN(threshold, d[i], dexp[i], "kalman_udu d[] calculation failed");
        }
        for (int i = 0; i < 4 * 4; i++)
        {
            TEST_FLOAT_WITHIN(threshold, U[i], Uexp[i], "kalman_udu U matrix calculation failed");
        }
        printf("[x] Kalman Filter Update (kalman_udu)\n");
    }
    // Temporal Update Test (source: predict_test())
    {
        const float Q[]      = { 0.1f, 0.2f };
        const float G[]      = { 1, 0, 0.5f, 0, 1, 0.5f };
        float       x[]      = { 1, 2, 3 };
        const float Phi[]    = { 1, 0, 0, 0.5, 1, 0, 0.25, 0.1, 1 };
        float       P[3 * 3] = { 1.050000f, 0.170000f,  -0.180000f, 0.170000f, 1.260000f,
                                 0.420000f, -0.180000f, 0.420000f,  1.040000f };
        int         n        = 3;
        int         r        = 2;

        kalman_predict(x, P, Phi, G, Q, n, r);

        float x_exp[3 * 1] = { 2.750000f, 2.300000f, 3.000000f };
        float P_exp[3 * 3] = { 1.715000f, 0.934000f, 0.340000f, 0.934000f, 1.554400f,
                               0.624000f, 0.340000f, 0.624000f, 1.115000f };

        const float threshold = 0.001f;
        for (int i = 0; i < n; i++)
        {
            TEST_FLOAT_WITHIN(threshold, x[i], x_exp[i], "kalman_predict x calculation failed");
        }
        for (int i = 0; i < n * n; i++)
        {
            TEST_FLOAT_WITHIN(threshold, P[i], P_exp[i],
                              "kalman_udu_predict U matrix calculation failed");
        }
        printf("[x] Kalman Filter Prediction Test\n");
    }
    // decorr Test
    {
        float R[4] = { 1.328125f, 8.45f, 8.45f, 56.2525f };
        float z[]  = { 16.25f, -11.0f };
        float Ht[] = { 1.0f, -0.5f, 0.25f, 0.1f, 5.0f, -2.0f };

        const float zexp[]    = { 14.100479758212652f, -72.481609669099413f };
        const float Hexp[]    = { 0.867721831274625f,  -0.433860915637312f, 0.216930457818656f,
                                  -3.968112807452599f, 5.183967022924173f,  -2.275160677878139f };
        const float threshold = 1.0e-04f;

        int result = decorrelate(z, Ht, R, 3, 2);

        check_equal(result, 0);
        // matprint(z, 2, 1, "%9.7f", "zdecorr");
        // matprint(Ht, 3, 2, "%9.7f", "Hdecorr'");

        for (int i = 0; i < 2; i++)
        {
            TEST_FLOAT_WITHIN(threshold, z[i], zexp[i], "decorrelate z calculation failed");
        }
        for (int i = 0; i < 2 * 2; i++)
        {
            TEST_FLOAT_WITHIN(threshold, Ht[i], Hexp[i], "decorrelate H calculation failed");
        }
        printf("[x] Measurement decorrelation test (decorrelate)\n");
    }
    // Robust UDU Kalman Filter Test
    {
        int         result;
        const float P[]  = { 144.010f, 120.0120f, 120.012f, 100.0144f };
        float       Ht[] = { 1.0f, -0.5f, 0.1f, 5.0f };
        float       x[]  = { 10.0f, -5.0f };
        float       z[]  = { 1.250000000000000e+01f, -1.240000000000000e+02f };
        float       R[]  = { 1.328125000000000e+00f, 8.449999999999999e+00f, 8.449999999999999e+00f,
                             5.625250000000000e+01f };

        float U[2 * 2];
        float d[2];
        result = udu(P, U, d, 2);
        check_equal(result, 0);
        const float chi2_threshold = 3.8415f;

        decorrelate(z, Ht, R, 2, 2);
        mateye(R, 2); // set R to eye(2)
        result = kalman_udu(x, U, d, z, R, Ht, 2, 2, chi2_threshold, 1);
        check_equal(result, 0);

        const float x_robust_exp[] = { 9.918531929653216f, -5.068338573737113f };
        const float U_exp[]        = { 1.0f, 0, 1.198931661436124f, 1.0f };
        const float d_exp[]        = { 1.932847767499027e-03f, 2.641859030819114e+00f };
        const float threshold      = 1.0e-04f;
        for (int i = 0; i < 2; i++)
        {
            TEST_FLOAT_WITHIN(threshold, x[i], x_robust_exp[i],
                              "kalman_udu robust state vector calculation failed");
            TEST_FLOAT_WITHIN(threshold, d[i], d_exp[i],
                              "kalman_udu robust d[] calculation failed");
        }
        for (int i = 0; i < 2 * 2; i++)
        {
            TEST_FLOAT_WITHIN(threshold, U[i], U_exp[i],
                              "kalman_udu robust U matrix calculation failed");
        }
        printf("[x] Robust UDU Kalman Filter Test with Outlier\n");
    }
    // UDU Temporal Update Test (source: thornton_test())
    {
        const float Q[]   = { 0.1f, 0.2f };
        const float G[]   = { 1, 0, 0.5f, 0, 1, 0.5f };
        float       x[]   = { 1, 2, 3 };
        const float Phi[] = { 1, 0, 0, 0.5, 1, 0, 0.25, 0.1, 1 };
        float       U[]   = { 1, 0, 0, 0.2226f, 1, 0, -0.1731f, 0.4038f, 1 };
        float       d[]   = { 0.9648f, 1.0904f, 1.0400f };
        int         n     = 3;
        int         r     = 2;

        const float x_exp[] = { 2.7500f, 2.3000f, 3.0000 };
        const float U_exp[] = { 1, 0, 0, 0.6171f, 1, 0, 0.3049, 0.5596, 1 };
        const float d_exp[] = { 1.1524f, 1.2052f, 1.1150f };

        kalman_udu_predict(x, U, d, Phi, G, Q, n, r);

        const float threshold = 0.001f;
        for (int i = 0; i < n; i++)
        {
            TEST_FLOAT_WITHIN(threshold, x[i], x_exp[i], "kalman_udu_predict x calculation failed");
            TEST_FLOAT_WITHIN(threshold, d[i], d_exp[i], "kalman_udu_predict d calculation failed");
        }
        for (int i = 0; i < n * n; i++)
        {
            TEST_FLOAT_WITHIN(threshold, U[i], U_exp[i],
                              "kalman_udu_predict U matrix calculation failed");
        }
        printf("[x] UDU Kalman Filter Prediction Test\n");
    }
}

static void testframetransform(void)
{
    const float threshold = 1.0e-06f;

    printf("Running frame_transform tests...\n");

    {
        float R[9];
        float t[3];

        frame_pose_identity(R, t);
        TEST_FLOAT_WITHIN(threshold, 1.0f, R[0], "Identity R00 failed");
        TEST_FLOAT_WITHIN(threshold, 1.0f, R[4], "Identity R11 failed");
        TEST_FLOAT_WITHIN(threshold, 1.0f, R[8], "Identity R22 failed");
        TEST_FLOAT_WITHIN(threshold, 0.0f, t[0], "Identity tx failed");
        TEST_FLOAT_WITHIN(threshold, 0.0f, t[1], "Identity ty failed");
        TEST_FLOAT_WITHIN(threshold, 0.0f, t[2], "Identity tz failed");
        printf("[x] Pose identity\n");
    }

    {
        const float R_a2b[9] = { 0.0f, 1.0f, 0.0f,
                                -1.0f, 0.0f, 0.0f,
                                 0.0f, 0.0f, 1.0f };
        const float t_a_in_b[3] = { 10.0f, 20.0f, 30.0f };
        const float p_a[3]      = { 1.0f, 2.0f, 3.0f };
        float       p_b[3];
        float       p_roundtrip[3];

        check_equal(frame_transform_point(R_a2b, t_a_in_b, p_a, p_b), 0);
        TEST_FLOAT_WITHIN(threshold, 8.0f, p_b[0], "Transform point x failed");
        TEST_FLOAT_WITHIN(threshold, 21.0f, p_b[1], "Transform point y failed");
        TEST_FLOAT_WITHIN(threshold, 33.0f, p_b[2], "Transform point z failed");
        check_equal(frame_inverse_transform_point(R_a2b, t_a_in_b, p_b, p_roundtrip), 0);
        for (int i = 0; i < 3; ++i)
        {
            TEST_FLOAT_WITHIN(threshold, p_a[i], p_roundtrip[i],
                              "Inverse transform point failed");
        }
        printf("[x] Point local/global transform round-trip\n");
    }

    {
        const float R_a2b[9] = { 0.0f, 1.0f, 0.0f,
                                -1.0f, 0.0f, 0.0f,
                                 0.0f, 0.0f, 1.0f };
        const float t_a_in_b[3] = { 10.0f, 20.0f, 30.0f };
        const float I[9]        = { 1.0f, 0.0f, 0.0f,
                                    0.0f, 1.0f, 0.0f,
                                    0.0f, 0.0f, 1.0f };
        float R_b2a[9];
        float t_b_in_a[3];
        float R_a2a[9];
        float t_a_in_a[3];

        check_equal(frame_pose_inverse(R_a2b, t_a_in_b, R_b2a, t_b_in_a), 0);
        check_equal(frame_pose_compose(R_a2b, t_a_in_b, R_b2a, t_b_in_a, R_a2a,
                                        t_a_in_a),
                     0);
        for (int i = 0; i < 9; ++i)
        {
            TEST_FLOAT_WITHIN(threshold, I[i], R_a2a[i], "Pose inverse compose R failed");
        }
        for (int i = 0; i < 3; ++i)
        {
            TEST_FLOAT_WITHIN(threshold, 0.0f, t_a_in_a[i],
                              "Pose inverse compose t failed");
        }
        printf("[x] Pose inverse and compose\n");
    }

    {
        const float p_enu[3] = { 1.0f, 2.0f, 3.0f };
        float       p_ned[3];
        float       p_enu_roundtrip[3];

        frame_enu_to_ned_point(p_enu, p_ned);
        TEST_FLOAT_WITHIN(threshold, 2.0f, p_ned[0], "ENU->NED north failed");
        TEST_FLOAT_WITHIN(threshold, 1.0f, p_ned[1], "ENU->NED east failed");
        TEST_FLOAT_WITHIN(threshold, -3.0f, p_ned[2], "ENU->NED down failed");
        frame_ned_to_enu_point(p_ned, p_enu_roundtrip);
        for (int i = 0; i < 3; ++i)
        {
            TEST_FLOAT_WITHIN(threshold, p_enu[i], p_enu_roundtrip[i],
                              "ENU/NED round-trip failed");
        }
        printf("[x] ENU/NED vector conversion\n");
    }

    {
        const float R_yaw90[9] = { 0.0f, 1.0f, 0.0f,
                                  -1.0f, 0.0f, 0.0f,
                                   0.0f, 0.0f, 1.0f };
        const float P_a[9]     = { 1.0f, 0.0f, 0.0f,
                                   0.0f, 4.0f, 0.0f,
                                   0.0f, 0.0f, 9.0f };
        float       P_b[9];

        check_equal(frame_transform_covariance3(R_yaw90, P_a, P_b), 0);
        TEST_FLOAT_WITHIN(threshold, 4.0f, P_b[0], "Covariance x variance failed");
        TEST_FLOAT_WITHIN(threshold, 1.0f, P_b[4], "Covariance y variance failed");
        TEST_FLOAT_WITHIN(threshold, 9.0f, P_b[8], "Covariance z variance failed");
        TEST_FLOAT_WITHIN(threshold, 0.0f, P_b[1], "Covariance xy failed");
        TEST_FLOAT_WITHIN(threshold, 0.0f, P_b[3], "Covariance yx failed");
        printf("[x] Covariance frame transform\n");
    }

    {
        const float p_rh[3] = { 1.0f, 2.0f, 3.0f };
        float       p_lh[3];

        frame_handedness_flip_y_point(p_rh, p_lh);
        TEST_FLOAT_WITHIN(threshold, 1.0f, p_lh[0], "Handedness x failed");
        TEST_FLOAT_WITHIN(threshold, -2.0f, p_lh[1], "Handedness y failed");
        TEST_FLOAT_WITHIN(threshold, 3.0f, p_lh[2], "Handedness z failed");
        printf("[x] Handedness Y-flip conversion\n");
    }
}

static void testsignalfilters(void)
{
    printf("Running signal filter tests...\n");

    {
        kf_signal_lowpass1f filter;
        float               y;

        check_equal(kf_signal_lowpass1f_init(&filter, 0.25f, 0.0f), 0);
        check_equal(kf_signal_lowpass1f_update(&filter, 4.0f, &y), 0);
        TEST_FLOAT_WITHIN(1.0e-06f, 1.0f, y, "Scalar low-pass first update failed");
        check_equal(kf_signal_lowpass1f_update(&filter, 5.0f, &y), 0);
        TEST_FLOAT_WITHIN(1.0e-06f, 2.0f, y, "Scalar low-pass second update failed");
        printf("[x] Scalar low-pass filter\n");
    }
    {
        float       y[2] = { 0.0f, 2.0f };
        const float x[2] = { 4.0f, 6.0f };

        check_equal(kf_signal_lowpass_vector(y, x, 2, 0.5f), 0);
        TEST_FLOAT_WITHIN(1.0e-06f, 2.0f, y[0], "Vector low-pass y[0] failed");
        TEST_FLOAT_WITHIN(1.0e-06f, 4.0f, y[1], "Vector low-pass y[1] failed");
        printf("[x] Vector low-pass filter\n");
    }
    {
        kf_signal_highpass1f filter;
        float                y;

        check_equal(kf_signal_highpass1f_init(&filter, 0.5f, 0.0f, 0.0f), 0);
        check_equal(kf_signal_highpass1f_update(&filter, 4.0f, &y), 0);
        TEST_FLOAT_WITHIN(1.0e-06f, 2.0f, y, "Scalar high-pass first update failed");
        check_equal(kf_signal_highpass1f_update(&filter, 4.0f, &y), 0);
        TEST_FLOAT_WITHIN(1.0e-06f, 1.0f, y, "Scalar high-pass DC decay failed");
        printf("[x] Scalar high-pass filter\n");
    }
    {
        float       y[2]      = { 0.0f, 1.0f };
        float       prev_x[2] = { 0.0f, 2.0f };
        const float x[2]      = { 4.0f, 5.0f };

        check_equal(kf_signal_highpass_vector(y, prev_x, x, 2, 0.5f), 0);
        TEST_FLOAT_WITHIN(1.0e-06f, 2.0f, y[0], "Vector high-pass y[0] failed");
        TEST_FLOAT_WITHIN(1.0e-06f, 2.0f, y[1], "Vector high-pass y[1] failed");
        TEST_FLOAT_WITHIN(1.0e-06f, 4.0f, prev_x[0], "Vector high-pass prev_x[0] failed");
        TEST_FLOAT_WITHIN(1.0e-06f, 5.0f, prev_x[1], "Vector high-pass prev_x[1] failed");
        printf("[x] Vector high-pass filter\n");
    }
    {
        kf_signal_moving_average1f filter;
        float                      window[3];
        float                      y;

        check_equal(kf_signal_moving_average1f_init(&filter, window, 3), 0);
        check_equal(kf_signal_moving_average1f_update(&filter, 3.0f, &y), 0);
        TEST_FLOAT_WITHIN(1.0e-06f, 3.0f, y, "Moving average first update failed");
        check_equal(kf_signal_moving_average1f_update(&filter, 6.0f, &y), 0);
        TEST_FLOAT_WITHIN(1.0e-06f, 4.5f, y, "Moving average partial window failed");
        check_equal(kf_signal_moving_average1f_update(&filter, 9.0f, &y), 0);
        TEST_FLOAT_WITHIN(1.0e-06f, 6.0f, y, "Moving average full window failed");
        check_equal(kf_signal_moving_average1f_update(&filter, 12.0f, &y), 0);
        TEST_FLOAT_WITHIN(1.0e-06f, 9.0f, y, "Moving average rolling window failed");
        printf("[x] Moving average filter\n");
    }
    {
        kf_signal_median1f filter;
        float              window[3];
        float              scratch[3];
        float              y;

        check_equal(kf_signal_median1f_init(&filter, window, scratch, 3), 0);
        check_equal(kf_signal_median1f_update(&filter, 3.0f, &y), 0);
        TEST_FLOAT_WITHIN(1.0e-06f, 3.0f, y, "Median first update failed");
        check_equal(kf_signal_median1f_update(&filter, 1.0f, &y), 0);
        TEST_FLOAT_WITHIN(1.0e-06f, 2.0f, y, "Median even window failed");
        check_equal(kf_signal_median1f_update(&filter, 10.0f, &y), 0);
        TEST_FLOAT_WITHIN(1.0e-06f, 3.0f, y, "Median full window failed");
        check_equal(kf_signal_median1f_update(&filter, 2.0f, &y), 0);
        TEST_FLOAT_WITHIN(1.0e-06f, 2.0f, y, "Median rolling window failed");
        printf("[x] Median filter\n");
    }
    {
        kf_signal_moving_rms1f filter;
        float                  window[3];
        float                  y;

        check_equal(kf_signal_moving_rms1f_init(&filter, window, 3), 0);
        check_equal(kf_signal_moving_rms1f_update(&filter, 3.0f, &y), 0);
        TEST_FLOAT_WITHIN(1.0e-06f, 3.0f, y, "Moving RMS first update failed");
        check_equal(kf_signal_moving_rms1f_update(&filter, 4.0f, &y), 0);
        TEST_FLOAT_WITHIN(1.0e-06f, 3.5355339f, y, "Moving RMS partial window failed");
        check_equal(kf_signal_moving_rms1f_update(&filter, 0.0f, &y), 0);
        TEST_FLOAT_WITHIN(1.0e-06f, 2.8867513f, y, "Moving RMS full window failed");
        check_equal(kf_signal_moving_rms1f_update(&filter, 0.0f, &y), 0);
        TEST_FLOAT_WITHIN(1.0e-06f, 2.3094010f, y, "Moving RMS rolling window failed");
        printf("[x] Moving RMS filter\n");
    }
    {
        kf_signal_trimmed_mean1f filter;
        float                    window[5];
        float                    scratch[5];
        float                    y;

        check_equal(kf_signal_trimmed_mean1f_init(&filter, window, scratch, 5, 1), 0);
        check_equal(kf_signal_trimmed_mean1f_update(&filter, 1.0f, &y), 0);
        check_equal(kf_signal_trimmed_mean1f_update(&filter, 2.0f, &y), 0);
        check_equal(kf_signal_trimmed_mean1f_update(&filter, 100.0f, &y), 0);
        check_equal(kf_signal_trimmed_mean1f_update(&filter, 3.0f, &y), 0);
        check_equal(kf_signal_trimmed_mean1f_update(&filter, 4.0f, &y), 0);
        TEST_FLOAT_WITHIN(1.0e-06f, 3.0f, y, "Trimmed mean failed");
        printf("[x] Trimmed mean filter\n");
    }
    {
        kf_signal_winsorized_mean1f filter;
        float                       window[5];
        float                       scratch[5];
        float                       y;

        check_equal(kf_signal_winsorized_mean1f_init(&filter, window, scratch, 5, 1), 0);
        check_equal(kf_signal_winsorized_mean1f_update(&filter, 1.0f, &y), 0);
        check_equal(kf_signal_winsorized_mean1f_update(&filter, 2.0f, &y), 0);
        check_equal(kf_signal_winsorized_mean1f_update(&filter, 100.0f, &y), 0);
        check_equal(kf_signal_winsorized_mean1f_update(&filter, 3.0f, &y), 0);
        check_equal(kf_signal_winsorized_mean1f_update(&filter, 4.0f, &y), 0);
        TEST_FLOAT_WITHIN(1.0e-06f, 3.0f, y, "Winsorized mean failed");
        printf("[x] Winsorized mean filter\n");
    }
    {
        kf_signal_savgol1f filter;
        const float        coeffs[5] = { -3.0f / 35.0f, 12.0f / 35.0f, 17.0f / 35.0f,
                                         12.0f / 35.0f, -3.0f / 35.0f };
        float              window[5];
        float              y;

        check_equal(kf_signal_savgol1f_init(&filter, coeffs, window, 5), 0);
        check_equal(kf_signal_savgol1f_update(&filter, 1.0f, &y), 0);
        check_equal(kf_signal_savgol1f_update(&filter, 2.0f, &y), 0);
        check_equal(kf_signal_savgol1f_update(&filter, 3.0f, &y), 0);
        check_equal(kf_signal_savgol1f_update(&filter, 4.0f, &y), 0);
        check_equal(kf_signal_savgol1f_update(&filter, 5.0f, &y), 0);
        TEST_FLOAT_WITHIN(1.0e-06f, 3.0f, y, "Savitzky-Golay smoother failed");
        printf("[x] Savitzky-Golay smoother\n");
    }
    {
        float y;

        check_equal(kf_signal_clamp1f(10.0f, 0.0f, 5.0f, &y), 0);
        TEST_FLOAT_WITHIN(1.0e-06f, 5.0f, y, "Clamp upper limit failed");
        check_equal(kf_signal_deadband1f(0.3f, 0.5f, &y), 0);
        TEST_FLOAT_WITHIN(1.0e-06f, 0.0f, y, "Deadband zero failed");
        check_equal(kf_signal_deadband1f(-2.0f, 0.5f, &y), 0);
        TEST_FLOAT_WITHIN(1.0e-06f, -1.5f, y, "Deadband negative failed");
        check_equal(kf_signal_slew_rate_limit1f(0.0f, 10.0f, 3.0f, &y), 0);
        TEST_FLOAT_WITHIN(1.0e-06f, 3.0f, y, "Slew rate positive limit failed");
        check_equal(kf_signal_slew_rate_limit1f(5.0f, 1.0f, 2.0f, &y), 0);
        TEST_FLOAT_WITHIN(1.0e-06f, 3.0f, y, "Slew rate negative limit failed");
        printf("[x] Clamp, deadband, and slew-rate filters\n");
    }
    {
        kf_signal_hampel1f filter;
        float              window[3];
        float              scratch[3];
        float              y;

        check_equal(kf_signal_hampel1f_init(&filter, window, scratch, 3), 0);
        check_equal(kf_signal_hampel1f_update(&filter, 10.0f, 3.0f, &y), 1);
        check_equal(kf_signal_hampel1f_update(&filter, 11.0f, 3.0f, &y), 1);
        check_equal(kf_signal_hampel1f_update(&filter, 9.0f, 3.0f, &y), 1);
        check_equal(kf_signal_hampel1f_update(&filter, 100.0f, 3.0f, &y), 0);
        TEST_FLOAT_WITHIN(1.0e-06f, 10.0f, y, "Hampel replacement failed");
        printf("[x] Hampel outlier filter\n");
    }
    {
        kf_signal_biquad_coeffs coeffs;
        kf_signal_biquad1f      filter;
        float                   y;
        float                   dc_num;
        float                   dc_den;

        check_equal(kf_signal_biquad_lowpass(100.0f, 10.0f, 0.70710678f, &coeffs), 0);
        dc_num = coeffs.b0 + coeffs.b1 + coeffs.b2;
        dc_den = 1.0f + coeffs.a1 + coeffs.a2;
        TEST_FLOAT_WITHIN(1.0e-05f, dc_den, dc_num, "Biquad low-pass DC gain failed");

        coeffs.b0 = 1.0f;
        coeffs.b1 = 0.0f;
        coeffs.b2 = 0.0f;
        coeffs.a1 = 0.0f;
        coeffs.a2 = 0.0f;
        check_equal(kf_signal_biquad1f_init(&filter, &coeffs), 0);
        check_equal(kf_signal_biquad1f_update(&filter, 2.5f, &y), 0);
        TEST_FLOAT_WITHIN(1.0e-06f, 2.5f, y, "Biquad passthrough failed");
        check_equal(kf_signal_biquad_notch(100.0f, 50.0f, 0.70710678f, &coeffs), -1);
        printf("[x] Biquad IIR filter\n");
    }
    {
        kf_signal_biquad1f filter;
        float              y;

        check_equal(kf_signal_biquad1f_init_lowpass(&filter, 100.0f, 10.0f, 0.70710678f), 0);
        check_equal(kf_signal_biquad1f_update(&filter, 1.0f, &y), 0);
        check(y > 0.0f);
        check_equal(kf_signal_biquad1f_init_notch(&filter, 100.0f, 50.0f, 0.70710678f), -1);
        printf("[x] Biquad convenience initializers\n");
    }
    {
        kf_signal_fir1f filter;
        const float     coeffs[3] = { 0.5f, 0.3f, 0.2f };
        float           state[3];
        float           y;

        check_equal(kf_signal_fir1f_init(&filter, coeffs, state, 3), 0);
        check_equal(kf_signal_fir1f_update(&filter, 10.0f, &y), 0);
        TEST_FLOAT_WITHIN(1.0e-06f, 5.0f, y, "FIR first update failed");
        check_equal(kf_signal_fir1f_update(&filter, 20.0f, &y), 0);
        TEST_FLOAT_WITHIN(1.0e-06f, 13.0f, y, "FIR second update failed");
        check_equal(kf_signal_fir1f_update(&filter, 30.0f, &y), 0);
        TEST_FLOAT_WITHIN(1.0e-06f, 23.0f, y, "FIR full window failed");
        printf("[x] FIR filter\n");
    }
    {
        kf_signal_iir1f filter;
        const float     b[1] = { 0.5f };
        const float     a[2] = { 1.0f, -0.5f };
        float           x_state[1];
        float           y_state[1];
        float           y;

        check_equal(kf_signal_iir1f_init(&filter, b, 1, a, 2, x_state, y_state), 0);
        check_equal(kf_signal_iir1f_update(&filter, 2.0f, &y), 0);
        TEST_FLOAT_WITHIN(1.0e-06f, 1.0f, y, "IIR first update failed");
        check_equal(kf_signal_iir1f_update(&filter, 2.0f, &y), 0);
        TEST_FLOAT_WITHIN(1.0e-06f, 1.5f, y, "IIR second update failed");
        printf("[x] Generic IIR filter\n");
    }
    {
        kf_signal_alpha_beta1f filter;
        float                  position;
        float                  velocity;

        check_equal(kf_signal_alpha_beta1f_init(&filter, 0.5f, 0.25f, 1.0f, 0.0f, 1.0f), 0);
        check_equal(kf_signal_alpha_beta1f_update(&filter, 2.0f, &position, &velocity), 0);
        TEST_FLOAT_WITHIN(1.0e-06f, 1.5f, position, "Alpha-beta position failed");
        TEST_FLOAT_WITHIN(1.0e-06f, 1.25f, velocity, "Alpha-beta velocity failed");
        printf("[x] Alpha-beta tracking filter\n");
    }
    {
        kf_signal_alpha_beta_gamma1f filter;
        float                        position;
        float                        velocity;
        float                        acceleration;

        check_equal(kf_signal_alpha_beta_gamma1f_init(&filter, 0.5f, 0.25f, 0.1f, 1.0f, 0.0f,
                                                       1.0f, 0.0f),
                     0);
        check_equal(kf_signal_alpha_beta_gamma1f_update(&filter, 2.0f, &position, &velocity,
                                                         &acceleration),
                     0);
        TEST_FLOAT_WITHIN(1.0e-06f, 1.5f, position, "Alpha-beta-gamma position failed");
        TEST_FLOAT_WITHIN(1.0e-06f, 1.25f, velocity, "Alpha-beta-gamma velocity failed");
        TEST_FLOAT_WITHIN(1.0e-06f, 0.2f, acceleration, "Alpha-beta-gamma acceleration failed");
        printf("[x] Alpha-beta-gamma tracking filter\n");
    }
    {
        kf_signal_complementary1f filter;
        float                     y;

        check_equal(kf_signal_complementary1f_init(&filter, 0.9f, 1.0f, 0.0f), 0);
        check_equal(kf_signal_complementary1f_update(&filter, 10.0f, 0.0f, &y), 0);
        TEST_FLOAT_WITHIN(1.0e-06f, 9.0f, y, "Complementary filter failed");
        printf("[x] Complementary filter\n");
    }
    {
        kf_signal_hysteresis1f filter;
        int                    state;

        check_equal(kf_signal_hysteresis1f_init(&filter, 2.0f, 5.0f, 0), 0);
        check_equal(kf_signal_hysteresis1f_update(&filter, 4.0f, &state), 0);
        check_equal(state, 0);
        check_equal(kf_signal_hysteresis1f_update(&filter, 6.0f, &state), 0);
        check_equal(state, 1);
        check_equal(kf_signal_hysteresis1f_update(&filter, 3.0f, &state), 0);
        check_equal(state, 1);
        check_equal(kf_signal_hysteresis1f_update(&filter, 1.0f, &state), 0);
        check_equal(state, 0);
        printf("[x] Hysteresis filter\n");
    }
    {
        kf_signal_debounce filter;
        int                state;

        check_equal(kf_signal_debounce_init(&filter, 2, 0), 0);
        check_equal(kf_signal_debounce_update(&filter, 1, &state), 0);
        check_equal(state, 0);
        check_equal(kf_signal_debounce_update(&filter, 1, &state), 0);
        check_equal(state, 1);
        check_equal(kf_signal_debounce_update(&filter, 0, &state), 0);
        check_equal(state, 1);
        check_equal(kf_signal_debounce_update(&filter, 0, &state), 0);
        check_equal(state, 0);
        printf("[x] Debounce filter\n");
    }
    {
        kf_signal_rate_limiter1f filter;
        float                    y;

        check_equal(kf_signal_rate_limiter1f_init(&filter, 2.0f, 0.5f, 0.0f), 0);
        check_equal(kf_signal_rate_limiter1f_update(&filter, 10.0f, &y), 0);
        TEST_FLOAT_WITHIN(1.0e-06f, 1.0f, y, "Rate limiter failed");
        printf("[x] Stateful rate limiter\n");
    }
    {
        kf_signal_accel_limiter1f filter;
        float                     y;
        float                     rate;

        check_equal(kf_signal_accel_limiter1f_init(&filter, 10.0f, 2.0f, 1.0f, 0.0f, 0.0f),
                     0);
        check_equal(kf_signal_accel_limiter1f_update(&filter, 10.0f, &y, &rate), 0);
        TEST_FLOAT_WITHIN(1.0e-06f, 2.0f, y, "Accel limiter first value failed");
        TEST_FLOAT_WITHIN(1.0e-06f, 2.0f, rate, "Accel limiter first rate failed");
        check_equal(kf_signal_accel_limiter1f_update(&filter, 10.0f, &y, &rate), 0);
        TEST_FLOAT_WITHIN(1.0e-06f, 6.0f, y, "Accel limiter second value failed");
        TEST_FLOAT_WITHIN(1.0e-06f, 4.0f, rate, "Accel limiter second rate failed");
        printf("[x] Stateful acceleration limiter\n");
    }
    {
        kf_signal_one_euro1f filter;
        float                y;

        check_equal(kf_signal_one_euro1f_init(&filter, 1.0f, 0.0f, 1.0f, 1.0f, 0.0f), 0);
        check_equal(kf_signal_one_euro1f_update(&filter, 1.0f, &y), 0);
        TEST_FLOAT_WITHIN(1.0e-06f, 0.8626974f, y, "One Euro filter failed");
        printf("[x] One Euro adaptive filter\n");
    }
    {
        kf_signal_adaptive_ema1f filter;
        float                    y;
        float                    alpha;

        check_equal(kf_signal_adaptive_ema1f_init(&filter, 0.1f, 0.9f, 1.0f, 0.0f), 0);
        check_equal(kf_signal_adaptive_ema1f_update(&filter, 1.0f, &y, &alpha), 0);
        TEST_FLOAT_WITHIN(1.0e-06f, 0.5f, alpha, "Adaptive EMA alpha failed");
        TEST_FLOAT_WITHIN(1.0e-06f, 0.5f, y, "Adaptive EMA output failed");
        printf("[x] Adaptive EMA filter\n");
    }
    {
        kf_signal_running_stats1f stats;
        float                     mean;
        float                     var_pop;
        float                     var_sample;
        float                     std_pop;
        float                     std_sample;

        check_equal(kf_signal_running_stats1f_init(&stats), 0);
        check_equal(kf_signal_running_stats1f_update(&stats, 1.0f), 0);
        check_equal(kf_signal_running_stats1f_update(&stats, 2.0f), 0);
        check_equal(kf_signal_running_stats1f_update(&stats, 3.0f), 0);
        check_equal(kf_signal_running_stats1f_update(&stats, 4.0f), 0);
        check_equal(kf_signal_running_stats1f_get(&stats, &mean, &var_pop, &var_sample, &std_pop,
                                                   &std_sample),
                     0);
        TEST_FLOAT_WITHIN(1.0e-06f, 2.5f, mean, "Running stats mean failed");
        TEST_FLOAT_WITHIN(1.0e-06f, 1.25f, var_pop, "Running stats population variance failed");
        TEST_FLOAT_WITHIN(1.0e-06f, 1.6666667f, var_sample, "Running stats sample variance failed");
        TEST_FLOAT_WITHIN(1.0e-06f, 1.1180340f, std_pop, "Running stats population stddev failed");
        printf("[x] Running variance/stddev estimator\n");
    }
    {
        kf_signal_biquad_coeffs coeffs = { 1.0f, 0.0f, 0.0f, 0.0f, 0.0f };
        kf_signal_biquad1f      sections[2];
        kf_signal_sos1f         sos;
        float                   y;

        check_equal(kf_signal_biquad1f_init(&sections[0], &coeffs), 0);
        check_equal(kf_signal_biquad1f_init(&sections[1], &coeffs), 0);
        check_equal(kf_signal_sos1f_init(&sos, sections, 2), 0);
        check_equal(kf_signal_sos1f_update(&sos, 2.5f, &y), 0);
        TEST_FLOAT_WITHIN(1.0e-06f, 2.5f, y, "SOS filter passthrough failed");
        printf("[x] Cascaded biquad/SOS filter\n");
    }
    {
        kf_signal_dc_blocker1f filter;
        float                  y;

        check_equal(kf_signal_dc_blocker1f_init(&filter, 0.5f, 0.0f, 0.0f), 0);
        check_equal(kf_signal_dc_blocker1f_update(&filter, 1.0f, &y), 0);
        TEST_FLOAT_WITHIN(1.0e-06f, 1.0f, y, "DC blocker first update failed");
        check_equal(kf_signal_dc_blocker1f_update(&filter, 1.0f, &y), 0);
        TEST_FLOAT_WITHIN(1.0e-06f, 0.5f, y, "DC blocker decay failed");
        printf("[x] DC blocker filter\n");
    }
    {
        kf_signal_quantile1f filter;
        float                window[5];
        float                scratch[5];
        float                y;

        check_equal(kf_signal_quantile1f_init(&filter, window, scratch, 5), 0);
        check_equal(kf_signal_quantile1f_update(&filter, 1.0f, 0.75f, &y), 0);
        check_equal(kf_signal_quantile1f_update(&filter, 2.0f, 0.75f, &y), 0);
        check_equal(kf_signal_quantile1f_update(&filter, 3.0f, 0.75f, &y), 0);
        check_equal(kf_signal_quantile1f_update(&filter, 4.0f, 0.75f, &y), 0);
        check_equal(kf_signal_quantile1f_update(&filter, 5.0f, 0.75f, &y), 0);
        TEST_FLOAT_WITHIN(1.0e-06f, 4.0f, y, "Quantile filter failed");
        printf("[x] Running quantile filter\n");
    }
    {
        const float values[5] = { 1.0f, 2.0f, 3.0f, 4.0f, 100.0f };
        float       scratch[5];
        float       median;
        float       sigma;

        check_equal(kf_signal_mad_noise_sigma(values, scratch, 5, &median, &sigma), 0);
        TEST_FLOAT_WITHIN(1.0e-06f, 3.0f, median, "MAD median failed");
        TEST_FLOAT_WITHIN(1.0e-06f, 1.4826f, sigma, "MAD sigma failed");
        printf("[x] MAD noise estimator\n");
    }
    {
        const float values[5] = { 1.0f, 2.0f, 3.0f, 4.0f, 5.0f };
        float       scratch[5];
        float       zscore;
        float       q1;
        float       q3;

        check_equal(kf_signal_zscore_gate(3.0f, 0.0f, 1.0f, 2.0f, &zscore), 0);
        TEST_FLOAT_WITHIN(1.0e-06f, 3.0f, zscore, "Z-score value failed");
        check_equal(kf_signal_zscore_gate(1.0f, 0.0f, 1.0f, 2.0f, &zscore), 1);
        check_equal(kf_signal_iqr_gate(5.0f, values, scratch, 5, 1.5f, &q1, &q3), 1);
        TEST_FLOAT_WITHIN(1.0e-06f, 2.0f, q1, "IQR q1 failed");
        TEST_FLOAT_WITHIN(1.0e-06f, 4.0f, q3, "IQR q3 failed");
        check_equal(kf_signal_iqr_gate(10.0f, values, scratch, 5, 1.5f, NULL, NULL), 0);
        printf("[x] Z-score and IQR gates\n");
    }
    {
        float weight;

        check_equal(kf_signal_huber_weight(4.0f, 2.0f, &weight), 0);
        TEST_FLOAT_WITHIN(1.0e-06f, 0.5f, weight, "Huber weight failed");
        check_equal(kf_signal_tukey_weight(1.0f, 2.0f, &weight), 0);
        TEST_FLOAT_WITHIN(1.0e-06f, 0.5625f, weight, "Tukey weight failed");
        check_equal(kf_signal_tukey_weight(3.0f, 2.0f, &weight), 0);
        TEST_FLOAT_WITHIN(1.0e-06f, 0.0f, weight, "Tukey rejection weight failed");
        printf("[x] Huber and Tukey robust weights\n");
    }
    {
        int state;

        check_equal(kf_signal_schmitt_trigger(6.0f, 2.0f, 5.0f, 0, &state), 0);
        check_equal(state, 1);
        check_equal(kf_signal_schmitt_trigger(3.0f, 2.0f, 5.0f, state, &state), 0);
        check_equal(state, 1);
        check_equal(kf_signal_schmitt_trigger(1.0f, 2.0f, 5.0f, state, &state), 0);
        check_equal(state, 0);
        printf("[x] Schmitt trigger helper\n");
    }
    {
        kf_signal_edge1f filter;
        int              rising;
        int              falling;

        check_equal(kf_signal_edge1f_init(&filter, 0), 0);
        check_equal(kf_signal_edge1f_update(&filter, 1, &rising, &falling), 0);
        check_equal(rising, 1);
        check_equal(falling, 0);
        check_equal(kf_signal_edge1f_update(&filter, 0, &rising, &falling), 0);
        check_equal(rising, 0);
        check_equal(falling, 1);
        printf("[x] Edge detector\n");
    }
    {
        kf_signal_sample_hold1f filter;
        float                   y;

        check_equal(kf_signal_sample_hold1f_init(&filter, 1.0f, 2), 0);
        check_equal(kf_signal_sample_hold1f_update(&filter, 5.0f, 1, &y), 1);
        TEST_FLOAT_WITHIN(1.0e-06f, 5.0f, y, "Sample hold valid update failed");
        check_equal(kf_signal_sample_hold1f_update(&filter, 10.0f, 0, &y), 0);
        TEST_FLOAT_WITHIN(1.0e-06f, 5.0f, y, "Sample hold dropout failed");
        printf("[x] Sample-and-hold dropout filter\n");
    }
    {
        kf_signal_moving_minmax1f filter;
        float                     window[3];
        float                     scratch[3];
        float                     min_value;
        float                     max_value;

        check_equal(kf_signal_moving_minmax1f_init(&filter, window, scratch, 3), 0);
        check_equal(kf_signal_moving_minmax1f_update(&filter, 3.0f, &min_value, &max_value), 0);
        check_equal(kf_signal_moving_minmax1f_update(&filter, 1.0f, &min_value, &max_value), 0);
        check_equal(kf_signal_moving_minmax1f_update(&filter, 5.0f, &min_value, &max_value), 0);
        TEST_FLOAT_WITHIN(1.0e-06f, 1.0f, min_value, "Moving min failed");
        TEST_FLOAT_WITHIN(1.0e-06f, 5.0f, max_value, "Moving max failed");
        check_equal(kf_signal_moving_minmax1f_update(&filter, 4.0f, &min_value, &max_value), 0);
        TEST_FLOAT_WITHIN(1.0e-06f, 1.0f, min_value, "Moving min rolling failed");
        TEST_FLOAT_WITHIN(1.0e-06f, 5.0f, max_value, "Moving max rolling failed");
        printf("[x] Moving min/max filter\n");
    }
    {
        kf_signal_ew_stats1f stats;
        float                mean;
        float                variance;

        check_equal(kf_signal_ew_stats1f_init(&stats, 0.5f, 0.0f, 0.0f), 0);
        check_equal(kf_signal_ew_stats1f_update(&stats, 2.0f, &mean, &variance), 0);
        TEST_FLOAT_WITHIN(1.0e-06f, 1.0f, mean, "EW stats mean first failed");
        TEST_FLOAT_WITHIN(1.0e-06f, 1.0f, variance, "EW stats variance first failed");
        check_equal(kf_signal_ew_stats1f_update(&stats, 2.0f, &mean, &variance), 0);
        TEST_FLOAT_WITHIN(1.0e-06f, 1.5f, mean, "EW stats mean second failed");
        TEST_FLOAT_WITHIN(1.0e-06f, 0.75f, variance, "EW stats variance second failed");
        printf("[x] Exponentially weighted stats\n");
    }
    {
        kf_signal_peak_hold1f filter;
        float                 peak;

        check_equal(kf_signal_peak_hold1f_init(&filter, 0.0f, 1.0f), 0);
        check_equal(kf_signal_peak_hold1f_update(&filter, 5.0f, &peak), 0);
        TEST_FLOAT_WITHIN(1.0e-06f, 5.0f, peak, "Peak hold capture failed");
        check_equal(kf_signal_peak_hold1f_update(&filter, 2.0f, &peak), 0);
        TEST_FLOAT_WITHIN(1.0e-06f, 4.0f, peak, "Peak hold decay failed");
        printf("[x] Peak hold filter\n");
    }
    {
        kf_signal_majority1f filter;
        int                  window[3];
        int                  output;

        check_equal(kf_signal_majority1f_init(&filter, window, 3, 0), 0);
        check_equal(kf_signal_majority1f_update(&filter, 1, &output), 0);
        check_equal(output, 0);
        check_equal(kf_signal_majority1f_update(&filter, 1, &output), 0);
        check_equal(output, 1);
        printf("[x] Majority vote filter\n");
    }
    {
        kf_signal_jerk_limiter1f filter;
        float                    y;
        float                    rate;
        float                    accel;

        check_equal(kf_signal_jerk_limiter1f_init(&filter, 10.0f, 10.0f, 2.0f, 1.0f, 0.0f,
                                                   0.0f, 0.0f),
                     0);
        check_equal(kf_signal_jerk_limiter1f_update(&filter, 10.0f, &y, &rate, &accel), 0);
        TEST_FLOAT_WITHIN(1.0e-06f, 2.0f, y, "Jerk limiter first value failed");
        TEST_FLOAT_WITHIN(1.0e-06f, 2.0f, rate, "Jerk limiter first rate failed");
        TEST_FLOAT_WITHIN(1.0e-06f, 2.0f, accel, "Jerk limiter first accel failed");
        printf("[x] Jerk limiter\n");
    }
    {
        kf_signal_dropout_decay1f filter;
        float                     y;

        check_equal(kf_signal_dropout_decay1f_init(&filter, 10.0f, 0.0f, 0.25f), 0);
        check_equal(kf_signal_dropout_decay1f_update(&filter, 8.0f, 1, &y), 1);
        TEST_FLOAT_WITHIN(1.0e-06f, 8.0f, y, "Dropout decay valid update failed");
        check_equal(kf_signal_dropout_decay1f_update(&filter, 0.0f, 0, &y), 0);
        TEST_FLOAT_WITHIN(1.0e-06f, 6.0f, y, "Dropout decay failed");
        printf("[x] Dropout decay filter\n");
    }
    {
        const float residual[2]   = { 2.0f, 0.0f };
        const float covariance[4] = { 4.0f, 0.0f, 0.0f, 1.0f };
        float       covariance_work[4];
        float       residual_work[2];
        float       distance_sq;

        check_equal(kf_signal_mahalanobis_distance_sq(residual, covariance, covariance_work,
                                                       residual_work, 2, &distance_sq),
                     0);
        TEST_FLOAT_WITHIN(1.0e-06f, 1.0f, distance_sq, "Mahalanobis distance failed");
        check_equal(kf_signal_mahalanobis_gate(residual, covariance, covariance_work,
                                                residual_work, 2, 1.1f, &distance_sq),
                     1);
        check_equal(kf_signal_mahalanobis_gate(residual, covariance, covariance_work,
                                                residual_work, 2, 0.5f, &distance_sq),
                     0);
        printf("[x] Mahalanobis distance gate\n");
    }
    {
        float alpha;

        check_equal(kf_signal_lowpass_alpha(1.0f, 1.0f, &alpha), 0);
        TEST_FLOAT_WITHIN(1.0e-06f, 0.8626974f, alpha, "Low-pass alpha calculation failed");
        check_equal(kf_signal_highpass_alpha(1.0f, 1.0f, &alpha), 0);
        TEST_FLOAT_WITHIN(1.0e-06f, 0.1373026f, alpha, "High-pass alpha calculation failed");
        printf("[x] Cutoff to alpha helpers\n");
    }
    {
        const float x[2] = { 3.0f, 4.0f };
        float       norm;

        check_equal(kf_signal_euclidean_norm(x, 2, &norm), 0);
        TEST_FLOAT_WITHIN(1.0e-06f, 5.0f, norm, "Euclidean norm failed");
        printf("[x] Euclidean norm\n");
    }
    {
        const float a[2] = { 1.0f, 2.0f };
        const float b[2] = { 4.0f, 6.0f };
        float       distance;

        check_equal(kf_signal_euclidean_distance(a, b, 2, &distance), 0);
        TEST_FLOAT_WITHIN(1.0e-06f, 5.0f, distance, "Euclidean distance failed");
        printf("[x] Euclidean distance\n");
    }
    {
        float       state[2]  = { 0.0f, 0.0f };
        const float sample[2] = { 3.0f, 4.0f };
        float       distance;
        int         result;

        result = kf_signal_euclidean_filter(state, sample, 2, 6.0f, 0.5f, &distance);
        check_equal(result, 1);
        TEST_FLOAT_WITHIN(1.0e-06f, 5.0f, distance, "Euclidean filter distance failed");
        TEST_FLOAT_WITHIN(1.0e-06f, 1.5f, state[0], "Euclidean filter state[0] failed");
        TEST_FLOAT_WITHIN(1.0e-06f, 2.0f, state[1], "Euclidean filter state[1] failed");

        {
            const float outlier[2] = { 10.0f, 0.0f };

            result = kf_signal_euclidean_filter(state, outlier, 2, 2.0f, 1.0f, &distance);
        }
        check_equal(result, 0);
        TEST_FLOAT_WITHIN(1.0e-06f, 1.5f, state[0], "Euclidean filter rejected state[0] changed");
        TEST_FLOAT_WITHIN(1.0e-06f, 2.0f, state[1], "Euclidean filter rejected state[1] changed");
        printf("[x] Euclidean distance-gated filter\n");
    }
    {
        kf_signal_lowpass1f filter = { 0 };
        float               y;

        check_equal(kf_signal_lowpass1f_init(&filter, 1.5f, 0.0f), -1);
        check_equal(kf_signal_lowpass1f_update(&filter, 1.0f, &y), -1);
        check_equal(kf_signal_lowpass_vector(NULL, &y, 1, 0.5f), -1);
        printf("[x] Signal filter invalid input checks\n");
    }
}

#ifdef APRILTAG_HAVE_MINIBLAS
static void testapriltagadapter(void)
{
    double              R_data[9] = { 0.0, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0 };
    double              R_identity_data[9] = { 1.0, 0.0, 0.0, 0.0, 1.0,
                                                0.0, 0.0, 0.0, 1.0 };
    double              t_data[3] = { 1.0, 2.0, 2.0 };
    matd_t              R         = { 3, 3, R_data };
    matd_t              R_identity = { 3, 3, R_identity_data };
    matd_t              t         = { 3, 1, t_data };
    apriltag_pose_t     pose      = { &R, &t };
    apriltag_pose_t     identity_pose = { &R_identity, &t };
    apriltag_pose_miniblas_t converted;
    const apriltag_map_entry_t tag_map = { 7,
                                           0.16f,
                                           { 1.0f, 0.0f, 0.0f, 0.0f, 1.0f,
                                             0.0f, 0.0f, 0.0f, 1.0f },
                                           { 0.0f, 0.0f, 0.0f } };
    apriltag_map_observation_t map_observation;
    float               position[3];
    float               q[4];

    printf("Running AprilTag miniblas adapter tests...\n");

    check_equal(apriltag_pose_to_miniblas(&pose, &converted), 0);

    TEST_FLOAT_WITHIN(1.0e-06f, 0.0f, MAT_ELEM(converted.R_tag2camera, 0, 0, 3, 3),
                      "AprilTag R layout failed");
    TEST_FLOAT_WITHIN(1.0e-06f, 1.0f, MAT_ELEM(converted.R_tag2camera, 1, 0, 3, 3),
                      "AprilTag R layout failed");
    TEST_FLOAT_WITHIN(1.0e-06f, -1.0f, MAT_ELEM(converted.R_tag2camera, 0, 1, 3, 3),
                      "AprilTag R layout failed");
    TEST_FLOAT_WITHIN(1.0e-06f, 1.0f, converted.rotation_det,
                      "AprilTag rotation determinant failed");
    TEST_FLOAT_WITHIN(1.0e-06f, 0.0f, converted.rotation_orthogonality_error,
                      "AprilTag rotation orthogonality failed");
    TEST_FLOAT_WITHIN(1.0e-06f, 3.0f, converted.translation_norm,
                      "AprilTag translation norm failed");
    TEST_FLOAT_WITHIN(1.0e-06f, 1.0f, converted.t_camera[0],
                      "AprilTag translation conversion failed");
    TEST_FLOAT_WITHIN(1.0e-06f, 2.0f, converted.t_camera[1],
                      "AprilTag translation conversion failed");
    TEST_FLOAT_WITHIN(1.0e-06f, 2.0f, converted.t_camera[2],
                      "AprilTag translation conversion failed");
    TEST_FLOAT_WITHIN(1.0e-05f, 0.70710677f, converted.q_tag2camera[0],
                      "AprilTag quaternion w failed");
    TEST_FLOAT_WITHIN(1.0e-05f, 0.70710677f, converted.q_tag2camera[3],
                      "AprilTag quaternion z failed");

    check_equal(apriltag_pose_to_fusion_measurement(&pose, position, q), 0);
    TEST_FLOAT_WITHIN(1.0e-06f, converted.t_camera[0], position[0],
                      "AprilTag fusion position failed");
    TEST_FLOAT_WITHIN(1.0e-06f, converted.q_tag2camera[3], q[3],
                      "AprilTag fusion quaternion failed");
    check_equal(apriltag_pose_to_miniblas(NULL, &converted), -1);

    check_equal(apriltag_pose_to_map_observation(&identity_pose, &tag_map, &map_observation),
                 0);
    check_equal(map_observation.tag_id, 7);
    TEST_FLOAT_WITHIN(1.0e-06f, -1.0f, map_observation.t_camera_map[0],
                      "AprilTag map camera x failed");
    TEST_FLOAT_WITHIN(1.0e-06f, -2.0f, map_observation.t_camera_map[1],
                      "AprilTag map camera y failed");
    TEST_FLOAT_WITHIN(1.0e-06f, -2.0f, map_observation.t_camera_map[2],
                      "AprilTag map camera z failed");
    TEST_FLOAT_WITHIN(1.0e-06f, 1.0f, map_observation.q_camera2map[0],
                      "AprilTag map quaternion failed");

    printf("[x] AprilTag pose adapter using miniblas/linalg\n");
}
#endif

static int ekf_test_transition(float* x_pred, float* Phi, const float* x, int n, void* user)
{
    const float dt = *(const float*)user;
    check_equal(n, 2);

    x_pred[0] = x[0] + dt * x[1] * x[1];
    x_pred[1] = x[1];

    MAT_ELEM(Phi, 0, 0, 2, 2) = 1.0f;
    MAT_ELEM(Phi, 1, 0, 2, 2) = 0.0f;
    MAT_ELEM(Phi, 0, 1, 2, 2) = 2.0f * dt * x[1];
    MAT_ELEM(Phi, 1, 1, 2, 2) = 1.0f;

    return 0;
}

static int ekf_test_measurement(float* z_pred, float* Ht, const float* x, int n, int m, void* user)
{
    (void)user;
    check_equal(n, 1);
    check_equal(m, 1);

    z_pred[0] = x[0] * x[0];
    Ht[0]     = 2.0f * x[0];

    return 0;
}

static int ekf_benchmark_measurement(float* z_pred, float* Ht, const float* x, int n, int m,
                                     void* user)
{
    (void)user;

    memset(Ht, 0, sizeof(Ht[0]) * n * m);
    for (int i = 0; i < m; ++i)
    {
        z_pred[i]                = x[i] * x[i];
        MAT_ELEM(Ht, i, i, n, m) = 2.0f * x[i];
    }

    return 0;
}

static void testekf(void)
{
    {
        float       x[2] = { 1.0f, 2.0f };
        float       P[4] = { 1.0f, 0.0f, 0.0f, 1.0f };
        const float dt   = 0.1f;

        check_equal(
            kalman_ekf_takasu_predict(x, P, ekf_test_transition, NULL, NULL, 2, 0, (void*)&dt), 0);

        TEST_FLOAT_WITHIN(1.0e-06f, 1.4f, x[0], "EKF Takasu prediction x[0] failed");
        TEST_FLOAT_WITHIN(1.0e-06f, 2.0f, x[1], "EKF Takasu prediction x[1] failed");
        TEST_FLOAT_WITHIN(1.0e-06f, 1.16f, MAT_ELEM(P, 0, 0, 2, 2),
                          "EKF Takasu prediction P00 failed");
        TEST_FLOAT_WITHIN(1.0e-06f, 0.4f, MAT_ELEM(P, 0, 1, 2, 2),
                          "EKF Takasu prediction P01 failed");
        TEST_FLOAT_WITHIN(1.0e-06f, 0.4f, MAT_ELEM(P, 1, 0, 2, 2),
                          "EKF Takasu prediction P10 failed");
        TEST_FLOAT_WITHIN(1.0e-06f, 1.0f, MAT_ELEM(P, 1, 1, 2, 2),
                          "EKF Takasu prediction P11 failed");
    }
    {
        float       x[2] = { 1.0f, 2.0f };
        float       U[4];
        float       d[2];
        const float P[4] = { 1.0f, 0.0f, 0.0f, 1.0f };
        const float dt   = 0.1f;

        check_equal(udu(P, U, d, 2), 0);
        check_equal(
            kalman_ekf_udu_predict(x, U, d, ekf_test_transition, NULL, NULL, 2, 0, (void*)&dt), 0);

        TEST_FLOAT_WITHIN(1.0e-06f, 1.4f, x[0], "EKF UDU prediction x[0] failed");
        TEST_FLOAT_WITHIN(1.0e-06f, 2.0f, x[1], "EKF UDU prediction x[1] failed");
        TEST_FLOAT_WITHIN(1.0e-06f, 1.0f, d[1], "EKF UDU prediction d1 failed");
        TEST_FLOAT_WITHIN(1.0e-06f, 0.4f, MAT_ELEM(U, 0, 1, 2, 2), "EKF UDU prediction U01 failed");
        TEST_FLOAT_WITHIN(1.0e-06f, 1.0f, MAT_ELEM(U, 0, 0, 2, 2), "EKF UDU prediction U00 failed");
        TEST_FLOAT_WITHIN(1.0e-06f, 1.0f, MAT_ELEM(U, 1, 1, 2, 2), "EKF UDU prediction U11 failed");
        TEST_FLOAT_WITHIN(1.0e-06f, 1.0f, d[0], "EKF UDU prediction d0 failed");
    }
    {
        float       x[1] = { 2.0f };
        float       P[1] = { 0.25f };
        const float z[1] = { 4.4f };
        const float R[1] = { 0.16f };

        check_equal(
            kalman_ekf_takasu_update(x, P, z, R, ekf_test_measurement, 1, 1, 0.0f, NULL, NULL), 0);

        TEST_FLOAT_WITHIN(1.0e-06f, 2.0961538f, x[0], "EKF Takasu update x failed");
        TEST_FLOAT_WITHIN(1.0e-06f, 0.0096154f, P[0], "EKF Takasu update P failed");
    }
    {
        float       x[1] = { 2.0f };
        float       U[1] = { 1.0f };
        float       d[1] = { 0.25f };
        const float z[1] = { 4.4f };
        const float R[1] = { 0.16f };

        check_equal(
            kalman_ekf_udu_update(x, U, d, z, R, ekf_test_measurement, 1, 1, 0.0f, 0, NULL), 0);

        TEST_FLOAT_WITHIN(1.0e-06f, 2.0961538f, x[0], "EKF UDU update x failed");
        TEST_FLOAT_WITHIN(1.0e-06f, 0.0096154f, d[0], "EKF UDU update d failed");
    }
}

static int ukf_square_transition(float* x_pred, const float* x, int n, void* user)
{
    (void)user;
    check_equal(n, 1);

    x_pred[0] = x[0] * x[0];

    return 0;
}

static int ukf_square_measurement(float* z_pred, const float* x, int n, int m, void* user)
{
    (void)user;
    check_equal(n, 1);
    check_equal(m, 1);

    z_pred[0] = x[0] * x[0];

    return 0;
}

static int ukf_benchmark_measurement(float* z_pred, const float* x, int n, int m, void* user)
{
    (void)n;
    (void)user;

    for (int i = 0; i < m; ++i)
    {
        z_pred[i] = x[i] * x[i];
    }

    return 0;
}

static void testukf(void)
{
    const kalman_ukf_params params = { 1.0f, 2.0f, 0.0f };

    {
        float       x[1] = { 2.0f };
        float       P[1] = { 0.25f };
        const float Q[1] = { 0.0f };

        check_equal(kalman_ukf_predict(x, P, Q, ukf_square_transition, 1, &params, NULL), 0);

        TEST_FLOAT_WITHIN(1.0e-06f, 4.25f, x[0], "UKF prediction x failed");
        TEST_FLOAT_WITHIN(1.0e-06f, 4.125f, P[0], "UKF prediction P failed");
    }
    {
        float       x[1] = { 2.0f };
        float       P[1] = { 0.25f };
        const float z[1] = { 4.4f };
        const float R[1] = { 0.16f };

        check_equal(
            kalman_ukf_update(x, P, z, R, ukf_square_measurement, 1, 1, &params, 0.0f, NULL, NULL),
            0);

        TEST_FLOAT_WITHIN(1.0e-05f, 2.0350058f, x[0], "UKF update x failed");
        TEST_FLOAT_WITHIN(1.0e-05f, 0.0166278f, P[0], "UKF update P failed");
    }
}

static void benchmark_core_routines(void)
{
    enum
    {
        n           = 15,
        m           = 3,
        bench_batch = 32
    };
    const float x0[15]   = { 1024.0f, 508.0f, 20.0f };
    const float R[3 * 3] = {
        2.902f, 1.395f, 0.900f, 1.395f, 2.295f, 0.698f, 0.900f, 0.698f, 2.362f
    };
    const float R_diag[3 * 3] = { 2.902f, 0.0f, 0.0f, 0.0f, 2.295f, 0.0f, 0.0f, 0.0f, 2.362f };
    const float dz[3]         = { 0.25f, -0.15f, 0.10f };
    const float z[3]          = { 1024.25f, 507.85f, 20.10f };
    const float ekf_x0[15]    = { 1.2f, -0.8f, 0.5f };
    const float ekf_z[3]      = { 1.54f, 0.58f, 0.31f };
    float       P0[15 * 15];
    float       Ht[15 * 3];

    memset(P0, 0, sizeof(P0));
    memset(Ht, 0, sizeof(Ht));
    for (int i = 0; i < n; ++i)
    {
        MAT_ELEM(P0, i, i, n, n) = 10.0f + (float)i;
    }
    MAT_ELEM(Ht, 0, 0, n, m) = 1.0f;
    MAT_ELEM(Ht, 1, 1, n, m) = 1.0f;
    MAT_ELEM(Ht, 2, 2, n, m) = 1.0f;

    benchmark("kalman_takasu update 15x3", 1000, bench_batch)
    {
        for (int b = 0; b < bench_batch; ++b)
        {
            float x[15];
            float P[15 * 15];

            memcpy(x, x0, sizeof(x));
            memcpy(P, P0, sizeof(P));
            const int result = kalman_takasu(x, P, dz, R, Ht, n, m, 0.0f, NULL);

            benchmark_sink = x[0] + P[0] + (float)result;
        }
    }

    {
        float x[15];
        float P[15 * 15];

        memcpy(x, x0, sizeof(x));
        memcpy(P, P0, sizeof(P));
        check_equal(kalman_takasu(x, P, dz, R, Ht, n, m, 0.0f, NULL), 0);
    }

    benchmark("kalman_udu update 15x3", 1000, bench_batch)
    {
        for (int b = 0; b < bench_batch; ++b)
        {
            float x[15];
            float P[15 * 15];
            float U[15 * 15];
            float d[15];

            memcpy(x, x0, sizeof(x));
            memcpy(P, P0, sizeof(P));
            const int result = udu(P, U, d, n) || kalman_udu(x, U, d, z, R_diag, Ht, n, m, 0.0f, 0);
            benchmark_sink   = x[0] + U[0] + d[0] + (float)result;
        }
    }

    {
        float x[15];
        float P[15 * 15];
        float U[15 * 15];
        float d[15];

        memcpy(x, x0, sizeof(x));
        memcpy(P, P0, sizeof(P));
        check_equal(udu(P, U, d, n), 0);
        check_equal(kalman_udu(x, U, d, z, R_diag, Ht, n, m, 0.0f, 0), 0);
    }

    benchmark("ekf_takasu update 15x3", 1000, bench_batch)
    {
        for (int b = 0; b < bench_batch; ++b)
        {
            float x[15];
            float P[15 * 15];

            memcpy(x, ekf_x0, sizeof(x));
            memcpy(P, P0, sizeof(P));
            const int result = kalman_ekf_takasu_update(
                x, P, ekf_z, R_diag, ekf_benchmark_measurement, n, m, 0.0f, NULL, NULL);

            benchmark_sink = x[0] + P[0] + (float)result;
        }
    }

    {
        float x[15];
        float P[15 * 15];

        memcpy(x, ekf_x0, sizeof(x));
        memcpy(P, P0, sizeof(P));
        check_equal(kalman_ekf_takasu_update(x, P, ekf_z, R_diag, ekf_benchmark_measurement, n, m,
                                              0.0f, NULL, NULL),
                     0);
    }

    benchmark("ekf_udu update 15x3", 1000, bench_batch)
    {
        for (int b = 0; b < bench_batch; ++b)
        {
            float x[15];
            float P[15 * 15];
            float U[15 * 15];
            float d[15];

            memcpy(x, ekf_x0, sizeof(x));
            memcpy(P, P0, sizeof(P));
            const int result = udu(P, U, d, n) || kalman_ekf_udu_update(x, U, d, ekf_z, R_diag,
                                                                        ekf_benchmark_measurement,
                                                                        n, m, 0.0f, 0, NULL);
            benchmark_sink   = x[0] + U[0] + d[0] + (float)result;
        }
    }

    {
        float x[15];
        float P[15 * 15];
        float U[15 * 15];
        float d[15];

        memcpy(x, ekf_x0, sizeof(x));
        memcpy(P, P0, sizeof(P));
        check_equal(udu(P, U, d, n), 0);
        check_equal(kalman_ekf_udu_update(x, U, d, ekf_z, R_diag, ekf_benchmark_measurement, n, m,
                                           0.0f, 0, NULL),
                     0);
    }

    benchmark("ukf update 15x3", 250, bench_batch)
    {
        for (int b = 0; b < bench_batch; ++b)
        {
            float                   x[15];
            float                   P[15 * 15];
            const kalman_ukf_params params = { 1.0f, 2.0f, 0.0f };

            memcpy(x, ekf_x0, sizeof(x));
            memcpy(P, P0, sizeof(P));
            const int result = kalman_ukf_update(x, P, ekf_z, R_diag, ukf_benchmark_measurement, n,
                                                 m, &params, 0.0f, NULL, NULL);

            benchmark_sink = x[0] + P[0] + (float)result;
        }
    }

    {
        float                   x[15];
        float                   P[15 * 15];
        const kalman_ukf_params params = { 1.0f, 2.0f, 0.0f };

        memcpy(x, ekf_x0, sizeof(x));
        memcpy(P, P0, sizeof(P));
        check_equal(kalman_ukf_update(x, P, ekf_z, R_diag, ukf_benchmark_measurement, n, m,
                                       &params, 0.0f, NULL, NULL),
                     0);
    }
}

spec("kfcore")
{
    it("passes linear algebra tests")
    {
        testlinalg();
    }

    it("passes navigation and Kalman filter tests")
    {
        testframetransform();
        testkalmanbounds();
        testnavtoolbox();
    }

    it("passes signal filter tests")
    {
        testsignalfilters();
    }

#ifdef APRILTAG_HAVE_MINIBLAS
    it("passes AprilTag miniblas adapter tests")
    {
        testapriltagadapter();
    }
#endif

    it("passes extended Kalman filter tests")
    {
        testekf();
    }

    it("passes unscented Kalman filter tests")
    {
        testukf();
    }

    it("benchmarks core routines")
    {
        benchmark_core_routines();
    }
}

static void hilbert(float* H, int n)
{
    /*
     *   Hilbert test matrix, lines are _almost_ linear dependent
     *
     *   [  1       1/2     1/3     ...  1/nA      ]
     *   [  1/2     1/3     1/4     ...  1/(n+1)   ]
     *   [  1/3     1/4     1/5     ...  1/(n+2)   ]
     *   [               ...                       ]
     *   [  1/n     1/(n+1) 1/(n+2) ...  1/(2*n-1) ]
     */

    int start = 1;
    for (int i = 0; i < n; i++) /* row */
    {
        int rowstart = start;
        for (int j = 0; j < n; j++) /* col */
        {
            MAT_ELEM(H, i, j, n, n) = 1.0f / rowstart;
            rowstart++;
        }
        start++;
    }
}

static void matprint(const float* R, const int n, const int m, const char* fmt, const char* name)
{
    if (name)
    {
        printf(" %s =\n", name);
        printf("\t");
    }
    for (int i = 0; i < n; i++) /* row */
    {
        for (int j = 0; j < m; j++) /* col */
        {
            printf(fmt, (double)MAT_ELEM(R, i, j, n, m));
            printf(" ");
        }
        printf("\n");
        if (name && i < (n - 1))
        {
            printf("\t");
        }
    }
}

/* @} */
