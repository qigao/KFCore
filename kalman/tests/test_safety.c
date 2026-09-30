#include <math.h>
#include <stdio.h>
#include <string.h>

#include "kalman_ekf.h"
#include "kalman_takasu.h"
#include "kalman_udu.h"
#include "kalman_ukf.h"
#include "linalg.h"

static int failures = 0;

static void expect_int(const char* name, int actual, int expected)
{
    if (actual != expected)
    {
        fprintf(stderr, "%s: expected %d, got %d\n", name, expected, actual);
        ++failures;
    }
}

static void expect_float(const char* name, float actual, float expected)
{
    if (actual != expected)
    {
        fprintf(stderr, "%s: expected %.9g, got %.9g\n",
                name, (double)expected, (double)actual);
        ++failures;
    }
}

static void test_udu_pivots(void)
{
    const float identity[4] = { 1.0f, 0.0f, 0.0f, 1.0f };
    const float negative[1] = { -1.0f };
    const float zero[1] = { 0.0f };
    const float nan_value[1] = { NAN };
    const float final_zero_pivot[4] = { 0.0f, 0.0f, 0.0f, 1.0f };
    float U[4];
    float d[2];

    expect_int("udu identity", udu(identity, U, d, 2), 0);
    expect_int("udu negative pivot", udu(negative, U, d, 1), -1);
    expect_int("udu zero pivot", udu(zero, U, d, 1), -1);
    expect_int("udu NaN pivot", udu(nan_value, U, d, 1), -1);
    expect_int("udu final zero pivot", udu(final_zero_pivot, U, d, 2), -1);
}

static void fill_identity(float* matrix, size_t n)
{
    memset(matrix, 0, sizeof(float) * n * n);
    for (size_t i = 0; i < n; ++i)
        matrix[i + i * n] = 1.0f;
}

static void test_linear_workspace_contract(void)
{
    enum { N = 40, M = 5 };
    float x[N] = { 0.0f };
    float P[N * N];
    float Phi[N * N];
    float G[N * N];
    float Q[N];
    float Ht[N * M];
    float R[M * M];
    float dz[M] = { 0.0f };
    float z[M] = { 0.0f };
    float U[N * N];
    float d[N];
    float predict_workspace[3300];
    float update_workspace[256];
    float udu_workspace[80];
    float udu_predict_workspace[3300];
    size_t required = 0U;

    fill_identity(P, N);
    fill_identity(Phi, N);
    fill_identity(G, N);
    fill_identity(U, N);
    fill_identity(R, M);
    memset(Ht, 0, sizeof(Ht));
    for (size_t i = 0; i < N; ++i)
    {
        x[i] = (float)i;
        Q[i] = 0.01f;
        d[i] = 1.0f;
    }
    for (size_t i = 0; i < M; ++i)
        Ht[i + i * N] = 1.0f;

    expect_int("predict workspace query",
               kalman_predict_workspace_floats(N, N, &required), KFCORE_KALMAN_OK);
    expect_int("predict workspace size", (int)required, N + N * N + N * N);
    expect_int("predict n>32",
               kalman_predict(x, P, Phi, G, Q, N, N,
                              predict_workspace, sizeof(predict_workspace)/sizeof(predict_workspace[0])),
               KFCORE_KALMAN_OK);

    expect_int("takasu workspace query",
               kalman_takasu_workspace_floats(N, M, &required), KFCORE_KALMAN_OK);
    expect_int("takasu workspace size", (int)required, N * M + M * M + M);
    expect_int("takasu n>32 m>4",
               kalman_takasu(x, P, dz, R, Ht, N, M, 0.0f, NULL,
                             update_workspace, sizeof(update_workspace)/sizeof(update_workspace[0])),
               KFCORE_KALMAN_OK);

    expect_int("udu workspace query",
               kalman_udu_workspace_floats(N, &required), KFCORE_KALMAN_OK);
    expect_int("udu workspace size", (int)required, 2 * N);
    expect_int("udu n>32 m>4",
               kalman_udu(x, U, d, z, R, Ht, N, M, 0.0f, 0,
                          udu_workspace, sizeof(udu_workspace)/sizeof(udu_workspace[0])),
               KFCORE_KALMAN_OK);

    expect_int("udu predict workspace query",
               kalman_udu_predict_workspace_floats(N, N, &required), KFCORE_KALMAN_OK);
    expect_int("udu predict n>32",
               kalman_udu_predict(x, U, d, Phi, G, Q, N, N,
                                  udu_predict_workspace,
                                  sizeof(udu_predict_workspace)/sizeof(udu_predict_workspace[0])),
               KFCORE_KALMAN_OK);

    {
        float sx[1] = { 7.0f };
        float sP[1] = { 9.0f };
        float sPhi[1] = { 1.0f };
        float tiny[1] = { 0.0f };

        expect_int("undersized predict workspace",
                   kalman_predict(sx, sP, sPhi, NULL, NULL, 1, 0, tiny, 1),
                   KFCORE_KALMAN_WORKSPACE_TOO_SMALL);
        expect_float("workspace failure preserves state", sx[0], 7.0f);
        expect_float("workspace failure preserves covariance", sP[0], 9.0f);
    }

    expect_int("invalid zero state query",
               kalman_predict_workspace_floats(0, 0, &required),
               KFCORE_KALMAN_INVALID_ARGUMENT);
}

static int ekf_identity_transition(float* x_pred, float* Phi, const float* x, int n, void* user)
{
    if (user && *(const int*)user != 0)
    {
        return -1;
    }
    memcpy(x_pred, x, sizeof(float) * (size_t)n);
    fill_identity(Phi, (size_t)n);
    return 0;
}

static int ekf_counting_transition(float* x_pred, float* Phi, const float* x,
                                   int n, void* user)
{
    int* calls = (int*)user;
    ++(*calls);
    memcpy(x_pred, x, sizeof(float) * (size_t)n);
    fill_identity(Phi, (size_t)n);
    return 0;
}

static int ekf_identity_measurement(float* z_pred, float* Ht, const float* x,
                                    int n, int m, void* user)
{
    if (user && *(const int*)user != 0)
    {
        return -1;
    }
    memset(Ht, 0, sizeof(float) * (size_t)n * (size_t)m);
    for (int i = 0; i < m; ++i)
    {
        z_pred[i] = x[i];
        Ht[(size_t)i + (size_t)i * (size_t)n] = 1.0f;
    }
    return 0;
}

static void test_ekf_workspace_contract(void)
{
    enum { N = 40, M = 5 };
    float x[N];
    float P[N * N];
    float U[N * N];
    float d[N];
    float z[M];
    float R[M * M];
    float workspace[4096];
    float tiny[1] = { 0.0f };
    size_t required = 0U;

    fill_identity(P, N);
    fill_identity(U, N);
    fill_identity(R, M);
    for (size_t i = 0U; i < N; ++i)
    {
        x[i] = (float)i;
        d[i] = 1.0f;
    }
    for (size_t i = 0U; i < M; ++i)
    {
        z[i] = x[i];
    }

    expect_int("EKF Takasu predict query",
               kalman_ekf_takasu_predict_workspace_floats(N, 0, &required),
               KFCORE_KALMAN_OK);
    expect_int("EKF Takasu predict size", (int)required, 2 * N + 2 * N * N);
    expect_int("EKF Takasu predict n>32",
               kalman_ekf_takasu_predict(
                   x, P, ekf_identity_transition, NULL, NULL, N, 0, NULL,
                   workspace, sizeof(workspace) / sizeof(workspace[0])),
               KFCORE_KALMAN_OK);

    expect_int("EKF Takasu update query",
               kalman_ekf_takasu_update_workspace_floats(N, M, &required),
               KFCORE_KALMAN_OK);
    expect_int("EKF Takasu update size", (int)required,
               3 * M + 2 * N * M + M * M);
    expect_int("EKF Takasu update n>32 m>3",
               kalman_ekf_takasu_update(
                   x, P, z, R, ekf_identity_measurement, N, M,
                   0.0f, NULL, NULL, workspace,
                   sizeof(workspace) / sizeof(workspace[0])),
               KFCORE_KALMAN_OK);

    expect_int("EKF UDU predict query",
               kalman_ekf_udu_predict_workspace_floats(N, 0, &required),
               KFCORE_KALMAN_OK);
    expect_int("EKF UDU predict size", (int)required, 3 * N + 2 * N * N);
    expect_int("EKF UDU predict n>32",
               kalman_ekf_udu_predict(
                   x, U, d, ekf_identity_transition, NULL, NULL, N, 0, NULL,
                   workspace, sizeof(workspace) / sizeof(workspace[0])),
               KFCORE_KALMAN_OK);

    expect_int("EKF UDU update query",
               kalman_ekf_udu_update_workspace_floats(N, M, &required),
               KFCORE_KALMAN_OK);
    expect_int("EKF UDU update size", (int)required,
               2 * M + N * M + M * M + 2 * N);
    expect_int("EKF UDU update n>32 m>3",
               kalman_ekf_udu_update(
                   x, U, d, z, R, ekf_identity_measurement, N, M,
                   0.0f, 0, NULL, workspace,
                   sizeof(workspace) / sizeof(workspace[0])),
               KFCORE_KALMAN_OK);

    {
        int callback_calls = 0;
        expect_int("EKF invalid process-noise arguments",
                   kalman_ekf_takasu_predict(
                       x, P, ekf_counting_transition, NULL, NULL, N, 1,
                       &callback_calls, workspace,
                       sizeof(workspace) / sizeof(workspace[0])),
                   KFCORE_KALMAN_INVALID_ARGUMENT);
        expect_int("EKF invalid arguments skip callback", callback_calls, 0);
    }

    {
        int callback_failure = 1;
        float before_x = x[0];
        float before_p = P[0];

        expect_int("EKF callback failure status",
                   kalman_ekf_takasu_predict(
                       x, P, ekf_identity_transition, NULL, NULL, N, 0,
                       &callback_failure, workspace,
                       sizeof(workspace) / sizeof(workspace[0])),
                   KFCORE_KALMAN_CALLBACK_FAILURE);
        expect_float("EKF callback failure preserves state", x[0], before_x);
        expect_float("EKF callback failure preserves covariance", P[0], before_p);
    }

    {
        float before_x = x[0];
        float before_p = P[0];

        expect_int("EKF undersized workspace",
                   kalman_ekf_takasu_update(
                       x, P, z, R, ekf_identity_measurement, N, M,
                       0.0f, NULL, NULL, tiny, 1),
                   KFCORE_KALMAN_WORKSPACE_TOO_SMALL);
        expect_float("EKF workspace failure preserves state", x[0], before_x);
        expect_float("EKF workspace failure preserves covariance", P[0], before_p);
    }

    expect_int("EKF invalid zero state query",
               kalman_ekf_takasu_predict_workspace_floats(0, 0, &required),
               KFCORE_KALMAN_INVALID_ARGUMENT);
}

static void test_remaining_fixed_wrappers(void)
{
    float x[1] = { 7.0f };
    float P[1] = { 9.0f };
    float z[1] = { 0.0f };
    float R[1] = { 1.0f };

    expect_int("ukf state overflow", kalman_ukf_predict(x, P, NULL, NULL, 33, NULL, NULL), -1);
    expect_int("ukf measurement overflow",
               kalman_ukf_update(x, P, z, R, NULL, 1, 4, NULL, 0.0f, NULL, NULL), -1);
}

int main(void)
{
    test_udu_pivots();
    test_linear_workspace_contract();
    test_ekf_workspace_contract();
    test_remaining_fixed_wrappers();

    if (failures != 0)
    {
        fprintf(stderr, "kalman safety contract: %d failure(s)\n", failures);
        return 1;
    }

    puts("kalman safety contract: PASS");
    return 0;
}
