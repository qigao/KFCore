#include <math.h>
#include <stdio.h>

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
    expect_int("udu null input", udu(NULL, U, d, 1), -1);
    expect_int("udu zero dimension", udu(identity, U, d, 0), -1);
}

static void test_release_dimension_guards(void)
{
    float x[1] = { 7.0f };
    float P[1] = { 9.0f };
    float U[1] = { 1.0f };
    float d[1] = { 2.0f };
    float z[1] = { 0.0f };
    float R[1] = { 1.0f };
    float Ht[1] = { 1.0f };
    float Phi[1] = { 1.0f };

    expect_int("takasu zero state", kalman_takasu(x, P, z, R, Ht, 0, 1, 0.0f, NULL), -1);
    expect_int("takasu state overflow", kalman_takasu(x, P, z, R, Ht, 33, 1, 0.0f, NULL), -1);
    expect_int("takasu measurement overflow", kalman_takasu(x, P, z, R, Ht, 1, 5, 0.0f, NULL), -1);
    expect_int("takasu null state", kalman_takasu(NULL, P, z, R, Ht, 1, 1, 0.0f, NULL), -1);

    expect_int("udu scalar state overflow", kalman_udu_scalar(x, U, d, 0.0f, 1.0f, Ht, 33), -1);
    expect_int("udu scalar NaN R", kalman_udu_scalar(x, U, d, 0.0f, NAN, Ht, 1), -1);
    expect_int("udu state overflow", kalman_udu(x, U, d, z, R, Ht, 33, 1, 0.0f, 0), -1);
    expect_int("udu zero measurements", kalman_udu(x, U, d, z, R, Ht, 1, 0, 0.0f, 0), -1);
    expect_int("decorrelate zero state", decorrelate(z, Ht, R, 0, 1), -1);

    expect_int("ekf Takasu state overflow",
               kalman_ekf_takasu_predict(x, P, NULL, NULL, NULL, 33, 0, NULL), -1);
    expect_int("ekf Takasu measurement overflow",
               kalman_ekf_takasu_update(x, P, z, R, NULL, 1, 4, 0.0f, NULL, NULL), -1);
    expect_int("ekf UDU state overflow",
               kalman_ekf_udu_predict(x, U, d, NULL, NULL, NULL, 33, 0, NULL), -1);
    expect_int("ekf UDU measurement overflow",
               kalman_ekf_udu_update(x, U, d, z, R, NULL, 1, 4, 0.0f, 0, NULL), -1);

    expect_int("ukf state overflow", kalman_ukf_predict(x, P, NULL, NULL, 33, NULL, NULL), -1);
    expect_int("ukf measurement overflow",
               kalman_ukf_update(x, P, z, R, NULL, 1, 4, NULL, 0.0f, NULL, NULL), -1);

    kalman_predict(x, P, Phi, NULL, NULL, 33, 0);
    expect_float("invalid kalman_predict state", x[0], 7.0f);
    expect_float("invalid kalman_predict covariance", P[0], 9.0f);

    kalman_udu_predict(x, U, d, Phi, NULL, NULL, 33, 0);
    expect_float("invalid kalman_udu_predict state", x[0], 7.0f);
    expect_float("invalid kalman_udu_predict U", U[0], 1.0f);
    expect_float("invalid kalman_udu_predict d", d[0], 2.0f);
}

int main(void)
{
    test_udu_pivots();
    test_release_dimension_guards();

    if (failures != 0)
    {
        fprintf(stderr, "kalman safety contract: %d failure(s)\n", failures);
        return 1;
    }

    puts("kalman safety contract: PASS");
    return 0;
}
