/** @file kalman_ukf.h
 * KFCore
 * @author Jan Zwiener (jan@zwiener.org)
 *
 * @brief Unscented Kalman Filter helpers
 * @{ */

#ifndef KALMAN_UKF_H
#define KALMAN_UKF_H

#include "kalman_status.h"

#include <stddef.h>

/** Unscented transform tuning parameters.
 *
 * If params is NULL, KFCore uses alpha = 1.0f, beta = 2.0f, kappa = 0.0f.
 */
typedef struct kalman_ukf_params
{
    float alpha;
    float beta;
    float kappa;
} kalman_ukf_params;

typedef int (*kalman_ukf_transition_fn)(float* x_pred, const float* x, int n, void* user);
typedef int (*kalman_ukf_measurement_fn)(float* z_pred, const float* x, int n, int m, void* user);

/** Return caller-owned float scratch required by kalman_ukf_predict(). */
kfcore_kalman_status kalman_ukf_predict_workspace_floats(size_t n, size_t* required);

/** UKF prediction with additive process noise and caller-owned scratch.
 *
 * Callback/workspace/Cholesky failure occurs before x/P mutation.
 */
kfcore_kalman_status kalman_ukf_predict(
    float* x, float* P, const float* Q, kalman_ukf_transition_fn transition,
    size_t n, const kalman_ukf_params* params, void* user,
    float* workspace, size_t workspace_floats);

/** Return caller-owned float scratch required by kalman_ukf_update(). */
kfcore_kalman_status kalman_ukf_update_workspace_floats(size_t n, size_t m,
                                                        size_t* required);

/** UKF correction with additive measurement noise and caller-owned scratch.
 *
 * Callback/workspace/innovation-Cholesky/outlier rejection occurs before x/P
 * mutation.
 */
kfcore_kalman_status kalman_ukf_update(
    float* x, float* P, const float* z, const float* R,
    kalman_ukf_measurement_fn measurement, size_t n, size_t m,
    const kalman_ukf_params* params, float chi2_threshold, float* chi2,
    void* user, float* workspace, size_t workspace_floats);

#endif /* KALMAN_UKF_H */

/* @} */
