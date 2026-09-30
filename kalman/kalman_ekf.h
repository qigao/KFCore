/** @file kalman_ekf.h
 * KFCore
 * @author Jan Zwiener (jan@zwiener.org)
 *
 * @brief Extended Kalman Filter helpers
 * @{ */

#ifndef KALMAN_EKF_H
#define KALMAN_EKF_H

#include "kalman_status.h"

#include <stddef.h>

/******************************************************************************
 * TYPEDEFS
 ******************************************************************************/

/** @brief Nonlinear state transition callback for EKF prediction.
 *
 * @param[out] x_pred Predicted state vector (n x 1).
 * @param[out] Phi State transition Jacobian evaluated at x (n x n).
 * @param[in] x Current state vector (n x 1).
 * @param[in] n Number of state variables.
 * @param[in,out] user Caller supplied context pointer.
 *
 * @return 0 on success, non-zero on error.
 */
typedef int (*kalman_ekf_transition_fn)(float* x_pred, float* Phi, const float* x, int n,
                                        void* user);

/** @brief Nonlinear measurement callback for EKF correction.
 *
 * @param[out] z_pred Predicted measurement h(x) (m x 1).
 * @param[out] Ht Transposed measurement Jacobian evaluated at x (n x m).
 * @param[in] x Current state vector (n x 1).
 * @param[in] n Number of state variables.
 * @param[in] m Number of measurements.
 * @param[in,out] user Caller supplied context pointer.
 *
 * @return 0 on success, non-zero on error.
 */
typedef int (*kalman_ekf_measurement_fn)(float* z_pred, float* Ht, const float* x, int n, int m,
                                         void* user);

/******************************************************************************
 * FUNCTION PROTOTYPES
 ******************************************************************************/

/** Return caller-owned float scratch required by kalman_ekf_takasu_predict(). */
kfcore_kalman_status kalman_ekf_takasu_predict_workspace_floats(size_t n, size_t r,
                                                                size_t* required);

/** EKF prediction using the Takasu covariance representation.
 *
 * The callback writes its predicted state/Jacobian into workspace. P is updated
 * only after callback success; x is copied from the predicted state only after
 * the linear covariance prediction succeeds.
 */
kfcore_kalman_status kalman_ekf_takasu_predict(
    float* x, float* P, kalman_ekf_transition_fn transition,
    const float* G, const float* Q, size_t n, size_t r, void* user,
    float* workspace, size_t workspace_floats);

/** Return caller-owned float scratch required by kalman_ekf_takasu_update(). */
kfcore_kalman_status kalman_ekf_takasu_update_workspace_floats(size_t n, size_t m,
                                                               size_t* required);

/** EKF correction using the Takasu update and caller-owned scratch. */
kfcore_kalman_status kalman_ekf_takasu_update(
    float* x, float* P, const float* z, const float* R,
    kalman_ekf_measurement_fn measurement, size_t n, size_t m,
    float chi2_threshold, float* chi2, void* user,
    float* workspace, size_t workspace_floats);

/** Return caller-owned float scratch required by kalman_ekf_udu_predict(). */
kfcore_kalman_status kalman_ekf_udu_predict_workspace_floats(size_t n, size_t r,
                                                             size_t* required);

/** EKF prediction using UDU covariance factors and caller-owned scratch. */
kfcore_kalman_status kalman_ekf_udu_predict(
    float* x, float* U, float* d, kalman_ekf_transition_fn transition,
    const float* G, const float* Q, size_t n, size_t r, void* user,
    float* workspace, size_t workspace_floats);

/** Return caller-owned float scratch required by kalman_ekf_udu_update(). */
kfcore_kalman_status kalman_ekf_udu_update_workspace_floats(size_t n, size_t m,
                                                            size_t* required);

/** EKF correction using UDU covariance factors and caller-owned scratch.
 *
 * A callback/validation/workspace failure occurs before filter-state mutation.
 * A later numerical failure may follow earlier successful scalar measurements;
 * in that case the caller must discard x/U/d.
 */
kfcore_kalman_status kalman_ekf_udu_update(
    float* x, float* U, float* d, const float* z, const float* R,
    kalman_ekf_measurement_fn measurement, size_t n, size_t m,
    float chi2_threshold, int downweight_outlier, void* user,
    float* workspace, size_t workspace_floats);

#endif /* KALMAN_EKF_H */

/* @} */
