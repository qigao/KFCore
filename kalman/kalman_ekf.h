/** @file kalman_ekf.h
 * KFCore
 * @author Jan Zwiener (jan@zwiener.org)
 *
 * @brief Extended Kalman Filter helpers
 * @{ */

#ifndef KALMAN_EKF_H
#define KALMAN_EKF_H

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

/** @brief EKF prediction using the Takasu covariance representation.
 *
 * Calls the nonlinear transition callback to compute x^- = f(x) and the
 * Jacobian Phi, then updates P with the existing linear prediction routine.
 *
 * @param[in,out] x State vector (n x 1).
 * @param[in,out] P Covariance matrix (n x n).
 * @param[in] transition Nonlinear transition callback.
 * @param[in] G Process noise distribution matrix (n x r), can be NULL if Q is NULL.
 * @param[in] Q Diagonal process noise covariance vector (r x 1), can be NULL.
 * @param[in] n Number of state variables.
 * @param[in] r Number of process noise variables.
 * @param[in,out] user Caller supplied callback context pointer.
 *
 * @return 0 on success, -1 on invalid input or callback failure.
 */
int kalman_ekf_takasu_predict(float* x, float* P, kalman_ekf_transition_fn transition,
                              const float* G, const float* Q, int n, int r, void* user);

/** @brief EKF correction using the Takasu update.
 *
 * Calls the nonlinear measurement callback to compute z_pred = h(x) and H,
 * forms dz = z - z_pred, and reuses kalman_takasu().
 *
 * @param[in,out] x State vector (n x 1).
 * @param[in,out] P Covariance matrix (n x n).
 * @param[in] z Measurement vector (m x 1).
 * @param[in] R Measurement covariance matrix (m x m).
 * @param[in] measurement Nonlinear measurement callback.
 * @param[in] n Number of state variables.
 * @param[in] m Number of measurements.
 * @param[in] chi2_threshold Scalar threshold for chi2 outlier removal. Set to 0.0f to disable.
 * @param[out] chi2 Optional chi2 statistic output.
 * @param[in,out] user Caller supplied callback context pointer.
 *
 * @return 0 on success, -1 on error, -2 if measurement is rejected as outlier.
 */
int kalman_ekf_takasu_update(float* x, float* P, const float* z, const float* R,
                             kalman_ekf_measurement_fn measurement, int n, int m,
                             float chi2_threshold, float* chi2, void* user);

/** @brief EKF prediction using UDU covariance factors.
 *
 * Calls the nonlinear transition callback to compute x^- = f(x) and Phi, then
 * updates U and d with the existing UDU prediction routine.
 *
 * @return 0 on success, -1 on invalid input or callback failure.
 */
int kalman_ekf_udu_predict(float* x, float* U, float* d, kalman_ekf_transition_fn transition,
                           const float* G, const float* Q, int n, int r, void* user);

/** @brief EKF correction using UDU covariance factors.
 *
 * Calls the nonlinear measurement callback to compute z_pred = h(x) and H,
 * forms dz = z - z_pred, decorrelates the residual/Jacobian with R, and applies
 * scalar UDU updates.
 *
 * @return 0 on success, -1 on error.
 */
int kalman_ekf_udu_update(float* x, float* U, float* d, const float* z, const float* R,
                          kalman_ekf_measurement_fn measurement, int n, int m, float chi2_threshold,
                          int downweight_outlier, void* user);

#endif /* KALMAN_EKF_H */

/* @} */
