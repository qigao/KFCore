/** @file kalman_ukf.h
 * KFCore
 * @author Jan Zwiener (jan@zwiener.org)
 *
 * @brief Unscented Kalman Filter helpers
 * @{ */

#ifndef KALMAN_UKF_H
#define KALMAN_UKF_H

/******************************************************************************
 * TYPEDEFS
 ******************************************************************************/

/** @brief Unscented transform tuning parameters.
 *
 * If params is NULL, KFCore uses alpha = 1.0f, beta = 2.0f, kappa = 0.0f.
 */
typedef struct kalman_ukf_params
{
    float alpha;
    float beta;
    float kappa;
} kalman_ukf_params;

/** @brief Nonlinear state transition callback for UKF prediction.
 *
 * @param[out] x_pred Predicted state vector (n x 1).
 * @param[in] x Input sigma point/state vector (n x 1).
 * @param[in] n Number of state variables.
 * @param[in,out] user Caller supplied context pointer.
 *
 * @return 0 on success, non-zero on error.
 */
typedef int (*kalman_ukf_transition_fn)(float* x_pred, const float* x, int n, void* user);

/** @brief Nonlinear measurement callback for UKF correction.
 *
 * @param[out] z_pred Predicted measurement h(x) (m x 1).
 * @param[in] x Input sigma point/state vector (n x 1).
 * @param[in] n Number of state variables.
 * @param[in] m Number of measurements.
 * @param[in,out] user Caller supplied context pointer.
 *
 * @return 0 on success, non-zero on error.
 */
typedef int (*kalman_ukf_measurement_fn)(float* z_pred, const float* x, int n, int m, void* user);

/******************************************************************************
 * FUNCTION PROTOTYPES
 ******************************************************************************/

/** @brief UKF prediction with additive process noise.
 *
 * @param[in,out] x State vector (n x 1).
 * @param[in,out] P Covariance matrix (n x n).
 * @param[in] Q Additive process noise covariance (n x n), can be NULL.
 * @param[in] transition Nonlinear transition callback.
 * @param[in] n Number of state variables.
 * @param[in] params Unscented transform tuning parameters, can be NULL.
 * @param[in,out] user Caller supplied callback context pointer.
 *
 * @return 0 on success, -1 on error.
 */
int kalman_ukf_predict(float* x, float* P, const float* Q, kalman_ukf_transition_fn transition,
                       int n, const kalman_ukf_params* params, void* user);

/** @brief UKF correction with additive measurement noise.
 *
 * @param[in,out] x State vector (n x 1).
 * @param[in,out] P Covariance matrix (n x n).
 * @param[in] z Measurement vector (m x 1).
 * @param[in] R Measurement covariance matrix (m x m).
 * @param[in] measurement Nonlinear measurement callback.
 * @param[in] n Number of state variables.
 * @param[in] m Number of measurements.
 * @param[in] params Unscented transform tuning parameters, can be NULL.
 * @param[in] chi2_threshold Scalar threshold for chi2 outlier removal. Set to 0.0f to disable.
 * @param[out] chi2 Optional chi2 statistic output.
 * @param[in,out] user Caller supplied callback context pointer.
 *
 * @return 0 on success, -1 on error, -2 if measurement is rejected as outlier.
 */
int kalman_ukf_update(float* x, float* P, const float* z, const float* R,
                      kalman_ukf_measurement_fn measurement, int n, int m,
                      const kalman_ukf_params* params, float chi2_threshold, float* chi2,
                      void* user);

#endif /* KALMAN_UKF_H */

/* @} */
