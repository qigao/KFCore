#ifndef KALMAN_TAKASU_H
#define KALMAN_TAKASU_H

#include "kalman_status.h"
#include <stddef.h>

/** @file kalman_takasu.h
 * KFCore
 * @author Jan Zwiener (jan@zwiener.org)
 *
 * @brief Kalman Filter Implementation (Takasu Formulation)
 * @{ */

/******************************************************************************
 * SYSTEM INCLUDE FILES
 ******************************************************************************/

/******************************************************************************
 * PROJECT INCLUDE FILES
 ******************************************************************************/

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
 * FUNCTION PROTOTYPES
 ******************************************************************************/

#ifdef __cplusplus
extern "C"
{
#endif

    /** @brief Kalman Filter Routine (Takasu Formulation).
     *
     * The Takasu formulation typically results in an efficient update step
     * which saves CPU time. Numerical stability is good (but not as good as
     * Joseph's form update or the UDU update). This is a good choice if speed
     * is important, the problem formulation is well conditioned numerically
     * and the UDU "square root" formulation is too invasive for the given
     * environment.
     *
     * @param[in,out] x System state (n x 1)
     * @param[in,out] P Upper triangular Covariance matrix of state estimation uncertainty (n x n)
     * @param[in] dz Measurement residual vector: measurement vs. expected measurement: z - H*x (m x
     * 1)
     * @param[in] R Full covariance matrix of measurement uncertainty (m x m)
     * @param[in] Ht Transposed (!) measurement sensitivity matrix (n x m) (H would be m x n)
     * @param[in] n Number of state variables
     * @param[in] m Number of measurements
     * @param[in] chi2_threshold Scalar threshold for chi2 outlier removal threshold. Set to 0.0f to
     * disable the outlier removal.
     * @param[out] chi2 Calculate the chi2 test statistics
     *
     * Note 1: only the upper triangular part of P is referenced and updated.
     * Note 2:
     *    chi2 threshold for 95% confidence is given in table below. E.g. if
     *    dz contains 3 measurements, use 7.8147 (MATLAB chi2inv(0.95, 3)).
     *
     *    chi2inv(0.95, 1:20):
     *    chi95%  = [  3.8415  5.9915  7.8147  9.4877 11.0705 12.5916 14.0671 ...
     *                15.5073 16.9190 18.3070 19.6751 21.0261 22.3620 23.6848 ...
     *                24.9958 26.2962 27.5871 28.8693 30.1435 31.4104 ];
     *
     *    chi2inv(0.99, 1:20):
     *    chi99% = [ 6.6349,  9.2103, 11.3449, 13.2767, 15.0863, 16.8119, ...
     *              18.4753, 20.0902, 21.6660, 23.2093, 24.7250, 26.2170, 27.6882, ...
     *              29.1412, 30.5779, 31.9999, 33.4087, 34.8053, 36.1909, 37.5662 ];
     *
     */
    /** Return the caller-owned float scratch required by kalman_takasu().
     * The query performs the same dimension/range checks as the operation and
     * fails closed on checked-size overflow.
     */
    kfcore_kalman_status kalman_takasu_workspace_floats(size_t n, size_t m, size_t* required);

    /** Takasu update using caller-owned scratch.
     * @param[in,out] x System state (n x 1).
     * @param[in,out] P Upper-triangular covariance (n x n).
     * @param[in] dz Measurement residual (m x 1).
     * @param[in] R Measurement covariance (m x m).
     * @param[in] Ht Transposed measurement matrix (n x m).
     * @param[in] n State dimension.
     * @param[in] m Measurement dimension.
     * @param[in] chi2_threshold Outlier threshold; <=0 disables rejection.
     * @param[out] chi2 Optional normalized chi-square statistic.
     * @param[in,out] workspace Caller-owned float scratch. Must not overlap inputs/state.
     * @param[in] workspace_floats Number of float elements available in workspace.
     *
     * Validation/workspace failure occurs before x/P mutation.
     * @return KFCORE_KALMAN_OK, KFCORE_KALMAN_REJECTED,
     * KFCORE_KALMAN_INVALID_ARGUMENT, KFCORE_KALMAN_WORKSPACE_TOO_SMALL,
     * KFCORE_KALMAN_SIZE_OVERFLOW, or KFCORE_KALMAN_NUMERICAL_FAILURE.
     */
    kfcore_kalman_status kalman_takasu(float* x, float* P, const float* dz, const float* R,
                                      const float* Ht, size_t n, size_t m,
                                      float chi2_threshold, float* chi2,
                                      float* workspace, size_t workspace_floats);

    /** @brief Kalman Temporal / Prediction Step
     *
     *  @param[in,out] x   (optional *) state vector with size (n x 1)
     *  @param[in,out] P   covariance matrix of state estimation
     *                     uncertainty P (n x n).
     *  @param[in] Phi     state transition matrix (n x n)
     *  @param[in] G       process noise distribution matrix (n x r)
     *  @param[in] Q       diagonal vector of covariance matrix of process noise
     *                     in the stochastic system model (r x 1) (diag(Q) has size r x r)
     *  @param[in] n       State vector size n.
     *  @param[in] r       Process noise matrix size.
     *
     * (*) Optional, as a non-linear filter will do the prediction of the state vector
     * with a dedicated (non-linear) function.
     * This will basically just predict the state vector with:
     *
     *      x^{-} = Phi*x^{+}
     *      P^{+} = Phi*P^{-}*Phi' + G*diag(Q)*G'
     */
    /** Return the caller-owned float scratch required by kalman_predict(). */
    kfcore_kalman_status kalman_predict_workspace_floats(size_t n, size_t r, size_t* required);

    /** Linear prediction using caller-owned scratch.
     * x and P are independently optional, but at least one must be non-NULL.
     * For r>0 with P requested, G and Q are required.
     * @param[in,out] workspace Caller-owned float scratch; no overlap with inputs/state.
     * @param[in] workspace_floats Number of float elements available.
     * @return KFCORE_KALMAN_OK or a validation/workspace/size failure status.
     */
    kfcore_kalman_status kalman_predict(float* x, float* P, const float* Phi, const float* G,
                                       const float* Q, size_t n, size_t r,
                                       float* workspace, size_t workspace_floats);

#ifdef __cplusplus
}
#endif

/* @} */

#endif /* KALMAN_TAKASU_H */
