#ifndef KALMAN_UDU_H
#define KALMAN_UDU_H

#include "kalman_status.h"
#include <stddef.h>

/** @file kalman_udu.h
 * @author Jan Zwiener (jan@zwiener.org)
 *
 * @brief UDU Kalman Filter (Bierman/Thornton "Square Root" Implementation)
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

    /** @brief (Robust) Square Root Kalman Filter (Bierman) update routine for linear systems.
     *
     * @param[in,out] x  System state (n x 1)
     * @param[in,out] U  Unit upper triangular factor of covariance matrix of a priori state
     *                   uncertainty (n x n)
     * @param[in,out] d  Unit upper triangular factor of covariance matrix of a priori state
     *                   uncertainty (n x 1)
     * @param[in] z      Measurement vector: z = H*x (m x 1)
     * @param[in] R      Full covariance matrix of measurement uncertainty (m x m)
     * @param[in] Ht     Transposed (!) measurement sensitivity matrix (n x m) (H would be m x n)
     * @param[in] n      Number of state variables
     * @param[in] m      Number of measurements
     * @param[in] chi2_threshold Scalar threshold for outlier classification. Set to 0.0f to
     * disable.
     * @param[in] downweight_outlier If set to 0, measurements classified as outliers are skipped.
     *
     */
    /** Return float scratch required by kalman_udu(). */
    kfcore_kalman_status kalman_udu_workspace_floats(size_t n, size_t* required);

    /** UDU measurement update using caller-owned scratch.
     * workspace must contain the queried number of float elements and must not
     * overlap state/factor/input buffers.
     * @return KFCORE_KALMAN_OK or a validation/workspace/size/numerical status.
     */
    kfcore_kalman_status kalman_udu(float* x, float* U, float* d, const float* z, const float* R,
                                   const float* Ht, size_t n, size_t m,
                                   float chi2_threshold, int downweight_outlier,
                                   float* workspace, size_t workspace_floats);

    /** @brief Square Root Kalman Filter (Bierman) Routine for a single scalar measurement.
     *
     * @param[in,out] x System state (n x 1)
     * @param[in,out] U Unit upper triangular factor of covariance matrix of a priori state
     *                  uncertainty (n x n)
     * @param[in,out] d Unit upper triangular factor of covariance matrix of a priori state
     *                  uncertainty (n x 1)
     * @param[in] dz    Measurement residual dz = z - H*x (1 x 1).
     * @param[in] R     Scalar covariance of measurement uncertainty (1 x 1)
     * @param[in] H_line Row of measurement sensitivity matrix (n x 1)
     * @param[in] n     Number of state variables
     */
    /** Return float scratch required by kalman_udu_scalar(). */
    kfcore_kalman_status kalman_udu_scalar_workspace_floats(size_t n, size_t* required);

    /** Scalar UDU update using caller-owned scratch. */
    kfcore_kalman_status kalman_udu_scalar(float* x, float* U, float* d, float dz, float R,
                                          const float* H_line, size_t n,
                                          float* workspace, size_t workspace_floats);

    /** @brief Decorrelate measurements. For a given covariance matrix R of correlated measurements,
     * calculate a vector of decorrelated measurements (and the matching H-matrix) so that
     * the new covariance R of the decorrelated measurements is an identity matrix.
     *
     * @param[in,out] z Vector of correlated measurements (m x 1).
     * @param[in,out] Ht Transposed measurement sensitivity matrix / design matrix so that z = Ht'*x
     * (n x m).
     * @param[in,out] R Measurement covariance matrix, replaced in-place by chol(R) (m x m).
     *                  So the input is R, overwritten with L such that L*L'=R.
     *                  This is useful for the decorrelation of other measurements.
     * @param[in] n     Number of columns in H (for a Kalman filter: length of state vector x).
     * @param[in] m     Number of measurements in z.
     *
     * Note: If R only has diagonal elements, the only use would be that an
     * identity covariance matrix R is required, otherwise a call to this function
     * is not required.
     *
     * A typical use case would be the decorrelation of meassurements for the UDU
     * filter:
     *
     *      decorrelate(z, Ht, R, n, m); // in-place decorrelation of z and Ht
     *      float Reye[3*3];
     *      mateye(Reye, 3); // set R to eye(2)
     *      size_t workspace_floats = 0;
     *      kalman_udu_workspace_floats(n, &workspace_floats);
     *      float workspace[WORKSPACE_FLOATS];
     *      kalman_udu(x, U, d, z, Reye, Ht, n, m, 0.0f, 0,
     *                 workspace, workspace_floats);
     *
     * If Ht and R do not change, subsequently only measurements z need to be decorrelated.
     * As the input R is replaced by L (such that L*L' = R), L can be reused to
     * decorrelate further measurements:
     *
     *     trisolve(L, z, m, 1, "N");
     *
     * @return KFCORE_KALMAN_OK on success. A numerical failure may leave z/Ht/R
     * partially transformed; discard them in that case.
     */
    kfcore_kalman_status decorrelate(float* z, float* Ht, float* R, size_t n, size_t m);

    /** @brief UDU' (Thornton) Filter Temporal / Prediction Step
     *
     *  Catherine Thornton's modified weighted Gram-Schmidt orthogonalization
     *  method for the predictor update of the U-D factors of the covariance matrix
     *  of estimation uncertainty in Kalman filtering. Source: [1].
     *
     *  P = U*D*U' = Uin * diag(din) * Uin'
     *
     *  @param[in,out] x   (optional *) state vector with size (n x 1)
     *  @param[in,out] U   unit upper triangular factor (U) of the modified Cholesky
     *                     factors (U-D factors) of the covariance matrix of
     *                     corrected state estimation uncertainty P^{+} (n x n).
     *                     Updated in-place to the modified factors (U-D)
     *                     of the covariance matrix of predicted state
     *                     estimation uncertainty P^{-}, so that
     *                     U*diag(d)*U' = P^{-} after this function.
     *  @param[in,out] d   diagonal factor (d) vector (n x 1) of the U-D factors
     *                     of the covariance matrix of corrected estimation
     *                     uncertainty P^{+}, so that diag(d) = D.
     *                     Updated in-place so that P^{-} = U*diag(d)*U'
     *  @param[in] Phi     state transition matrix (n x n)
     *  @param[in,out] G   process noise distribution matrix (modified, if necessary to
     *                     make the associated process noise covariance diagonal) (n x r)
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
     *
     * References:
     *  [1] Grewal, Weill, Andrews. "Global positioning systems, inertial
     *      navigation, and integration". 1st ed. John Wiley & Sons, New York, 2001.
     */
    /** Return float scratch required by kalman_udu_predict(). */
    kfcore_kalman_status kalman_udu_predict_workspace_floats(size_t n, size_t r,
                                                             size_t* required);

    /** Thornton UDU prediction using caller-owned scratch.
     * For r>0 both G and Q are required. Validation/workspace failure occurs
     * before x/U/d mutation; numerical failure may leave outputs modified and
     * the caller must discard them.
     */
    kfcore_kalman_status kalman_udu_predict(float* x, float* U, float* d, const float* Phi,
                                           const float* G, const float* Q, size_t n, size_t r,
                                           float* workspace, size_t workspace_floats);

#ifdef __cplusplus
}
#endif

/* @} */

#endif /* KALMAN_UDU_H */
