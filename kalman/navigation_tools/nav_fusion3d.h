/** @file nav_fusion3d.h
 * KFCore
 *
 * @brief Lightweight 3D quaternion error-state navigation fusion layer.
 * @{ */

#ifndef NAV_FUSION3D_H
#define NAV_FUSION3D_H

#ifdef __cplusplus
extern "C"
{
#endif

enum
{
    NAV_FUSION3D_ERROR_SIZE = 15
};

enum
{
    NAV_FUSION3D_DP = 0,
    NAV_FUSION3D_DV = 3,
    NAV_FUSION3D_DTHETA = 6,
    NAV_FUSION3D_DBA = 9,
    NAV_FUSION3D_DBG = 12
};

typedef struct nav_fusion3d
{
    float p[3];                         /**< Navigation-frame position. */
    float v[3];                         /**< Navigation-frame velocity. */
    float q[4];                         /**< Body-to-navigation quaternion [w, x, y, z]. */
    float ba[3];                        /**< Accelerometer bias, body frame. */
    float bg[3];                        /**< Gyro bias, body frame. */
    float P[NAV_FUSION3D_ERROR_SIZE * NAV_FUSION3D_ERROR_SIZE]; /**< Error covariance. */
} nav_fusion3d;

/** @brief Initialize 3D fusion state.
 *
 * initial_std is a 15-element diagonal standard deviation vector for
 * dp,dv,dtheta,dba,dbg. It can be NULL.
 *
 * @return 0 on success, -1 on invalid input.
 */
int nav_fusion3d_init(nav_fusion3d* fusion, const float p[3], const float v[3], const float q[4],
                      const float ba[3], const float bg[3],
                      const float initial_std[NAV_FUSION3D_ERROR_SIZE]);

/** @brief IMU prediction with quaternion nominal state.
 *
 * process_var_diag is a 15-element diagonal process noise vector for the error
 * state. It can be NULL.
 *
 * @return 0 on success, -1 on invalid input.
 */
int nav_fusion3d_predict_imu(nav_fusion3d* fusion, const float accel_body_m_s2[3],
                             const float gyro_rad_s[3], float dt_s,
                             const float process_var_diag[NAV_FUSION3D_ERROR_SIZE]);

/** @brief Update navigation-frame position.
 *
 * @return Kalman update return code.
 */
int nav_fusion3d_update_position(nav_fusion3d* fusion, const float position_nav[3],
                                 const float R_pos[9], float chi2_threshold, float* chi2);

/** @brief Update navigation-frame velocity.
 *
 * @return Kalman update return code.
 */
int nav_fusion3d_update_velocity(nav_fusion3d* fusion, const float velocity_nav[3],
                                 const float R_vel[9], float chi2_threshold, float* chi2);

/** @brief Zero-velocity update.
 *
 * @return Kalman update return code.
 */
int nav_fusion3d_update_zupt(nav_fusion3d* fusion, const float R_vel[9], float chi2_threshold,
                             float* chi2);

/** @brief Update attitude from a body-to-navigation quaternion measurement.
 *
 * @return Kalman update return code.
 */
int nav_fusion3d_update_attitude(nav_fusion3d* fusion, const float q_body2nav_meas[4],
                                 const float R_att[9], float chi2_threshold, float* chi2);

#ifdef __cplusplus
}
#endif

#endif /* NAV_FUSION3D_H */

/* @} */
