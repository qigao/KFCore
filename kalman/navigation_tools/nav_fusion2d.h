/** @file nav_fusion2d.h
 * KFCore
 *
 * @brief Lightweight 2D navigation fusion system layer.
 * @{ */

#ifndef NAV_FUSION2D_H
#define NAV_FUSION2D_H

#ifdef __cplusplus
extern "C"
{
#endif

enum
{
    NAV_FUSION2D_STATE_SIZE = 8
};

enum
{
    NAV_FUSION2D_X = 0,
    NAV_FUSION2D_Y = 1,
    NAV_FUSION2D_VX = 2,
    NAV_FUSION2D_VY = 3,
    NAV_FUSION2D_YAW = 4,
    NAV_FUSION2D_BAX = 5,
    NAV_FUSION2D_BAY = 6,
    NAV_FUSION2D_BGZ = 7
};

typedef struct nav_fusion2d
{
    float x[NAV_FUSION2D_STATE_SIZE];
    float P[NAV_FUSION2D_STATE_SIZE * NAV_FUSION2D_STATE_SIZE];
} nav_fusion2d;

/** @brief Initialize state and diagonal covariance.
 *
 * initial_std can be NULL. If supplied, P diagonal is initial_std[i]^2.
 *
 * @return 0 on success, -1 on invalid input.
 */
int nav_fusion2d_init(nav_fusion2d* fusion, const float initial_state[NAV_FUSION2D_STATE_SIZE],
                      const float initial_std[NAV_FUSION2D_STATE_SIZE]);

/** @brief IMU prediction using body-frame acceleration x/y and yaw-rate z.
 *
 * process_var_diag can be NULL. If supplied, it is a diagonal process noise
 * vector for all 8 states.
 *
 * @return 0 on success, -1 on invalid input.
 */
int nav_fusion2d_predict_imu(nav_fusion2d* fusion, const float accel_body_m_s2[2],
                             float gyro_z_rad_s, float dt_s,
                             const float process_var_diag[NAV_FUSION2D_STATE_SIZE]);

/** @brief Update x/y position, e.g. GNSS or visual pose.
 *
 * @return Kalman update return code.
 */
int nav_fusion2d_update_position(nav_fusion2d* fusion, const float position_xy[2],
                                 const float R_pos[4], float chi2_threshold, float* chi2);

/** @brief Update navigation-frame velocity x/y.
 *
 * @return Kalman update return code.
 */
int nav_fusion2d_update_velocity_nav(nav_fusion2d* fusion, const float velocity_xy[2],
                                     const float R_vel[4], float chi2_threshold, float* chi2);

/** @brief Update body-frame velocity x/y, e.g. optical flow converted to metric velocity.
 *
 * @return Kalman update return code.
 */
int nav_fusion2d_update_velocity_body(nav_fusion2d* fusion, const float velocity_body_xy[2],
                                      const float R_vel_body[4], float chi2_threshold,
                                      float* chi2);

/** @brief Update yaw angle measurement.
 *
 * @return Kalman update return code.
 */
int nav_fusion2d_update_yaw(nav_fusion2d* fusion, float yaw_rad, float R_yaw,
                            float chi2_threshold, float* chi2);

/** @brief Zero-velocity update.
 *
 * @return Kalman update return code.
 */
int nav_fusion2d_update_zupt(nav_fusion2d* fusion, const float R_vel[4], float chi2_threshold,
                             float* chi2);

#ifdef __cplusplus
}
#endif

#endif /* NAV_FUSION2D_H */

/* @} */
