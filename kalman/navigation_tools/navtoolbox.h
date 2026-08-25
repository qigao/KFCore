/** @file navtoolbox.h
 * KFCore
 * @author Jan Zwiener (jan@zwiener.org)
 *
 * @brief Navigation Toolbox Helper Functions
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
#ifndef PI_FLOAT
#define PI_FLOAT (3.141592653589793f)
#endif
#ifndef RAD2DEG
#define RAD2DEG(x) ((x) * (180.0f / PI_FLOAT))
#endif
#ifndef DEG2RAD
#define DEG2RAD(x) ((x) * (PI_FLOAT / 180.0f))
#endif
#ifndef CLIGHT
#define CLIGHT (299792458.0)    /* speed of light (m/s) */
#endif
#ifndef OMGE
#define OMGE (7.2921151467E-5f) /* Earth rotation rate 15deg/h */
#endif
#ifndef GRAVITY
#define GRAVITY (9.81f)         /* Gravity */
#endif

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

    /** @brief Calculate an approximate orientation from accelerometer data,
     * assuming that the accelerometer measurement is mainly gravity.
     * Beware that the equations become nearly singular near 90 degrees pitch.
     *
     * @param[in] f Specific force measurement x,y,z component (m/s^2)
     * @param[out] roll_rad Output roll angle (rad)
     * @param[out] pitch_rad Output pitch angle (rad) */
    void nav_roll_pitch_from_accelerometer(const float f[3], float* roll_rad, float* pitch_rad);

    /** @brief Calculate a matrix R that transforms from
     * the body-frame (b) to the navigation-frame (n): R^n_b.
     * @param[in] roll_rad Roll angle in (rad)
     * @param[in] pitch_rad Pitch angle in (rad)
     * @param[in] yaw_rad Yaw angle in (rad)
     * @param[out] R_output Output 3x3 matrix in column-major format */
    void nav_matrix_body2nav(const float roll_rad, const float pitch_rad, const float yaw_rad,
                             float R_output[9]);

    /** @brief Calculate the magnetic heading from magnetometer measurements.
     * The orientation (roll/pitch) of the magnetometer measurements must be known.
     * The magnetometer measurements are given in the body frame.
     *
     * Note: Beware of pitch angles near +/- 90 degrees.
     *
     * @param[in] mb (3x1) Magnetometer measurement in body frame (Tesla or Gauss).
     *                     x (mb[0]) pointing to forward/roll-axis of the vehicle,
     *                     y (mb[1]) pointing to the right of the vehicle (pitch axis)
     *                     z (mb[2]) pointing down (yaw-axis)
     * @param[in] roll_rad Roll angle (rad) of the vehicle relative to Earth tangent plane n-frame.
     * @param[in] pitch_rad Pitch angle (rad) of the vehicle relative to Earth tangent plane n-frame.
     *
     * @return Calculated output yaw angle (rad) of the vehicle relative to magnetic North.
     *
     * Note: This is not the geodetic heading, no declination correction is applied. */
    float nav_mag_heading(const float mb[3], float roll_rad, float pitch_rad);

    /** @brief Wrap angle to [-pi, pi). */
    float nav_wrap_pi(float angle_rad);

    /** @brief Set quaternion to identity [w, x, y, z]. */
    void nav_quat_identity(float q[4]);

    /** @brief Normalize quaternion [w, x, y, z].
     *
     * @return 0 on success, -1 on invalid input or near-zero norm.
     */
    int nav_quat_normalize(float q[4]);

    /** @brief Quaternion multiplication out = a * b, [w, x, y, z]. */
    void nav_quat_multiply(const float a[4], const float b[4], float out[4]);

    /** @brief Conjugate quaternion [w, x, y, z]. */
    void nav_quat_conjugate(const float q[4], float out[4]);

    /** @brief Convert roll/pitch/yaw to quaternion [w, x, y, z]. */
    int nav_quat_from_euler(float roll_rad, float pitch_rad, float yaw_rad, float q[4]);

    /** @brief Convert quaternion [w, x, y, z] to roll/pitch/yaw.
     *
     * Any output pointer can be NULL.
     *
     * @return 0 on success, -1 on invalid input.
     */
    int nav_quat_to_euler(const float q[4], float* roll_rad, float* pitch_rad, float* yaw_rad);

    /** @brief Convert body-to-navigation rotation matrix to quaternion [w, x, y, z].
     *
     * @return 0 on success, -1 on invalid input.
     */
    int nav_quat_from_matrix_body2nav(const float R[9], float q[4]);

    /** @brief Convert quaternion [w, x, y, z] to body-to-navigation matrix. */
    int nav_quat_to_matrix_body2nav(const float q[4], float R[9]);

    /** @brief Rotate a body-frame vector to navigation-frame using quaternion. */
    int nav_quat_rotate_body_to_nav(const float q_body2nav[4], const float v_body[3],
                                    float v_nav[3]);

    /** @brief Rotate a navigation-frame vector to body-frame using quaternion. */
    int nav_quat_rotate_nav_to_body(const float q_body2nav[4], const float v_nav[3],
                                    float v_body[3]);

    /** @brief Integrate quaternion with body-frame gyro rates.
     *
     * @return 0 on success, -1 on invalid input.
     */
    int nav_quat_integrate_gyro(float q_body2nav[4], const float gyro_rad_s[3], float dt_s);

    /** @brief Quaternion IMU complementary update using gyro and accelerometer.
     *
     * The accelerometer is used only as a gravity direction observation. Set
     * accel_gain to 0 for gyro-only propagation.
     *
     * @return 0 on success, -1 on invalid input.
     */
    int nav_quat_complementary_imu(float q_body2nav[4], const float gyro_rad_s[3],
                                   const float accel_body_m_s2[3], float dt_s,
                                   float accel_gain);

    /** @brief Quaternion MARG complementary update using gyro, accelerometer, and magnetometer.
     *
     * mag_ref_nav is the expected magnetic field direction in navigation frame.
     * Both accel and mag measurements are treated as direction observations.
     *
     * @return 0 on success, -1 on invalid input.
     */
    int nav_quat_complementary_marg(float q_body2nav[4], const float gyro_rad_s[3],
                                    const float accel_body_m_s2[3], const float mag_body[3],
                                    const float mag_ref_nav[3], float dt_s, float accel_gain,
                                    float mag_gain);

    /** @brief Remove gravity from a body-frame accelerometer sample using quaternion attitude.
     *
     * Output is linear acceleration in body frame.
     *
     * @return 0 on success, -1 on invalid input.
     */
    int nav_remove_gravity_body_quat(const float accel_body_m_s2[3],
                                     const float q_body2nav[4],
                                     float linear_accel_body_m_s2[3]);

    /** @brief Integrate Euler angles with body-frame gyro rates.
     *
     * Compatibility helper. Quaternion propagation is preferred for navigation
     * computation; Euler angles should normally be used only as input/output.
     *
     * Uses the standard 3-2-1 Euler angle rate equations.
     *
     * @return 0 on success, -1 on invalid input or near singular pitch.
     */
    int nav_euler_integrate_gyro(float* roll_rad, float* pitch_rad, float* yaw_rad,
                                 const float gyro_rad_s[3], float dt_s);

    /** @brief Complementary roll/pitch update from gyro and accelerometer.
     *
     * alpha blends gyro prediction with accelerometer tilt: 1 keeps gyro only,
     * 0 uses accelerometer tilt only.
     *
     * @return 0 on success, -1 on invalid input.
     */
    int nav_complementary_roll_pitch(float* roll_rad, float* pitch_rad, const float gyro_rad_s[3],
                                     const float accel_m_s2[3], float dt_s, float alpha);

    /** @brief Complementary yaw update from gyro z-rate and magnetometer heading.
     *
     * @return 0 on success, -1 on invalid input.
     */
    int nav_complementary_yaw(float* yaw_rad, float gyro_z_rad_s, const float mag_body[3],
                              float roll_rad, float pitch_rad, float dt_s, float alpha);

    /** @brief Compute accelerometer magnitude. */
    float nav_accel_norm(const float accel_m_s2[3]);

    /** @brief Gate accelerometer samples by closeness to gravity.
     *
     * @return 1 if accepted, 0 if rejected, -1 on invalid input.
     */
    int nav_accel_gravity_gate(const float accel_m_s2[3], float tolerance_m_s2,
                               float* norm_m_s2);

    /** @brief Remove gravity from a body-frame accelerometer sample.
     *
     * Output is linear acceleration in body frame.
     */
    void nav_remove_gravity_body(const float accel_body_m_s2[3], float roll_rad, float pitch_rad,
                                 float yaw_rad, float linear_accel_body_m_s2[3]);

    /** @brief Static/ZUPT detector using gyro norm and acceleration norm.
     *
     * @return 1 if static, 0 if moving, -1 on invalid input.
     */
    int nav_zupt_static_gate(const float accel_m_s2[3], const float gyro_rad_s[3],
                             float accel_tolerance_m_s2, float gyro_threshold_rad_s,
                             float* accel_norm_m_s2, float* gyro_norm_rad_s);

#ifdef __cplusplus
}
#endif

/* @} */
