/** @file navtoolbox.c
 * KFCore
 * @author Jan Zwiener (jan@zwiener.org)
 *
 * @brief Navigation Toolbox Helper Functions
 * @{ */

/******************************************************************************
 * SYSTEM INCLUDE FILES
 ******************************************************************************/

#include <math.h>
#include <stddef.h>
#include <assert.h>

/******************************************************************************
 * PROJECT INCLUDE FILES
 ******************************************************************************/

#include "navtoolbox.h"
#include "linalg.h"

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

static int nav_vec3_normalize_copy(const float in[3], float out[3]);
static void nav_vec3_cross(const float a[3], const float b[3], float out[3]);
static void nav_vec3_accumulate_scaled(float acc[3], const float x[3], float scale);

/******************************************************************************
 * FUNCTION BODIES
 ******************************************************************************/

void nav_roll_pitch_from_accelerometer(const float f[3], float* roll_rad, float* pitch_rad)
{
    /* Source: Farrell, Jay. Aided navigation: GPS with high rate sensors.
     * McGraw-Hill, Inc., 2008.  */
    if (roll_rad)
    {
        *roll_rad = atan2f(-f[1], -f[2]); /* eq. 10.14 */
    }
    if (pitch_rad)
    {
        *pitch_rad = atan2f(f[0], SQRTF(f[1] * f[1] + f[2] * f[2])); /* eq. 10.15 */
    }
}

void nav_matrix_body2nav(const float roll_rad, const float pitch_rad, const float yaw_rad,
                         float R_output[9])
{
    const float sinr = sinf(roll_rad);
    const float sinp = sinf(pitch_rad);
    const float siny = sinf(yaw_rad);
    const float cosr = cosf(roll_rad);
    const float cosp = cosf(pitch_rad);
    const float cosy = cosf(yaw_rad);
    /* Source: Farrell, Jay. Aided navigation: GPS with high rate sensors.
     * McGraw-Hill, Inc., 2008. eq. 2.43 */
    R_output[0] = cosp * cosy;
    R_output[1] = cosp * siny;
    R_output[2] = -sinp;
    R_output[3] = sinr * sinp * cosy - cosr * siny;
    R_output[4] = sinr * sinp * siny + cosr * cosy;
    R_output[5] = sinr * cosp;
    R_output[6] = cosr * sinp * cosy + sinr * siny;
    R_output[7] = cosr * sinp * siny - sinr * cosy;
    R_output[8] = cosr * cosp;
}

float nav_mag_heading(const float mb[3], float roll_rad, float pitch_rad)
{
    const float sinr = sinf(roll_rad);
    const float sinp = sinf(pitch_rad);
    const float cosr = cosf(roll_rad);
    const float cosp = cosf(pitch_rad);

    /* Source: Farrell, Jay. Aided navigation: GPS with high rate sensors.
     * McGraw-Hill, Inc., 2008.  */
    /* Transform the magnetometer measurement in the body frame (mb) to the
     * w-frame.  The w-frame is an intermediate frame of reference defined by the
     * projection of the vehicle u-axis onto the Earth tangent plane */
    float mw_x = cosp*mb[0] + sinp*sinr*mb[1] + sinp*cosr*mb[2]; /* eq. 10.16 */
    float mw_y =                   cosr*mb[1] -      sinr*mb[2]; /* eq. 10.16 */

    return atan2f(-mw_y, mw_x);
}

float nav_wrap_pi(float angle_rad)
{
    while (angle_rad >= PI_FLOAT)
    {
        angle_rad -= 2.0f * PI_FLOAT;
    }
    while (angle_rad < -PI_FLOAT)
    {
        angle_rad += 2.0f * PI_FLOAT;
    }

    return angle_rad;
}

void nav_quat_identity(float q[4])
{
    if (!q)
    {
        return;
    }

    q[0] = 1.0f;
    q[1] = 0.0f;
    q[2] = 0.0f;
    q[3] = 0.0f;
}

int nav_quat_normalize(float q[4])
{
    float norm;

    if (!q)
    {
        return -1;
    }

    norm = vecnorm(q, 4);
    if (norm <= 1.0e-12f)
    {
        return -1;
    }

    for (int i = 0; i < 4; ++i)
    {
        q[i] /= norm;
    }

    return 0;
}

void nav_quat_multiply(const float a[4], const float b[4], float out[4])
{
    const float aw = a[0];
    const float ax = a[1];
    const float ay = a[2];
    const float az = a[3];
    const float bw = b[0];
    const float bx = b[1];
    const float by = b[2];
    const float bz = b[3];

    out[0] = aw * bw - ax * bx - ay * by - az * bz;
    out[1] = aw * bx + ax * bw + ay * bz - az * by;
    out[2] = aw * by - ax * bz + ay * bw + az * bx;
    out[3] = aw * bz + ax * by - ay * bx + az * bw;
}

void nav_quat_conjugate(const float q[4], float out[4])
{
    out[0] = q[0];
    out[1] = -q[1];
    out[2] = -q[2];
    out[3] = -q[3];
}

int nav_quat_from_euler(float roll_rad, float pitch_rad, float yaw_rad, float q[4])
{
    const float cr = cosf(0.5f * roll_rad);
    const float sr = sinf(0.5f * roll_rad);
    const float cp = cosf(0.5f * pitch_rad);
    const float sp = sinf(0.5f * pitch_rad);
    const float cy = cosf(0.5f * yaw_rad);
    const float sy = sinf(0.5f * yaw_rad);

    if (!q)
    {
        return -1;
    }

    q[0] = cr * cp * cy + sr * sp * sy;
    q[1] = sr * cp * cy - cr * sp * sy;
    q[2] = cr * sp * cy + sr * cp * sy;
    q[3] = cr * cp * sy - sr * sp * cy;

    return nav_quat_normalize(q);
}

int nav_quat_to_euler(const float q_in[4], float* roll_rad, float* pitch_rad, float* yaw_rad)
{
    float q[4];
    float sinr_cosp;
    float cosr_cosp;
    float sinp;
    float siny_cosp;
    float cosy_cosp;

    if (!q_in)
    {
        return -1;
    }

    for (int i = 0; i < 4; ++i)
    {
        q[i] = q_in[i];
    }
    if (nav_quat_normalize(q) != 0)
    {
        return -1;
    }

    sinr_cosp = 2.0f * (q[0] * q[1] + q[2] * q[3]);
    cosr_cosp = 1.0f - 2.0f * (q[1] * q[1] + q[2] * q[2]);
    sinp      = 2.0f * (q[0] * q[2] - q[3] * q[1]);
    siny_cosp = 2.0f * (q[0] * q[3] + q[1] * q[2]);
    cosy_cosp = 1.0f - 2.0f * (q[2] * q[2] + q[3] * q[3]);

    if (roll_rad)
    {
        *roll_rad = atan2f(sinr_cosp, cosr_cosp);
    }
    if (pitch_rad)
    {
        if (sinp >= 1.0f)
        {
            *pitch_rad = PI_FLOAT * 0.5f;
        }
        else if (sinp <= -1.0f)
        {
            *pitch_rad = -PI_FLOAT * 0.5f;
        }
        else
        {
            *pitch_rad = asinf(sinp);
        }
    }
    if (yaw_rad)
    {
        *yaw_rad = atan2f(siny_cosp, cosy_cosp);
    }

    return 0;
}

int nav_quat_from_matrix_body2nav(const float R[9], float q[4])
{
    float trace;
    float s;

    if (!R || !q)
    {
        return -1;
    }

    trace = R[0] + R[4] + R[8];
    if (trace > 0.0f)
    {
        s    = sqrtf(trace + 1.0f) * 2.0f;
        q[0] = 0.25f * s;
        q[1] = (R[5] - R[7]) / s;
        q[2] = (R[6] - R[2]) / s;
        q[3] = (R[1] - R[3]) / s;
    }
    else if (R[0] > R[4] && R[0] > R[8])
    {
        s    = sqrtf(1.0f + R[0] - R[4] - R[8]) * 2.0f;
        q[0] = (R[5] - R[7]) / s;
        q[1] = 0.25f * s;
        q[2] = (R[3] + R[1]) / s;
        q[3] = (R[6] + R[2]) / s;
    }
    else if (R[4] > R[8])
    {
        s    = sqrtf(1.0f + R[4] - R[0] - R[8]) * 2.0f;
        q[0] = (R[6] - R[2]) / s;
        q[1] = (R[3] + R[1]) / s;
        q[2] = 0.25f * s;
        q[3] = (R[7] + R[5]) / s;
    }
    else
    {
        s    = sqrtf(1.0f + R[8] - R[0] - R[4]) * 2.0f;
        q[0] = (R[1] - R[3]) / s;
        q[1] = (R[6] + R[2]) / s;
        q[2] = (R[7] + R[5]) / s;
        q[3] = 0.25f * s;
    }

    return nav_quat_normalize(q);
}

int nav_quat_to_matrix_body2nav(const float q_in[4], float R[9])
{
    float q[4];
    float ww;
    float xx;
    float yy;
    float zz;
    float wx;
    float wy;
    float wz;
    float xy;
    float xz;
    float yz;

    if (!q_in || !R)
    {
        return -1;
    }

    for (int i = 0; i < 4; ++i)
    {
        q[i] = q_in[i];
    }
    if (nav_quat_normalize(q) != 0)
    {
        return -1;
    }

    ww = q[0] * q[0];
    xx = q[1] * q[1];
    yy = q[2] * q[2];
    zz = q[3] * q[3];
    wx = q[0] * q[1];
    wy = q[0] * q[2];
    wz = q[0] * q[3];
    xy = q[1] * q[2];
    xz = q[1] * q[3];
    yz = q[2] * q[3];

    R[0] = ww + xx - yy - zz;
    R[1] = 2.0f * (xy + wz);
    R[2] = 2.0f * (xz - wy);
    R[3] = 2.0f * (xy - wz);
    R[4] = ww - xx + yy - zz;
    R[5] = 2.0f * (yz + wx);
    R[6] = 2.0f * (xz + wy);
    R[7] = 2.0f * (yz - wx);
    R[8] = ww - xx - yy + zz;

    return 0;
}

int nav_quat_rotate_body_to_nav(const float q_body2nav[4], const float v_body[3], float v_nav[3])
{
    float R[9];

    if (!q_body2nav || !v_body || !v_nav)
    {
        return -1;
    }

    if (nav_quat_to_matrix_body2nav(q_body2nav, R) != 0)
    {
        return -1;
    }
    matvec("N", 3, 3, 1.0f, R, v_body, 0.0f, v_nav);

    return 0;
}

int nav_quat_rotate_nav_to_body(const float q_body2nav[4], const float v_nav[3], float v_body[3])
{
    float R[9];

    if (!q_body2nav || !v_nav || !v_body)
    {
        return -1;
    }

    if (nav_quat_to_matrix_body2nav(q_body2nav, R) != 0)
    {
        return -1;
    }
    matvec("T", 3, 3, 1.0f, R, v_nav, 0.0f, v_body);

    return 0;
}

int nav_quat_integrate_gyro(float q_body2nav[4], const float gyro_rad_s[3], float dt_s)
{
    float omega_norm;
    float half_angle;
    float sin_half;
    float dq[4];
    float q_new[4];

    if (!q_body2nav || !gyro_rad_s || dt_s <= 0.0f)
    {
        return -1;
    }

    omega_norm = vecnorm(gyro_rad_s, 3);
    if (omega_norm <= 1.0e-12f)
    {
        return nav_quat_normalize(q_body2nav);
    }

    half_angle = 0.5f * omega_norm * dt_s;
    sin_half   = sinf(half_angle);
    dq[0]      = cosf(half_angle);
    dq[1]      = sin_half * gyro_rad_s[0] / omega_norm;
    dq[2]      = sin_half * gyro_rad_s[1] / omega_norm;
    dq[3]      = sin_half * gyro_rad_s[2] / omega_norm;

    nav_quat_multiply(q_body2nav, dq, q_new);
    for (int i = 0; i < 4; ++i)
    {
        q_body2nav[i] = q_new[i];
    }

    return nav_quat_normalize(q_body2nav);
}

int nav_quat_complementary_imu(float q_body2nav[4], const float gyro_rad_s[3],
                               const float accel_body_m_s2[3], float dt_s, float accel_gain)
{
    const float gravity_nav[3] = { 0.0f, 0.0f, -1.0f };
    float       accel_unit[3];
    float       gravity_body[3];
    float       error[3];
    float       correction[3];

    if (!q_body2nav || !gyro_rad_s || !accel_body_m_s2 || dt_s <= 0.0f || accel_gain < 0.0f)
    {
        return -1;
    }

    if (nav_quat_integrate_gyro(q_body2nav, gyro_rad_s, dt_s) != 0)
    {
        return -1;
    }

    if (accel_gain > 0.0f && nav_vec3_normalize_copy(accel_body_m_s2, accel_unit) == 0)
    {
        if (nav_quat_rotate_nav_to_body(q_body2nav, gravity_nav, gravity_body) != 0)
        {
            return -1;
        }
        if (nav_vec3_normalize_copy(gravity_body, gravity_body) != 0)
        {
            return -1;
        }

        nav_vec3_cross(accel_unit, gravity_body, error);
        correction[0] = accel_gain * error[0];
        correction[1] = accel_gain * error[1];
        correction[2] = accel_gain * error[2];
        return nav_quat_integrate_gyro(q_body2nav, correction, 1.0f);
    }

    return 0;
}

int nav_quat_complementary_marg(float q_body2nav[4], const float gyro_rad_s[3],
                                const float accel_body_m_s2[3], const float mag_body[3],
                                const float mag_ref_nav[3], float dt_s, float accel_gain,
                                float mag_gain)
{
    const float gravity_nav[3] = { 0.0f, 0.0f, -1.0f };
    float       accel_unit[3];
    float       mag_unit[3];
    float       mag_ref_unit[3];
    float       gravity_body[3];
    float       mag_ref_body[3];
    float       error[3]          = { 0.0f, 0.0f, 0.0f };
    float       partial_error[3]  = { 0.0f, 0.0f, 0.0f };

    if (!q_body2nav || !gyro_rad_s || dt_s <= 0.0f || accel_gain < 0.0f || mag_gain < 0.0f ||
        (accel_gain > 0.0f && !accel_body_m_s2) || (mag_gain > 0.0f && (!mag_body || !mag_ref_nav)))
    {
        return -1;
    }

    if (nav_quat_integrate_gyro(q_body2nav, gyro_rad_s, dt_s) != 0)
    {
        return -1;
    }

    if (accel_gain > 0.0f && nav_vec3_normalize_copy(accel_body_m_s2, accel_unit) == 0)
    {
        if (nav_quat_rotate_nav_to_body(q_body2nav, gravity_nav, gravity_body) != 0)
        {
            return -1;
        }
        if (nav_vec3_normalize_copy(gravity_body, gravity_body) != 0)
        {
            return -1;
        }
        nav_vec3_cross(accel_unit, gravity_body, partial_error);
        nav_vec3_accumulate_scaled(error, partial_error, accel_gain);
    }

    if (mag_gain > 0.0f && nav_vec3_normalize_copy(mag_body, mag_unit) == 0 &&
        nav_vec3_normalize_copy(mag_ref_nav, mag_ref_unit) == 0)
    {
        if (nav_quat_rotate_nav_to_body(q_body2nav, mag_ref_unit, mag_ref_body) != 0)
        {
            return -1;
        }
        if (nav_vec3_normalize_copy(mag_ref_body, mag_ref_body) != 0)
        {
            return -1;
        }
        nav_vec3_cross(mag_unit, mag_ref_body, partial_error);
        nav_vec3_accumulate_scaled(error, partial_error, mag_gain);
    }

    return nav_quat_integrate_gyro(q_body2nav, error, 1.0f);
}

int nav_remove_gravity_body_quat(const float accel_body_m_s2[3], const float q_body2nav[4],
                                 float linear_accel_body_m_s2[3])
{
    const float gravity_nav[3] = { 0.0f, 0.0f, -GRAVITY };
    float       gravity_body[3];

    if (!accel_body_m_s2 || !q_body2nav || !linear_accel_body_m_s2)
    {
        return -1;
    }

    if (nav_quat_rotate_nav_to_body(q_body2nav, gravity_nav, gravity_body) != 0)
    {
        return -1;
    }

    for (int i = 0; i < 3; ++i)
    {
        linear_accel_body_m_s2[i] = accel_body_m_s2[i] - gravity_body[i];
    }

    return 0;
}

int nav_euler_integrate_gyro(float* roll_rad, float* pitch_rad, float* yaw_rad,
                             const float gyro_rad_s[3], float dt_s)
{
    float roll;
    float pitch;
    float tan_pitch;
    float sec_pitch;
    float roll_dot;
    float pitch_dot;
    float yaw_dot;

    if (!roll_rad || !pitch_rad || !yaw_rad || !gyro_rad_s || dt_s <= 0.0f)
    {
        return -1;
    }

    roll  = *roll_rad;
    pitch = *pitch_rad;
    if (fabsf(cosf(pitch)) < 1.0e-4f)
    {
        return -1;
    }

    tan_pitch = tanf(pitch);
    sec_pitch = 1.0f / cosf(pitch);

    roll_dot  = gyro_rad_s[0] + sinf(roll) * tan_pitch * gyro_rad_s[1] +
               cosf(roll) * tan_pitch * gyro_rad_s[2];
    pitch_dot = cosf(roll) * gyro_rad_s[1] - sinf(roll) * gyro_rad_s[2];
    yaw_dot   = sinf(roll) * sec_pitch * gyro_rad_s[1] +
              cosf(roll) * sec_pitch * gyro_rad_s[2];

    *roll_rad  = nav_wrap_pi(*roll_rad + roll_dot * dt_s);
    *pitch_rad = nav_wrap_pi(*pitch_rad + pitch_dot * dt_s);
    *yaw_rad   = nav_wrap_pi(*yaw_rad + yaw_dot * dt_s);

    return 0;
}

int nav_complementary_roll_pitch(float* roll_rad, float* pitch_rad, const float gyro_rad_s[3],
                                 const float accel_m_s2[3], float dt_s, float alpha)
{
    float q[4];

    if (!roll_rad || !pitch_rad || !gyro_rad_s || !accel_m_s2 || dt_s <= 0.0f || alpha < 0.0f ||
        alpha > 1.0f)
    {
        return -1;
    }

    if (nav_quat_from_euler(*roll_rad, *pitch_rad, 0.0f, q) != 0)
    {
        return -1;
    }
    if (nav_quat_complementary_imu(q, gyro_rad_s, accel_m_s2, dt_s, 1.0f - alpha) != 0)
    {
        return -1;
    }
    if (nav_quat_to_euler(q, roll_rad, pitch_rad, NULL) != 0)
    {
        return -1;
    }

    return 0;
}

int nav_complementary_yaw(float* yaw_rad, float gyro_z_rad_s, const float mag_body[3],
                          float roll_rad, float pitch_rad, float dt_s, float alpha)
{
    const float mag_ref_nav[3] = { 1.0f, 0.0f, 0.0f };
    float       q[4];
    float       gyro[3] = { 0.0f, 0.0f, gyro_z_rad_s };

    if (!yaw_rad || !mag_body || dt_s <= 0.0f || alpha < 0.0f || alpha > 1.0f)
    {
        return -1;
    }

    if (nav_quat_from_euler(roll_rad, pitch_rad, *yaw_rad, q) != 0)
    {
        return -1;
    }
    if (nav_quat_complementary_marg(q, gyro, NULL, mag_body, mag_ref_nav, dt_s, 0.0f,
                                    1.0f - alpha) != 0)
    {
        return -1;
    }
    if (nav_quat_to_euler(q, NULL, NULL, yaw_rad) != 0)
    {
        return -1;
    }
    *yaw_rad = nav_wrap_pi(*yaw_rad);

    return 0;
}

float nav_accel_norm(const float accel_m_s2[3])
{
    if (!accel_m_s2)
    {
        return 0.0f;
    }

    return vecnorm(accel_m_s2, 3);
}

int nav_accel_gravity_gate(const float accel_m_s2[3], float tolerance_m_s2, float* norm_m_s2)
{
    float norm;

    if (!accel_m_s2 || tolerance_m_s2 < 0.0f)
    {
        return -1;
    }

    norm = nav_accel_norm(accel_m_s2);
    if (norm_m_s2)
    {
        *norm_m_s2 = norm;
    }

    return fabsf(norm - GRAVITY) <= tolerance_m_s2 ? 1 : 0;
}

void nav_remove_gravity_body(const float accel_body_m_s2[3], float roll_rad, float pitch_rad,
                             float yaw_rad, float linear_accel_body_m_s2[3])
{
    float q[4];

    if (nav_quat_from_euler(roll_rad, pitch_rad, yaw_rad, q) != 0)
    {
        return;
    }
    (void)nav_remove_gravity_body_quat(accel_body_m_s2, q, linear_accel_body_m_s2);
}

int nav_zupt_static_gate(const float accel_m_s2[3], const float gyro_rad_s[3],
                         float accel_tolerance_m_s2, float gyro_threshold_rad_s,
                         float* accel_norm_m_s2, float* gyro_norm_rad_s)
{
    float accel_norm;
    float gyro_norm;

    if (!accel_m_s2 || !gyro_rad_s || accel_tolerance_m_s2 < 0.0f || gyro_threshold_rad_s < 0.0f)
    {
        return -1;
    }

    accel_norm = nav_accel_norm(accel_m_s2);
    gyro_norm  = vecnorm(gyro_rad_s, 3);

    if (accel_norm_m_s2)
    {
        *accel_norm_m_s2 = accel_norm;
    }
    if (gyro_norm_rad_s)
    {
        *gyro_norm_rad_s = gyro_norm;
    }

    return fabsf(accel_norm - GRAVITY) <= accel_tolerance_m_s2 &&
                   gyro_norm <= gyro_threshold_rad_s
               ? 1
               : 0;
}

static int nav_vec3_normalize_copy(const float in[3], float out[3])
{
    float norm;

    if (!in || !out)
    {
        return -1;
    }

    norm = vecnorm(in, 3);
    if (norm <= 1.0e-12f)
    {
        return -1;
    }

    out[0] = in[0] / norm;
    out[1] = in[1] / norm;
    out[2] = in[2] / norm;

    return 0;
}

static void nav_vec3_cross(const float a[3], const float b[3], float out[3])
{
    out[0] = a[1] * b[2] - a[2] * b[1];
    out[1] = a[2] * b[0] - a[0] * b[2];
    out[2] = a[0] * b[1] - a[1] * b[0];
}

static void nav_vec3_accumulate_scaled(float acc[3], const float x[3], float scale)
{
    acc[0] += scale * x[0];
    acc[1] += scale * x[1];
    acc[2] += scale * x[2];
}

/* @} */
