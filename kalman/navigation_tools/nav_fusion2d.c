/** @file nav_fusion2d.c
 * KFCore
 *
 * @brief Lightweight 2D navigation fusion system layer.
 * @{ */

#include <math.h>
#include <string.h>

#include "kalman_takasu.h"
#include "linalg.h"
#include "nav_fusion2d.h"
#include "navtoolbox.h"

static void fusion2d_identity(float* A, int n)
{
    mateye(A, n);
}

int nav_fusion2d_init(nav_fusion2d* fusion, const float initial_state[NAV_FUSION2D_STATE_SIZE],
                      const float initial_std[NAV_FUSION2D_STATE_SIZE])
{
    if (!fusion)
    {
        return -1;
    }

    for (int i = 0; i < NAV_FUSION2D_STATE_SIZE; ++i)
    {
        fusion->x[i] = initial_state ? initial_state[i] : 0.0f;
    }

    memset(fusion->P, 0, sizeof(fusion->P));
    for (int i = 0; i < NAV_FUSION2D_STATE_SIZE; ++i)
    {
        const float stddev = initial_std ? initial_std[i] : 1.0f;
        MAT_ELEM(fusion->P, i, i, NAV_FUSION2D_STATE_SIZE, NAV_FUSION2D_STATE_SIZE) =
            stddev * stddev;
    }

    fusion->x[NAV_FUSION2D_YAW] = nav_wrap_pi(fusion->x[NAV_FUSION2D_YAW]);

    return 0;
}

int nav_fusion2d_predict_imu(nav_fusion2d* fusion, const float accel_body_m_s2[2],
                             float gyro_z_rad_s, float dt_s,
                             const float process_var_diag[NAV_FUSION2D_STATE_SIZE])
{
    float Phi[NAV_FUSION2D_STATE_SIZE * NAV_FUSION2D_STATE_SIZE];
    float G[NAV_FUSION2D_STATE_SIZE * NAV_FUSION2D_STATE_SIZE];
    float ax;
    float ay;
    float yaw;
    float cy;
    float sy;
    float an_x;
    float an_y;

    if (!fusion || !accel_body_m_s2 || dt_s <= 0.0f)
    {
        return -1;
    }

    ax  = accel_body_m_s2[0] - fusion->x[NAV_FUSION2D_BAX];
    ay  = accel_body_m_s2[1] - fusion->x[NAV_FUSION2D_BAY];
    yaw = fusion->x[NAV_FUSION2D_YAW];
    cy  = cosf(yaw);
    sy  = sinf(yaw);

    an_x = cy * ax - sy * ay;
    an_y = sy * ax + cy * ay;

    fusion->x[NAV_FUSION2D_X] += fusion->x[NAV_FUSION2D_VX] * dt_s + 0.5f * an_x * dt_s * dt_s;
    fusion->x[NAV_FUSION2D_Y] += fusion->x[NAV_FUSION2D_VY] * dt_s + 0.5f * an_y * dt_s * dt_s;
    fusion->x[NAV_FUSION2D_VX] += an_x * dt_s;
    fusion->x[NAV_FUSION2D_VY] += an_y * dt_s;
    fusion->x[NAV_FUSION2D_YAW] =
        nav_wrap_pi(fusion->x[NAV_FUSION2D_YAW] +
                    (gyro_z_rad_s - fusion->x[NAV_FUSION2D_BGZ]) * dt_s);

    fusion2d_identity(Phi, NAV_FUSION2D_STATE_SIZE);
    MAT_ELEM(Phi, NAV_FUSION2D_X, NAV_FUSION2D_VX, NAV_FUSION2D_STATE_SIZE,
             NAV_FUSION2D_STATE_SIZE) = dt_s;
    MAT_ELEM(Phi, NAV_FUSION2D_Y, NAV_FUSION2D_VY, NAV_FUSION2D_STATE_SIZE,
             NAV_FUSION2D_STATE_SIZE) = dt_s;
    MAT_ELEM(Phi, NAV_FUSION2D_VX, NAV_FUSION2D_YAW, NAV_FUSION2D_STATE_SIZE,
             NAV_FUSION2D_STATE_SIZE) = (-sy * ax - cy * ay) * dt_s;
    MAT_ELEM(Phi, NAV_FUSION2D_VY, NAV_FUSION2D_YAW, NAV_FUSION2D_STATE_SIZE,
             NAV_FUSION2D_STATE_SIZE) = (cy * ax - sy * ay) * dt_s;
    MAT_ELEM(Phi, NAV_FUSION2D_VX, NAV_FUSION2D_BAX, NAV_FUSION2D_STATE_SIZE,
             NAV_FUSION2D_STATE_SIZE) = -cy * dt_s;
    MAT_ELEM(Phi, NAV_FUSION2D_VX, NAV_FUSION2D_BAY, NAV_FUSION2D_STATE_SIZE,
             NAV_FUSION2D_STATE_SIZE) = sy * dt_s;
    MAT_ELEM(Phi, NAV_FUSION2D_VY, NAV_FUSION2D_BAX, NAV_FUSION2D_STATE_SIZE,
             NAV_FUSION2D_STATE_SIZE) = -sy * dt_s;
    MAT_ELEM(Phi, NAV_FUSION2D_VY, NAV_FUSION2D_BAY, NAV_FUSION2D_STATE_SIZE,
             NAV_FUSION2D_STATE_SIZE) = -cy * dt_s;
    MAT_ELEM(Phi, NAV_FUSION2D_YAW, NAV_FUSION2D_BGZ, NAV_FUSION2D_STATE_SIZE,
             NAV_FUSION2D_STATE_SIZE) = -dt_s;

    if (process_var_diag)
    {
        fusion2d_identity(G, NAV_FUSION2D_STATE_SIZE);
        kalman_predict(NULL, fusion->P, Phi, G, process_var_diag, NAV_FUSION2D_STATE_SIZE,
                       NAV_FUSION2D_STATE_SIZE);
    }
    else
    {
        kalman_predict(NULL, fusion->P, Phi, NULL, NULL, NAV_FUSION2D_STATE_SIZE, 0);
    }

    return 0;
}

int nav_fusion2d_update_position(nav_fusion2d* fusion, const float position_xy[2],
                                 const float R_pos[4], float chi2_threshold, float* chi2)
{
    float dz[2];
    float Ht[NAV_FUSION2D_STATE_SIZE * 2] = { 0.0f };

    if (!fusion || !position_xy || !R_pos)
    {
        return -1;
    }

    dz[0] = position_xy[0] - fusion->x[NAV_FUSION2D_X];
    dz[1] = position_xy[1] - fusion->x[NAV_FUSION2D_Y];
    MAT_ELEM(Ht, NAV_FUSION2D_X, 0, NAV_FUSION2D_STATE_SIZE, 2) = 1.0f;
    MAT_ELEM(Ht, NAV_FUSION2D_Y, 1, NAV_FUSION2D_STATE_SIZE, 2) = 1.0f;

    return kalman_takasu(fusion->x, fusion->P, dz, R_pos, Ht, NAV_FUSION2D_STATE_SIZE, 2,
                         chi2_threshold, chi2);
}

int nav_fusion2d_update_velocity_nav(nav_fusion2d* fusion, const float velocity_xy[2],
                                     const float R_vel[4], float chi2_threshold, float* chi2)
{
    float dz[2];
    float Ht[NAV_FUSION2D_STATE_SIZE * 2] = { 0.0f };

    if (!fusion || !velocity_xy || !R_vel)
    {
        return -1;
    }

    dz[0] = velocity_xy[0] - fusion->x[NAV_FUSION2D_VX];
    dz[1] = velocity_xy[1] - fusion->x[NAV_FUSION2D_VY];
    MAT_ELEM(Ht, NAV_FUSION2D_VX, 0, NAV_FUSION2D_STATE_SIZE, 2) = 1.0f;
    MAT_ELEM(Ht, NAV_FUSION2D_VY, 1, NAV_FUSION2D_STATE_SIZE, 2) = 1.0f;

    return kalman_takasu(fusion->x, fusion->P, dz, R_vel, Ht, NAV_FUSION2D_STATE_SIZE, 2,
                         chi2_threshold, chi2);
}

int nav_fusion2d_update_velocity_body(nav_fusion2d* fusion, const float velocity_body_xy[2],
                                      const float R_vel_body[4], float chi2_threshold,
                                      float* chi2)
{
    float dz[2];
    float Ht[NAV_FUSION2D_STATE_SIZE * 2] = { 0.0f };
    float yaw;
    float cy;
    float sy;
    float vx;
    float vy;
    float pred_bx;
    float pred_by;

    if (!fusion || !velocity_body_xy || !R_vel_body)
    {
        return -1;
    }

    yaw     = fusion->x[NAV_FUSION2D_YAW];
    cy      = cosf(yaw);
    sy      = sinf(yaw);
    vx      = fusion->x[NAV_FUSION2D_VX];
    vy      = fusion->x[NAV_FUSION2D_VY];
    pred_bx = cy * vx + sy * vy;
    pred_by = -sy * vx + cy * vy;

    dz[0] = velocity_body_xy[0] - pred_bx;
    dz[1] = velocity_body_xy[1] - pred_by;

    MAT_ELEM(Ht, NAV_FUSION2D_VX, 0, NAV_FUSION2D_STATE_SIZE, 2) = cy;
    MAT_ELEM(Ht, NAV_FUSION2D_VY, 0, NAV_FUSION2D_STATE_SIZE, 2) = sy;
    MAT_ELEM(Ht, NAV_FUSION2D_YAW, 0, NAV_FUSION2D_STATE_SIZE, 2) = -sy * vx + cy * vy;
    MAT_ELEM(Ht, NAV_FUSION2D_VX, 1, NAV_FUSION2D_STATE_SIZE, 2) = -sy;
    MAT_ELEM(Ht, NAV_FUSION2D_VY, 1, NAV_FUSION2D_STATE_SIZE, 2) = cy;
    MAT_ELEM(Ht, NAV_FUSION2D_YAW, 1, NAV_FUSION2D_STATE_SIZE, 2) = -cy * vx - sy * vy;

    return kalman_takasu(fusion->x, fusion->P, dz, R_vel_body, Ht, NAV_FUSION2D_STATE_SIZE, 2,
                         chi2_threshold, chi2);
}

int nav_fusion2d_update_yaw(nav_fusion2d* fusion, float yaw_rad, float R_yaw,
                            float chi2_threshold, float* chi2)
{
    float dz[1];
    float R[1];
    float Ht[NAV_FUSION2D_STATE_SIZE] = { 0.0f };
    int   ret;

    if (!fusion || R_yaw <= 0.0f)
    {
        return -1;
    }

    dz[0]                       = nav_wrap_pi(yaw_rad - fusion->x[NAV_FUSION2D_YAW]);
    R[0]                        = R_yaw;
    Ht[NAV_FUSION2D_YAW]        = 1.0f;
    ret = kalman_takasu(fusion->x, fusion->P, dz, R, Ht, NAV_FUSION2D_STATE_SIZE, 1,
                        chi2_threshold, chi2);
    fusion->x[NAV_FUSION2D_YAW] = nav_wrap_pi(fusion->x[NAV_FUSION2D_YAW]);

    return ret;
}

int nav_fusion2d_update_zupt(nav_fusion2d* fusion, const float R_vel[4], float chi2_threshold,
                             float* chi2)
{
    const float zero_velocity[2] = { 0.0f, 0.0f };

    return nav_fusion2d_update_velocity_nav(fusion, zero_velocity, R_vel, chi2_threshold, chi2);
}

/* @} */
