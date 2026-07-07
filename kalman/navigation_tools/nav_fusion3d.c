/** @file nav_fusion3d.c
 * KFCore
 *
 * @brief Lightweight 3D quaternion error-state navigation fusion layer.
 * @{ */

#include <math.h>
#include <string.h>

#include "kalman_takasu.h"
#include "linalg.h"
#include "nav_fusion3d.h"
#include "navtoolbox.h"

static void fusion3d_set_identity(float* A, int n)
{
    mateye(A, n);
}

static void fusion3d_skew(const float v[3], float S[9])
{
    S[0] = 0.0f;
    S[1] = v[2];
    S[2] = -v[1];
    S[3] = -v[2];
    S[4] = 0.0f;
    S[5] = v[0];
    S[6] = v[1];
    S[7] = -v[0];
    S[8] = 0.0f;
}

static void fusion3d_inject(nav_fusion3d* fusion, const float dx[NAV_FUSION3D_ERROR_SIZE])
{
    float dq[4];
    float q_new[4];

    for (int i = 0; i < 3; ++i)
    {
        fusion->p[i] += dx[NAV_FUSION3D_DP + i];
        fusion->v[i] += dx[NAV_FUSION3D_DV + i];
        fusion->ba[i] += dx[NAV_FUSION3D_DBA + i];
        fusion->bg[i] += dx[NAV_FUSION3D_DBG + i];
    }

    dq[0] = 1.0f;
    dq[1] = 0.5f * dx[NAV_FUSION3D_DTHETA + 0];
    dq[2] = 0.5f * dx[NAV_FUSION3D_DTHETA + 1];
    dq[3] = 0.5f * dx[NAV_FUSION3D_DTHETA + 2];
    (void)nav_quat_normalize(dq);
    nav_quat_multiply(dq, fusion->q, q_new);
    for (int i = 0; i < 4; ++i)
    {
        fusion->q[i] = q_new[i];
    }
    (void)nav_quat_normalize(fusion->q);
}

static int fusion3d_update_error(nav_fusion3d* fusion, const float* dz, const float* R,
                                 const float* Ht, int m, float chi2_threshold, float* chi2)
{
    float dx[NAV_FUSION3D_ERROR_SIZE] = { 0.0f };
    int   ret;

    ret = kalman_takasu(dx, fusion->P, dz, R, Ht, NAV_FUSION3D_ERROR_SIZE, m, chi2_threshold,
                        chi2);
    if (ret == 0)
    {
        fusion3d_inject(fusion, dx);
    }

    return ret;
}

int nav_fusion3d_init(nav_fusion3d* fusion, const float p[3], const float v[3], const float q[4],
                      const float ba[3], const float bg[3],
                      const float initial_std[NAV_FUSION3D_ERROR_SIZE])
{
    if (!fusion)
    {
        return -1;
    }

    for (int i = 0; i < 3; ++i)
    {
        fusion->p[i]  = p ? p[i] : 0.0f;
        fusion->v[i]  = v ? v[i] : 0.0f;
        fusion->ba[i] = ba ? ba[i] : 0.0f;
        fusion->bg[i] = bg ? bg[i] : 0.0f;
    }
    if (q)
    {
        for (int i = 0; i < 4; ++i)
        {
            fusion->q[i] = q[i];
        }
        if (nav_quat_normalize(fusion->q) != 0)
        {
            return -1;
        }
    }
    else
    {
        nav_quat_identity(fusion->q);
    }

    memset(fusion->P, 0, sizeof(fusion->P));
    for (int i = 0; i < NAV_FUSION3D_ERROR_SIZE; ++i)
    {
        const float stddev = initial_std ? initial_std[i] : 1.0f;
        MAT_ELEM(fusion->P, i, i, NAV_FUSION3D_ERROR_SIZE, NAV_FUSION3D_ERROR_SIZE) =
            stddev * stddev;
    }

    return 0;
}

int nav_fusion3d_predict_imu(nav_fusion3d* fusion, const float accel_body_m_s2[3],
                             const float gyro_rad_s[3], float dt_s,
                             const float process_var_diag[NAV_FUSION3D_ERROR_SIZE])
{
    float accel_corr[3];
    float gyro_corr[3];
    float accel_nav_raw[3];
    float accel_nav[3];
    float gravity_nav[3] = { 0.0f, 0.0f, -GRAVITY };
    float Rbn[9];
    float skew_accel[9];
    float R_skew[9];
    float Phi[NAV_FUSION3D_ERROR_SIZE * NAV_FUSION3D_ERROR_SIZE];
    float G[NAV_FUSION3D_ERROR_SIZE * NAV_FUSION3D_ERROR_SIZE];

    if (!fusion || !accel_body_m_s2 || !gyro_rad_s || dt_s <= 0.0f)
    {
        return -1;
    }

    for (int i = 0; i < 3; ++i)
    {
        accel_corr[i] = accel_body_m_s2[i] - fusion->ba[i];
        gyro_corr[i]  = gyro_rad_s[i] - fusion->bg[i];
    }

    if (nav_quat_integrate_gyro(fusion->q, gyro_corr, dt_s) != 0)
    {
        return -1;
    }
    if (nav_quat_rotate_body_to_nav(fusion->q, accel_corr, accel_nav_raw) != 0)
    {
        return -1;
    }
    for (int i = 0; i < 3; ++i)
    {
        accel_nav[i] = accel_nav_raw[i] - gravity_nav[i];
        fusion->p[i] += fusion->v[i] * dt_s + 0.5f * accel_nav[i] * dt_s * dt_s;
        fusion->v[i] += accel_nav[i] * dt_s;
    }

    fusion3d_set_identity(Phi, NAV_FUSION3D_ERROR_SIZE);
    for (int i = 0; i < 3; ++i)
    {
        MAT_ELEM(Phi, NAV_FUSION3D_DP + i, NAV_FUSION3D_DV + i, NAV_FUSION3D_ERROR_SIZE,
                 NAV_FUSION3D_ERROR_SIZE) = dt_s;
        MAT_ELEM(Phi, NAV_FUSION3D_DTHETA + i, NAV_FUSION3D_DBG + i,
                 NAV_FUSION3D_ERROR_SIZE, NAV_FUSION3D_ERROR_SIZE) = -dt_s;
    }

    if (nav_quat_to_matrix_body2nav(fusion->q, Rbn) != 0)
    {
        return -1;
    }
    fusion3d_skew(accel_corr, skew_accel);
    matmul("N", "N", 3, 3, 3, 1.0f, Rbn, skew_accel, 0.0f, R_skew);
    for (int row = 0; row < 3; ++row)
    {
        for (int col = 0; col < 3; ++col)
        {
            MAT_ELEM(Phi, NAV_FUSION3D_DV + row, NAV_FUSION3D_DTHETA + col,
                     NAV_FUSION3D_ERROR_SIZE, NAV_FUSION3D_ERROR_SIZE) =
                -MAT_ELEM(R_skew, row, col, 3, 3) * dt_s;
            MAT_ELEM(Phi, NAV_FUSION3D_DV + row, NAV_FUSION3D_DBA + col,
                     NAV_FUSION3D_ERROR_SIZE, NAV_FUSION3D_ERROR_SIZE) =
                -MAT_ELEM(Rbn, row, col, 3, 3) * dt_s;
        }
    }

    if (process_var_diag)
    {
        fusion3d_set_identity(G, NAV_FUSION3D_ERROR_SIZE);
        kalman_predict(NULL, fusion->P, Phi, G, process_var_diag, NAV_FUSION3D_ERROR_SIZE,
                       NAV_FUSION3D_ERROR_SIZE);
    }
    else
    {
        kalman_predict(NULL, fusion->P, Phi, NULL, NULL, NAV_FUSION3D_ERROR_SIZE, 0);
    }

    return 0;
}

int nav_fusion3d_update_position(nav_fusion3d* fusion, const float position_nav[3],
                                 const float R_pos[9], float chi2_threshold, float* chi2)
{
    float dz[3];
    float Ht[NAV_FUSION3D_ERROR_SIZE * 3] = { 0.0f };

    if (!fusion || !position_nav || !R_pos)
    {
        return -1;
    }

    for (int i = 0; i < 3; ++i)
    {
        dz[i] = position_nav[i] - fusion->p[i];
        MAT_ELEM(Ht, NAV_FUSION3D_DP + i, i, NAV_FUSION3D_ERROR_SIZE, 3) = 1.0f;
    }

    return fusion3d_update_error(fusion, dz, R_pos, Ht, 3, chi2_threshold, chi2);
}

int nav_fusion3d_update_velocity(nav_fusion3d* fusion, const float velocity_nav[3],
                                 const float R_vel[9], float chi2_threshold, float* chi2)
{
    float dz[3];
    float Ht[NAV_FUSION3D_ERROR_SIZE * 3] = { 0.0f };

    if (!fusion || !velocity_nav || !R_vel)
    {
        return -1;
    }

    for (int i = 0; i < 3; ++i)
    {
        dz[i] = velocity_nav[i] - fusion->v[i];
        MAT_ELEM(Ht, NAV_FUSION3D_DV + i, i, NAV_FUSION3D_ERROR_SIZE, 3) = 1.0f;
    }

    return fusion3d_update_error(fusion, dz, R_vel, Ht, 3, chi2_threshold, chi2);
}

int nav_fusion3d_update_zupt(nav_fusion3d* fusion, const float R_vel[9], float chi2_threshold,
                             float* chi2)
{
    const float zero_velocity[3] = { 0.0f, 0.0f, 0.0f };

    return nav_fusion3d_update_velocity(fusion, zero_velocity, R_vel, chi2_threshold, chi2);
}

int nav_fusion3d_update_attitude(nav_fusion3d* fusion, const float q_body2nav_meas[4],
                                 const float R_att[9], float chi2_threshold, float* chi2)
{
    float q_meas[4];
    float q_nom_conj[4];
    float q_err[4];
    float dz[3];
    float Ht[NAV_FUSION3D_ERROR_SIZE * 3] = { 0.0f };

    if (!fusion || !q_body2nav_meas || !R_att)
    {
        return -1;
    }

    for (int i = 0; i < 4; ++i)
    {
        q_meas[i] = q_body2nav_meas[i];
    }
    if (nav_quat_normalize(q_meas) != 0)
    {
        return -1;
    }

    nav_quat_conjugate(fusion->q, q_nom_conj);
    nav_quat_multiply(q_meas, q_nom_conj, q_err);
    if (q_err[0] < 0.0f)
    {
        for (int i = 0; i < 4; ++i)
        {
            q_err[i] = -q_err[i];
        }
    }
    dz[0] = 2.0f * q_err[1];
    dz[1] = 2.0f * q_err[2];
    dz[2] = 2.0f * q_err[3];

    for (int i = 0; i < 3; ++i)
    {
        MAT_ELEM(Ht, NAV_FUSION3D_DTHETA + i, i, NAV_FUSION3D_ERROR_SIZE, 3) = 1.0f;
    }

    return fusion3d_update_error(fusion, dz, R_att, Ht, 3, chi2_threshold, chi2);
}

/* @} */
