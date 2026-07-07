/** @file frame_transform.c
 * KFCore
 *
 * @brief Coordinate-frame transform helpers.
 * @{ */

#include "frame_transform.h"

#include <string.h>

#include "linalg.h"

static const float FRAME_ENU_TO_NED[9] = {
    0.0f, 1.0f, 0.0f,
    1.0f, 0.0f, 0.0f,
    0.0f, 0.0f, -1.0f
};

static const float FRAME_FLIP_Y[9] = {
    1.0f, 0.0f, 0.0f,
    0.0f, -1.0f, 0.0f,
    0.0f, 0.0f, 1.0f
};

void frame_pose_identity(float R[9], float t[3])
{
    if (R)
    {
        mateye(R, 3);
    }
    if (t)
    {
        t[0] = 0.0f;
        t[1] = 0.0f;
        t[2] = 0.0f;
    }
}

int frame_transform_point(const float R_a2b[9], const float t_a_in_b[3],
                          const float p_a[3], float p_b[3])
{
    if (!R_a2b || !t_a_in_b || !p_a || !p_b)
    {
        return -1;
    }

    matvec("N", 3, 3, 1.0f, R_a2b, p_a, 0.0f, p_b);
    for (int i = 0; i < 3; ++i)
    {
        p_b[i] += t_a_in_b[i];
    }

    return 0;
}

int frame_inverse_transform_point(const float R_a2b[9], const float t_a_in_b[3],
                                  const float p_b[3], float p_a[3])
{
    float shifted[3];

    if (!R_a2b || !t_a_in_b || !p_b || !p_a)
    {
        return -1;
    }

    for (int i = 0; i < 3; ++i)
    {
        shifted[i] = p_b[i] - t_a_in_b[i];
    }
    matvec("T", 3, 3, 1.0f, R_a2b, shifted, 0.0f, p_a);

    return 0;
}

int frame_rotate_vector(const float R_a2b[9], const float v_a[3], float v_b[3])
{
    if (!R_a2b || !v_a || !v_b)
    {
        return -1;
    }

    matvec("N", 3, 3, 1.0f, R_a2b, v_a, 0.0f, v_b);
    return 0;
}

int frame_inverse_rotate_vector(const float R_a2b[9], const float v_b[3], float v_a[3])
{
    if (!R_a2b || !v_b || !v_a)
    {
        return -1;
    }

    matvec("T", 3, 3, 1.0f, R_a2b, v_b, 0.0f, v_a);
    return 0;
}

int frame_pose_inverse(const float R_a2b[9], const float t_a_in_b[3],
                       float R_b2a[9], float t_b_in_a[3])
{
    float neg_t[3];

    if (!R_a2b || !t_a_in_b || !R_b2a || !t_b_in_a)
    {
        return -1;
    }

    for (int row = 0; row < 3; ++row)
    {
        for (int col = 0; col < 3; ++col)
        {
            MAT_ELEM(R_b2a, row, col, 3, 3) = MAT_ELEM(R_a2b, col, row, 3, 3);
        }
        neg_t[row] = -t_a_in_b[row];
    }

    matvec("T", 3, 3, 1.0f, R_a2b, neg_t, 0.0f, t_b_in_a);
    return 0;
}

int frame_pose_compose(const float R_a2b[9], const float t_a_in_b[3],
                       const float R_b2c[9], const float t_b_in_c[3],
                       float R_a2c[9], float t_a_in_c[3])
{
    if (!R_a2b || !t_a_in_b || !R_b2c || !t_b_in_c || !R_a2c || !t_a_in_c)
    {
        return -1;
    }

    matmul("N", "N", 3, 3, 3, 1.0f, R_b2c, R_a2b, 0.0f, R_a2c);
    matvec("N", 3, 3, 1.0f, R_b2c, t_a_in_b, 0.0f, t_a_in_c);
    for (int i = 0; i < 3; ++i)
    {
        t_a_in_c[i] += t_b_in_c[i];
    }

    return 0;
}

int frame_transform_covariance3(const float R_a2b[9], const float P_a[9], float P_b[9])
{
    float tmp[9];

    if (!R_a2b || !P_a || !P_b)
    {
        return -1;
    }

    matmul("N", "N", 3, 3, 3, 1.0f, R_a2b, P_a, 0.0f, tmp);
    matmul("N", "T", 3, 3, 3, 1.0f, tmp, R_a2b, 0.0f, P_b);
    return 0;
}

void frame_enu_to_ned_point(const float p_enu[3], float p_ned[3])
{
    if (!p_enu || !p_ned)
    {
        return;
    }

    p_ned[0] = p_enu[1];
    p_ned[1] = p_enu[0];
    p_ned[2] = -p_enu[2];
}

void frame_ned_to_enu_point(const float p_ned[3], float p_enu[3])
{
    frame_enu_to_ned_point(p_ned, p_enu);
}

int frame_enu_to_ned_rotation(const float R_enu[9], float R_ned[9])
{
    float tmp[9];

    if (!R_enu || !R_ned)
    {
        return -1;
    }

    matmul("N", "N", 3, 3, 3, 1.0f, FRAME_ENU_TO_NED, R_enu, 0.0f, tmp);
    matmul("N", "T", 3, 3, 3, 1.0f, tmp, FRAME_ENU_TO_NED, 0.0f, R_ned);
    return 0;
}

int frame_ned_to_enu_rotation(const float R_ned[9], float R_enu[9])
{
    return frame_enu_to_ned_rotation(R_ned, R_enu);
}

void frame_handedness_flip_y_point(const float p_in[3], float p_out[3])
{
    if (!p_in || !p_out)
    {
        return;
    }

    p_out[0] = p_in[0];
    p_out[1] = -p_in[1];
    p_out[2] = p_in[2];
}

int frame_handedness_flip_y_rotation(const float R_in[9], float R_out[9])
{
    float tmp[9];

    if (!R_in || !R_out)
    {
        return -1;
    }

    matmul("N", "N", 3, 3, 3, 1.0f, FRAME_FLIP_Y, R_in, 0.0f, tmp);
    matmul("N", "N", 3, 3, 3, 1.0f, tmp, FRAME_FLIP_Y, 0.0f, R_out);
    return 0;
}

/* @} */
