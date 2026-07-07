#include "apriltag_pose_miniblas.h"

#include <math.h>
#include <string.h>

#include "linalg.h"

static int pose_has_expected_shape(const apriltag_pose_t* pose)
{
    return pose && pose->R && pose->t && pose->R->nrows == 3 && pose->R->ncols == 3 &&
           pose->t->nrows == 3 && pose->t->ncols == 1;
}

static int quat_normalize(float q[4])
{
    float norm;

    if (!q)
    {
        return -1;
    }

    norm = vecnorm(q, 4);
    if (norm <= 1.0e-12f || !isfinite(norm))
    {
        return -1;
    }

    for (int i = 0; i < 4; ++i)
    {
        q[i] /= norm;
    }
    return 0;
}

static int quat_from_colmajor_rotation(const float R[9], float q[4])
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

    return quat_normalize(q);
}

int apriltag_pose_to_miniblas(const apriltag_pose_t* pose, apriltag_pose_miniblas_t* out)
{
    float RtR[9];
    float I[9];
    float diff[9];

    if (!pose_has_expected_shape(pose) || !out)
    {
        return -1;
    }

    memset(out, 0, sizeof(*out));

    for (int row = 0; row < 3; ++row)
    {
        for (int col = 0; col < 3; ++col)
        {
            MAT_ELEM(out->R_tag2camera, row, col, 3, 3) =
                (float)MATD_EL(pose->R, row, col);
        }
        out->t_camera[row] = (float)MATD_EL(pose->t, row, 0);
    }

    out->rotation_det     = mat3det(out->R_tag2camera);
    out->translation_norm = vecnorm(out->t_camera, 3);

    matmul("T", "N", 3, 3, 3, 1.0f, out->R_tag2camera, out->R_tag2camera, 0.0f, RtR);
    mateye(I, 3);
    for (int i = 0; i < 9; ++i)
    {
        diff[i] = RtR[i] - I[i];
    }
    out->rotation_orthogonality_error = vecnorm(diff, 9);

    return quat_from_colmajor_rotation(out->R_tag2camera, out->q_tag2camera);
}

int apriltag_pose_to_fusion_measurement(const apriltag_pose_t* pose, float position_camera[3],
                                        float q_tag2camera[4])
{
    apriltag_pose_miniblas_t converted;

    if (!position_camera || !q_tag2camera ||
        apriltag_pose_to_miniblas(pose, &converted) != 0)
    {
        return -1;
    }

    memcpy(position_camera, converted.t_camera, sizeof(converted.t_camera));
    memcpy(q_tag2camera, converted.q_tag2camera, sizeof(converted.q_tag2camera));
    return 0;
}

int apriltag_map_localize_camera(const apriltag_pose_miniblas_t* observed_tag_in_camera,
                                 const apriltag_map_entry_t* tag_in_map,
                                 apriltag_map_observation_t* out)
{
    float tag_camera_in_map[3];

    if (!observed_tag_in_camera || !tag_in_map || !out)
    {
        return -1;
    }

    memset(out, 0, sizeof(*out));
    out->tag_id       = tag_in_map->id;
    out->tag_distance = observed_tag_in_camera->translation_norm;

    matmul("N", "T", 3, 3, 3, 1.0f, tag_in_map->R_tag2map,
           observed_tag_in_camera->R_tag2camera, 0.0f, out->R_camera2map);
    matvec("N", 3, 3, 1.0f, out->R_camera2map, observed_tag_in_camera->t_camera, 0.0f,
           tag_camera_in_map);

    for (int i = 0; i < 3; ++i)
    {
        out->t_camera_map[i] = tag_in_map->t_tag_map[i] - tag_camera_in_map[i];
    }

    return quat_from_colmajor_rotation(out->R_camera2map, out->q_camera2map);
}

int apriltag_pose_to_map_observation(const apriltag_pose_t* pose,
                                     const apriltag_map_entry_t* tag_in_map,
                                     apriltag_map_observation_t* out)
{
    apriltag_pose_miniblas_t observed;

    if (apriltag_pose_to_miniblas(pose, &observed) != 0)
    {
        return -1;
    }

    return apriltag_map_localize_camera(&observed, tag_in_map, out);
}
