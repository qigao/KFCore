#pragma once

#ifdef __cplusplus
extern "C" {
#endif

#include "apriltag_pose.h"

/**
 * KFCore/miniblas-friendly representation of an AprilTag pose.
 *
 * AprilTag stores pose matrices as double-precision row-major matd_t values.
 * miniblas/linalg and KFCore navigation helpers use float column-major matrices.
 */
typedef struct
{
    float R_tag2camera[9];               /**< 3x3 column-major rotation matrix. */
    float t_camera[3];                   /**< Tag origin in the camera frame. */
    float q_tag2camera[4];               /**< Quaternion [w, x, y, z] from tag to camera. */
    float rotation_det;                  /**< det(R), useful for sanity checks. */
    float rotation_orthogonality_error;  /**< Frobenius norm of R'R - I. */
    float translation_norm;              /**< Euclidean norm of t_camera. */
} apriltag_pose_miniblas_t;

/**
 * Known AprilTag landmark pose in a map/navigation frame.
 *
 * R_tag2map maps vectors from the tag frame into the map frame. t_tag_map is
 * the tag origin expressed in the map frame.
 */
typedef struct
{
    int   id;
    float size_m;
    float R_tag2map[9]; /**< 3x3 column-major rotation matrix. */
    float t_tag_map[3];
} apriltag_map_entry_t;

/**
 * Camera pose inferred from observing a mapped tag.
 *
 * R_camera2map maps camera-frame vectors into the map frame. t_camera_map is
 * the camera origin expressed in the map frame.
 */
typedef struct
{
    int   tag_id;
    float R_camera2map[9]; /**< 3x3 column-major rotation matrix. */
    float t_camera_map[3];
    float q_camera2map[4]; /**< Quaternion [w, x, y, z]. */
    float tag_distance;
} apriltag_map_observation_t;

/**
 * Convert an AprilTag pose into miniblas/KFCore-friendly arrays.
 *
 * @return 0 on success, -1 on invalid input or invalid quaternion conversion.
 */
int apriltag_pose_to_miniblas(const apriltag_pose_t* pose, apriltag_pose_miniblas_t* out);

/**
 * Convenience wrapper for fusion code that only needs position and quaternion.
 *
 * @return 0 on success, -1 on invalid input.
 */
int apriltag_pose_to_fusion_measurement(const apriltag_pose_t* pose, float position_camera[3],
                                        float q_tag2camera[4]);

/**
 * Infer camera pose in a map frame from a known tag-map entry and an observed
 * tag pose returned by AprilTag.
 *
 * Convention: observed_tag_in_camera must represent tag->camera. The map entry
 * must represent tag->map. The returned observation is camera->map.
 *
 * @return 0 on success, -1 on invalid input.
 */
int apriltag_map_localize_camera(const apriltag_pose_miniblas_t* observed_tag_in_camera,
                                 const apriltag_map_entry_t* tag_in_map,
                                 apriltag_map_observation_t* out);

/**
 * Convenience wrapper that converts AprilTag's matd pose and localizes the
 * camera against a known map tag.
 *
 * @return 0 on success, -1 on invalid input.
 */
int apriltag_pose_to_map_observation(const apriltag_pose_t* pose,
                                     const apriltag_map_entry_t* tag_in_map,
                                     apriltag_map_observation_t* out);

#ifdef __cplusplus
}
#endif
