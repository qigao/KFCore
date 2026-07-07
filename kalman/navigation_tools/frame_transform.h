/** @file frame_transform.h
 * KFCore
 *
 * @brief Coordinate-frame transform helpers.
 *
 * All matrices are 3x3 float column-major matrices. A pose named R_a2b,
 * t_a_in_b maps points from frame A into frame B:
 *
 *   p_b = R_a2b * p_a + t_a_in_b
 *
 * The translation is the origin of frame A expressed in frame B.
 * @{ */

#pragma once

#ifdef __cplusplus
extern "C"
{
#endif

/** @brief Set R to identity and t to zero. Any output pointer may be NULL. */
void frame_pose_identity(float R[9], float t[3]);

/** @brief Transform a point from frame A to frame B. */
int frame_transform_point(const float R_a2b[9], const float t_a_in_b[3],
                          const float p_a[3], float p_b[3]);

/** @brief Transform a point from frame B back to frame A using inverse pose. */
int frame_inverse_transform_point(const float R_a2b[9], const float t_a_in_b[3],
                                  const float p_b[3], float p_a[3]);

/** @brief Rotate a vector from frame A to frame B without translation. */
int frame_rotate_vector(const float R_a2b[9], const float v_a[3], float v_b[3]);

/** @brief Rotate a vector from frame B back to frame A using R'. */
int frame_inverse_rotate_vector(const float R_a2b[9], const float v_b[3], float v_a[3]);

/** @brief Invert pose A->B into B->A. */
int frame_pose_inverse(const float R_a2b[9], const float t_a_in_b[3],
                       float R_b2a[9], float t_b_in_a[3]);

/** @brief Compose A->B and B->C into A->C. */
int frame_pose_compose(const float R_a2b[9], const float t_a_in_b[3],
                       const float R_b2c[9], const float t_b_in_c[3],
                       float R_a2c[9], float t_a_in_c[3]);

/** @brief Transform a 3x3 covariance: P_b = R_a2b * P_a * R_a2b'. */
int frame_transform_covariance3(const float R_a2b[9], const float P_a[9], float P_b[9]);

/** @brief Convert an ENU vector to NED. */
void frame_enu_to_ned_point(const float p_enu[3], float p_ned[3]);

/** @brief Convert a NED vector to ENU. Same operation as ENU->NED. */
void frame_ned_to_enu_point(const float p_ned[3], float p_enu[3]);

/** @brief Convert an ENU-frame rotation matrix into NED convention.
 *
 * R_ned = C * R_enu * C', where C maps ENU vectors to NED vectors:
 * [north, east, down] = [enu_y, enu_x, -enu_z].
 */
int frame_enu_to_ned_rotation(const float R_enu[9], float R_ned[9]);

/** @brief Convert a NED-frame rotation matrix into ENU convention. */
int frame_ned_to_enu_rotation(const float R_ned[9], float R_enu[9]);

/** @brief Convert a vector between handedness conventions by flipping Y. */
void frame_handedness_flip_y_point(const float p_in[3], float p_out[3]);

/** @brief Convert rotation between handedness conventions by flipping Y.
 *
 * R_out = S * R_in * S, S = diag(1, -1, 1). This is intended for changing
 * coordinate conventions, not for applying a physical rotation.
 */
int frame_handedness_flip_y_rotation(const float R_in[9], float R_out[9]);

#ifdef __cplusplus
}
#endif

/* @} */
