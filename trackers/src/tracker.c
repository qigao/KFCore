#include "trackers/tracker.h"

#include <math.h>
#include <limits.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#ifdef KFCORE_TRACKERS_TEST_ALLOCATOR
#include "tracker_test_alloc.h"
#define malloc trackers_test_malloc
#define calloc trackers_test_calloc
#define realloc trackers_test_realloc
#endif

#include "kalman_takasu.h"

#define TRACKER_KALMAN_WORKSPACE_FLOATS 136
#include "linalg.h"

#define TRACKERS_EPS 1.0e-6f
#define TRACKERS_PI 3.14159265358979323846f

#define MAT_INDEX(row, col, rows) ((row) + (col) * (rows))

typedef struct assignment_result {
    int* match_rows;
    int* match_cols;
    size_t match_count;
    int* unmatched_rows;
    size_t unmatched_row_count;
    int* unmatched_cols;
    size_t unmatched_col_count;
    int status;
} assignment_result_t;

typedef struct byte_frame_scratch {
    int* high_indices;
    int* low_indices;
    box_t* track_boxes;
    box_t* high_boxes;
    box_t* remaining_boxes;
    box_t* low_boxes;
    float* high_iou;
    float* low_iou;
    int* high_row_used;
    int* high_col_used;
    int* high_match_rows;
    int* high_match_cols;
    int* high_unmatched_rows;
    int* high_unmatched_cols;
    int* low_row_used;
    int* low_col_used;
    int* low_match_rows;
    int* low_match_cols;
    int* low_unmatched_rows;
    int* low_unmatched_cols;
    void* track_snapshot;
} byte_frame_scratch_t;

typedef struct kf_xyxy {
    float x[8];
    float P[8 * 8];
    float Phi[8 * 8];
    float G[8 * 8];
    float Q[8];
    float Ht[8 * 4];
    float R[4 * 4];
} kf_xyxy_t;

typedef struct kf_xcycsr {
    float x[7];
    float P[7 * 7];
    float Phi[7 * 7];
    float G[7 * 7];
    float Q[7];
    float Ht[7 * 4];
    float R[4 * 4];
} kf_xcycsr_t;

typedef struct kf7_snapshot {
    float x[7];
    float P[7 * 7];
} kf7_snapshot_t;

typedef struct kf_xyxy_track {
    int tracker_id;
    int number_of_successful_updates;
    int time_since_update;
    kf_xyxy_t estimator;
} kf_xyxy_track_t;

typedef kf_xyxy_track_t sort_track_t;
typedef kf_xyxy_track_t byte_track_t;
typedef kf_xyxy_track_t cbiou_track_t;

typedef struct observation {
    int age;
    box_t box;
} observation_t;

typedef struct ocsort_track {
    int age;
    int tracker_id;
    int number_of_successful_updates;
    int time_since_update;
    int delta_t;
    kf_xcycsr_t estimator;
    box_t last_observation;
    observation_t* observations;
    size_t observation_count;
    size_t observation_capacity;
    float velocity[2];
    int has_velocity;
    kf7_snapshot_t frozen_state;
    int has_frozen_state;
    int observed;
} ocsort_track_t;

struct sort {
    int maximum_frames_without_update;
    int minimum_consecutive_frames;
    float minimum_iou_threshold;
    float track_activation_threshold;
    int next_id;
    sort_track_t* tracks;
    size_t track_count;
    size_t track_capacity;
};

struct bytetrack {
    int maximum_frames_without_update;
    int minimum_consecutive_frames;
    float minimum_iou_threshold;
    float track_activation_threshold;
    float high_conf_det_threshold;
    int next_id;
    byte_track_t* tracks;
    size_t track_count;
    size_t track_capacity;
};

struct cbiou {
    int maximum_frames_without_update;
    int minimum_consecutive_frames;
    float minimum_biou_threshold;
    float track_activation_threshold;
    float high_conf_det_threshold;
    float low_conf_det_threshold;
    float first_buffer_ratio;
    float second_buffer_ratio;
    int fuse_detection_score;
    int next_id;
    cbiou_track_t* tracks;
    size_t track_count;
    size_t track_capacity;
};

struct ocsort {
    int maximum_frames_without_update;
    int minimum_consecutive_frames;
    float minimum_iou_threshold;
    float direction_consistency_weight;
    float high_conf_det_threshold;
    int delta_t;
    int frame_count;
    int next_id;
    ocsort_track_t* tracks;
    size_t track_count;
    size_t track_capacity;
};

static float box_width(box_t box) {
    return box.x2 - box.x1;
}

static float box_height(box_t box) {
    return box.y2 - box.y1;
}

static float confidence_or(detection_t detection, float fallback) {
    return detection.has_confidence ? detection.confidence : fallback;
}

static int confidence_passes(detection_t detection, float threshold) {
    return !detection.has_confidence || detection.confidence >= threshold;
}

static int scaled_lost_buffer_checked(
    int lost_track_buffer,
    float frame_rate,
    int* result
) {
    const float scaled = frame_rate / 30.0f * (float)lost_track_buffer;
    const double range_checked = (double)scaled;
    if (!result || !isfinite(scaled) ||
        range_checked < (double)INT_MIN || range_checked > (double)INT_MAX) {
        return 0;
    }
    *result = (int)scaled;
    return 1;
}

static void set_identity(float* matrix, int dim) {
    memset(matrix, 0, sizeof(float) * (size_t)dim * (size_t)dim);
    for (int i = 0; i < dim; ++i) {
        matrix[MAT_INDEX(i, i, dim)] = 1.0f;
    }
}

static void scale_matrix(float* matrix, size_t count, float scale) {
    for (size_t i = 0; i < count; ++i) {
        matrix[i] *= scale;
    }
}

static void scale_diagonal_block(float* matrix, int dim, int offset, int block_dim, float scale) {
    for (int i = 0; i < block_dim; ++i) {
        matrix[MAT_INDEX(offset + i, offset + i, dim)] *= scale;
    }
}

static void set_top_left_identity(float* matrix, int rows, int dim) {
    for (int i = 0; i < dim; ++i) {
        matrix[MAT_INDEX(i, i, rows)] = 1.0f;
    }
}

static void xyxy_to_xcycsr(box_t box, float out[4]) {
    const float w = box_width(box);
    const float h = box_height(box);
    out[0] = box.x1 + w * 0.5f;
    out[1] = box.y1 + h * 0.5f;
    out[2] = w * h;
    out[3] = w / (h + TRACKERS_EPS);
}

static box_t xcycsr_to_xyxy(const float state[4]) {
    const float w = sqrtf(fmaxf(0.0f, state[2] * state[3]));
    const float h = state[2] / fmaxf(w, TRACKERS_EPS);
    box_t box;
    box.x1 = state[0] - w * 0.5f;
    box.y1 = state[1] - h * 0.5f;
    box.x2 = state[0] + w * 0.5f;
    box.y2 = state[1] + h * 0.5f;
    return box;
}

static float compute_iou(box_t lhs, box_t rhs) {
    const float xx1 = fmaxf(lhs.x1, rhs.x1);
    const float yy1 = fmaxf(lhs.y1, rhs.y1);
    const float xx2 = fminf(lhs.x2, rhs.x2);
    const float yy2 = fminf(lhs.y2, rhs.y2);
    const float iw = fmaxf(0.0f, xx2 - xx1);
    const float ih = fmaxf(0.0f, yy2 - yy1);
    const float intersection = iw * ih;
    const float lhs_area = fmaxf(0.0f, box_width(lhs)) * fmaxf(0.0f, box_height(lhs));
    const float rhs_area = fmaxf(0.0f, box_width(rhs)) * fmaxf(0.0f, box_height(rhs));
    const float denom = lhs_area + rhs_area - intersection;
    return denom <= TRACKERS_EPS ? 0.0f : intersection / denom;
}

static box_t buffered_box(box_t box, float buffer_ratio) {
    const float w = box_width(box);
    const float h = box_height(box);
    const float cx = (box.x1 + box.x2) * 0.5f;
    const float cy = (box.y1 + box.y2) * 0.5f;
    const float bw = w * (1.0f + fmaxf(0.0f, buffer_ratio));
    const float bh = h * (1.0f + fmaxf(0.0f, buffer_ratio));
    box_t out;
    out.x1 = cx - bw * 0.5f;
    out.y1 = cy - bh * 0.5f;
    out.x2 = cx + bw * 0.5f;
    out.y2 = cy + bh * 0.5f;
    return out;
}

static float compute_biou(box_t lhs, box_t rhs, float buffer_ratio) {
    return compute_iou(buffered_box(lhs, buffer_ratio), buffered_box(rhs, buffer_ratio));
}

static void compute_velocity(box_t from, box_t to, float out[2]) {
    const float cx1 = (from.x1 + from.x2) * 0.5f;
    const float cy1 = (from.y1 + from.y2) * 0.5f;
    const float cx2 = (to.x1 + to.x2) * 0.5f;
    const float cy2 = (to.y1 + to.y2) * 0.5f;
    out[0] = cx2 - cx1;
    out[1] = cy2 - cy1;
    const float norm = sqrtf(out[0] * out[0] + out[1] * out[1]) + TRACKERS_EPS;
    out[0] /= norm;
    out[1] /= norm;
}

static tracker_status_t reserve_capacity(
    void** data,
    size_t* capacity,
    size_t elem_size,
    size_t needed
) {
    if (!data || !capacity || elem_size == 0) {
        return TRACKER_STATUS_INVALID_ARGUMENT;
    }
    if (*capacity >= needed) {
        return TRACKER_STATUS_OK;
    }

    size_t new_capacity = *capacity == 0 ? 4 : *capacity;
    while (new_capacity < needed) {
        if (new_capacity > SIZE_MAX / 2) {
            new_capacity = needed;
            break;
        }
        new_capacity *= 2;
    }
    if (new_capacity < needed || new_capacity > SIZE_MAX / elem_size) {
        return TRACKER_STATUS_OVERFLOW;
    }

    void* new_data = realloc(*data, elem_size * new_capacity);
    if (!new_data) {
        return TRACKER_STATUS_ALLOCATION_FAILED;
    }
    *data = new_data;
    *capacity = new_capacity;
    return TRACKER_STATUS_OK;
}

static int ensure_capacity(void** data, size_t* capacity, size_t elem_size, size_t needed) {
    return reserve_capacity(data, capacity, elem_size, needed) == TRACKER_STATUS_OK;
}

static void free_assignment(assignment_result_t* result) {
    free(result->match_rows);
    free(result->match_cols);
    free(result->unmatched_rows);
    free(result->unmatched_cols);
    memset(result, 0, sizeof(*result));
}

static assignment_result_t assign_greedy(
    const float* scores,
    size_t rows,
    size_t cols,
    float min_score
) {
    assignment_result_t result;
    memset(&result, 0, sizeof(result));
    if (rows && cols && !scores) {
        result.status = -1;
        return result;
    }
    const size_t max_matches = rows < cols ? rows : cols;
    int* row_used = rows ? calloc(rows, sizeof(int)) : NULL;
    int* col_used = cols ? calloc(cols, sizeof(int)) : NULL;
    result.match_rows = max_matches ? malloc(sizeof(int) * max_matches) : NULL;
    result.match_cols = max_matches ? malloc(sizeof(int) * max_matches) : NULL;
    result.unmatched_rows = rows ? malloc(sizeof(int) * rows) : NULL;
    result.unmatched_cols = cols ? malloc(sizeof(int) * cols) : NULL;

    if ((rows && !row_used) || (cols && !col_used) ||
        (max_matches && (!result.match_rows || !result.match_cols)) ||
        (rows && !result.unmatched_rows) || (cols && !result.unmatched_cols)) {
        free(row_used);
        free(col_used);
        free_assignment(&result);
        result.status = -1;
        return result;
    }

    for (;;) {
        int best_row = -1;
        int best_col = -1;
        float best_score = -INFINITY;
        for (size_t row = 0; row < rows; ++row) {
            if (row_used[row]) {
                continue;
            }
            for (size_t col = 0; col < cols; ++col) {
                if (col_used[col]) {
                    continue;
                }
                const float score = scores[row * cols + col];
                if (score > best_score) {
                    best_score = score;
                    best_row = (int)row;
                    best_col = (int)col;
                }
            }
        }
        if (best_row < 0 || best_col < 0 || best_score < min_score) {
            break;
        }
        row_used[best_row] = 1;
        col_used[best_col] = 1;
        result.match_rows[result.match_count] = best_row;
        result.match_cols[result.match_count] = best_col;
        ++result.match_count;
    }

    for (size_t row = 0; row < rows; ++row) {
        if (!row_used[row]) {
            result.unmatched_rows[result.unmatched_row_count++] = (int)row;
        }
    }
    for (size_t col = 0; col < cols; ++col) {
        if (!col_used[col]) {
            result.unmatched_cols[result.unmatched_col_count++] = (int)col;
        }
    }

    for (size_t i = 1; i < result.match_count; ++i) {
        const int row = result.match_rows[i];
        const int col = result.match_cols[i];
        size_t j = i;
        while (j > 0 && result.match_rows[j - 1] > row) {
            result.match_rows[j] = result.match_rows[j - 1];
            result.match_cols[j] = result.match_cols[j - 1];
            --j;
        }
        result.match_rows[j] = row;
        result.match_cols[j] = col;
    }

    free(row_used);
    free(col_used);
    return result;
}

static void assign_greedy_into(
    const float* scores,
    size_t rows,
    size_t cols,
    float min_score,
    int* row_used,
    int* col_used,
    assignment_result_t* result
) {
    result->match_count = 0;
    result->unmatched_row_count = 0;
    result->unmatched_col_count = 0;
    result->status = 0;
    if (rows) {
        memset(row_used, 0, sizeof(*row_used) * rows);
    }
    if (cols) {
        memset(col_used, 0, sizeof(*col_used) * cols);
    }

    for (;;) {
        int best_row = -1;
        int best_col = -1;
        float best_score = -INFINITY;
        for (size_t row = 0; row < rows; ++row) {
            if (row_used[row]) {
                continue;
            }
            for (size_t col = 0; col < cols; ++col) {
                if (col_used[col]) {
                    continue;
                }
                const float score = scores[row * cols + col];
                if (score > best_score) {
                    best_score = score;
                    best_row = (int)row;
                    best_col = (int)col;
                }
            }
        }
        if (best_row < 0 || best_col < 0 || best_score < min_score) {
            break;
        }
        row_used[best_row] = 1;
        col_used[best_col] = 1;
        result->match_rows[result->match_count] = best_row;
        result->match_cols[result->match_count] = best_col;
        ++result->match_count;
    }

    for (size_t row = 0; row < rows; ++row) {
        if (!row_used[row]) {
            result->unmatched_rows[result->unmatched_row_count++] = (int)row;
        }
    }
    for (size_t col = 0; col < cols; ++col) {
        if (!col_used[col]) {
            result->unmatched_cols[result->unmatched_col_count++] = (int)col;
        }
    }

    for (size_t i = 1; i < result->match_count; ++i) {
        const int row = result->match_rows[i];
        const int col = result->match_cols[i];
        size_t j = i;
        while (j > 0 && result->match_rows[j - 1] > row) {
            result->match_rows[j] = result->match_rows[j - 1];
            result->match_cols[j] = result->match_cols[j - 1];
            --j;
        }
        result->match_rows[j] = row;
        result->match_cols[j] = col;
    }
}

static int checked_multiply_size(size_t left, size_t right, size_t* result) {
    if (left != 0 && right > SIZE_MAX / left) {
        return 0;
    }
    *result = left * right;
    return 1;
}

static int checked_add_size(size_t left, size_t right, size_t* result) {
    if (right > SIZE_MAX - left) {
        return 0;
    }
    *result = left + right;
    return 1;
}

static tracker_status_t allocate_array(void** data, size_t count, size_t element_size) {
    size_t bytes = 0;
    if (!checked_multiply_size(count, element_size, &bytes)) {
        return TRACKER_STATUS_OVERFLOW;
    }
    *data = count ? malloc(bytes) : NULL;
    return count && !*data ? TRACKER_STATUS_ALLOCATION_FAILED : TRACKER_STATUS_OK;
}

static void byte_frame_scratch_release(byte_frame_scratch_t* scratch) {
    free(scratch->high_indices);
    free(scratch->low_indices);
    free(scratch->track_boxes);
    free(scratch->high_boxes);
    free(scratch->remaining_boxes);
    free(scratch->low_boxes);
    free(scratch->high_iou);
    free(scratch->low_iou);
    free(scratch->high_row_used);
    free(scratch->high_col_used);
    free(scratch->high_match_rows);
    free(scratch->high_match_cols);
    free(scratch->high_unmatched_rows);
    free(scratch->high_unmatched_cols);
    free(scratch->low_row_used);
    free(scratch->low_col_used);
    free(scratch->low_match_rows);
    free(scratch->low_match_cols);
    free(scratch->low_unmatched_rows);
    free(scratch->low_unmatched_cols);
    free(scratch->track_snapshot);
    memset(scratch, 0, sizeof(*scratch));
}

static void gather_boxes_into(
    box_t* boxes,
    const detection_t* detections,
    const int* indices,
    size_t count
) {
    for (size_t i = 0; i < count; ++i) {
        boxes[i] = detections[indices[i]].box;
    }
}

static void build_iou_matrix_into(
    float* matrix,
    const box_t* rows,
    size_t row_count,
    const box_t* cols,
    size_t col_count
) {
    for (size_t row = 0; row < row_count; ++row) {
        for (size_t col = 0; col < col_count; ++col) {
            matrix[row * col_count + col] = compute_iou(rows[row], cols[col]);
        }
    }
}

static float* build_iou_matrix(const box_t* rows, size_t row_count,
                               const box_t* cols, size_t col_count) {
    size_t count = 0;
    if (!checked_multiply_size(row_count, col_count, &count) ||
        count > SIZE_MAX / sizeof(float)) {
        return NULL;
    }
    float* matrix = count ? malloc(sizeof(float) * count) : NULL;
    if (!matrix && count) {
        return NULL;
    }
    for (size_t row = 0; row < row_count; ++row) {
        for (size_t col = 0; col < col_count; ++col) {
            matrix[row * col_count + col] = compute_iou(rows[row], cols[col]);
        }
    }
    return matrix;
}

static float* build_biou_score_matrix(
    const box_t* rows,
    size_t row_count,
    const box_t* cols,
    const int* detection_indices,
    const detection_t* detections,
    size_t col_count,
    float buffer_ratio,
    int fuse_detection_score
) {
    size_t count = 0;
    if (!checked_multiply_size(row_count, col_count, &count) ||
        count > SIZE_MAX / sizeof(float)) {
        return NULL;
    }
    float* matrix = count ? malloc(sizeof(float) * count) : NULL;
    if (!matrix && count) {
        return NULL;
    }
    for (size_t row = 0; row < row_count; ++row) {
        for (size_t col = 0; col < col_count; ++col) {
            float score = compute_biou(rows[row], cols[col], buffer_ratio);
            if (fuse_detection_score) {
                score *= confidence_or(detections[detection_indices[col]], 1.0f);
            }
            matrix[row * col_count + col] = score;
        }
    }
    return matrix;
}

static void kf_xyxy_init(kf_xyxy_t* kf, box_t bbox) {
    memset(kf, 0, sizeof(*kf));
    set_identity(kf->P, 8);
    set_identity(kf->Phi, 8);
    set_identity(kf->G, 8);
    for (int i = 0; i < 8; ++i) {
        kf->Q[i] = 0.01f;
    }
    set_identity(kf->R, 4);
    scale_matrix(kf->R, 16, 0.1f);
    set_top_left_identity(kf->Ht, 8, 4);
    kf->Phi[MAT_INDEX(0, 4, 8)] = 1.0f;
    kf->Phi[MAT_INDEX(1, 5, 8)] = 1.0f;
    kf->Phi[MAT_INDEX(2, 6, 8)] = 1.0f;
    kf->Phi[MAT_INDEX(3, 7, 8)] = 1.0f;
    kf->x[0] = bbox.x1;
    kf->x[1] = bbox.y1;
    kf->x[2] = bbox.x2;
    kf->x[3] = bbox.y2;
}

static tracker_status_t kf_xyxy_predict(kf_xyxy_t* kf) {
    kf_xyxy_t candidate = *kf;
    float workspace[TRACKER_KALMAN_WORKSPACE_FLOATS];

#ifdef KFCORE_TRACKERS_TEST_ALLOCATOR
    if (trackers_test_kalman_should_fail()) {
        return TRACKER_STATUS_NUMERICAL_FAILURE;
    }
#endif

    if (kalman_predict(candidate.x, candidate.P, candidate.Phi, candidate.G, candidate.Q,
                       8, 8, workspace, TRACKER_KALMAN_WORKSPACE_FLOATS) != KFCORE_KALMAN_OK) {
        return TRACKER_STATUS_NUMERICAL_FAILURE;
    }

    *kf = candidate;
    return TRACKER_STATUS_OK;
}

static tracker_status_t kf_xyxy_update(kf_xyxy_t* kf, box_t bbox) {
    kf_xyxy_t candidate = *kf;
    float z[4] = {bbox.x1, bbox.y1, bbox.x2, bbox.y2};
    float dz[4];
    float workspace[TRACKER_KALMAN_WORKSPACE_FLOATS];

#ifdef KFCORE_TRACKERS_TEST_ALLOCATOR
    if (trackers_test_kalman_should_fail()) {
        return TRACKER_STATUS_NUMERICAL_FAILURE;
    }
#endif

    memcpy(dz, z, sizeof(dz));
    matmul("T", "N", 4, 1, 8, -1.0f, candidate.Ht, candidate.x, 1.0f, dz);
    if (kalman_takasu(candidate.x, candidate.P, dz, candidate.R, candidate.Ht,
                      8, 4, 0.0f, NULL, workspace,
                      TRACKER_KALMAN_WORKSPACE_FLOATS) != KFCORE_KALMAN_OK) {
        return TRACKER_STATUS_NUMERICAL_FAILURE;
    }

    *kf = candidate;
    return TRACKER_STATUS_OK;
}

static box_t kf_xyxy_box(const kf_xyxy_t* kf) {
    box_t box = {kf->x[0], kf->x[1], kf->x[2], kf->x[3]};
    return box;
}

static void kf_xcycsr_init(kf_xcycsr_t* kf, box_t bbox) {
    memset(kf, 0, sizeof(*kf));
    set_identity(kf->P, 7);
    set_identity(kf->Phi, 7);
    set_identity(kf->G, 7);
    for (int i = 0; i < 7; ++i) {
        kf->Q[i] = 1.0f;
    }
    set_identity(kf->R, 4);
    set_top_left_identity(kf->Ht, 7, 4);
    kf->Phi[MAT_INDEX(0, 4, 7)] = 1.0f;
    kf->Phi[MAT_INDEX(1, 5, 7)] = 1.0f;
    kf->Phi[MAT_INDEX(2, 6, 7)] = 1.0f;
    scale_diagonal_block(kf->R, 4, 2, 2, 10.0f);
    scale_diagonal_block(kf->P, 7, 4, 3, 1000.0f);
    scale_matrix(kf->P, 49, 10.0f);
    kf->Q[6] *= 0.01f;
    for (int i = 4; i < 7; ++i) {
        kf->Q[i] *= 0.01f;
    }
    xyxy_to_xcycsr(bbox, kf->x);
}

static tracker_status_t kf_xcycsr_predict(kf_xcycsr_t* kf) {
    kf_xcycsr_t candidate = *kf;
    float workspace[TRACKER_KALMAN_WORKSPACE_FLOATS];

    if (candidate.x[6] + candidate.x[2] <= 0.0f) {
        candidate.x[6] = 0.0f;
    }
#ifdef KFCORE_TRACKERS_TEST_ALLOCATOR
    if (trackers_test_kalman_should_fail()) {
        return TRACKER_STATUS_NUMERICAL_FAILURE;
    }
#endif
    if (kalman_predict(candidate.x, candidate.P, candidate.Phi, candidate.G, candidate.Q,
                       7, 7, workspace, TRACKER_KALMAN_WORKSPACE_FLOATS) != KFCORE_KALMAN_OK) {
        return TRACKER_STATUS_NUMERICAL_FAILURE;
    }

    *kf = candidate;
    return TRACKER_STATUS_OK;
}

static tracker_status_t kf_xcycsr_predict_raw(kf_xcycsr_t* kf) {
    kf_xcycsr_t candidate = *kf;
    float workspace[TRACKER_KALMAN_WORKSPACE_FLOATS];

#ifdef KFCORE_TRACKERS_TEST_ALLOCATOR
    if (trackers_test_kalman_should_fail()) {
        return TRACKER_STATUS_NUMERICAL_FAILURE;
    }
#endif
    if (kalman_predict(candidate.x, candidate.P, candidate.Phi, candidate.G, candidate.Q,
                       7, 7, workspace, TRACKER_KALMAN_WORKSPACE_FLOATS) != KFCORE_KALMAN_OK) {
        return TRACKER_STATUS_NUMERICAL_FAILURE;
    }

    *kf = candidate;
    return TRACKER_STATUS_OK;
}

static tracker_status_t kf_xcycsr_update_measurement(kf_xcycsr_t* kf, const float z[4]) {
    kf_xcycsr_t candidate = *kf;
    float dz[4];
    float workspace[TRACKER_KALMAN_WORKSPACE_FLOATS];

#ifdef KFCORE_TRACKERS_TEST_ALLOCATOR
    if (trackers_test_kalman_should_fail()) {
        return TRACKER_STATUS_NUMERICAL_FAILURE;
    }
#endif
    memcpy(dz, z, sizeof(dz));
    matmul("T", "N", 4, 1, 7, -1.0f, candidate.Ht, candidate.x, 1.0f, dz);
    if (kalman_takasu(candidate.x, candidate.P, dz, candidate.R, candidate.Ht,
                      7, 4, 0.0f, NULL, workspace,
                      TRACKER_KALMAN_WORKSPACE_FLOATS) != KFCORE_KALMAN_OK) {
        return TRACKER_STATUS_NUMERICAL_FAILURE;
    }

    *kf = candidate;
    return TRACKER_STATUS_OK;
}

static tracker_status_t kf_xcycsr_update(kf_xcycsr_t* kf, box_t bbox) {
    float z[4];
    xyxy_to_xcycsr(bbox, z);
    return kf_xcycsr_update_measurement(kf, z);
}

static box_t kf_xcycsr_box(const kf_xcycsr_t* kf) {
    return xcycsr_to_xyxy(kf->x);
}

static int push_tracked(tracked_detection_t* output, size_t capacity, size_t* count,
                        detection_t detection, int tracker_id) {
    if (*count < capacity && output) {
        output[*count].detection = detection;
        output[*count].tracker_id = tracker_id;
    }
    ++(*count);
    return 1;
}

sort_config_t sort_default_config(void) {
    sort_config_t config = {30, 30.0f, 0.25f, 3, 0.3f};
    return config;
}

bytetrack_config_t bytetrack_default_config(void) {
    bytetrack_config_t config = {30, 30.0f, 0.7f, 2, 0.1f, 0.6f};
    return config;
}

cbiou_config_t cbiou_default_config(void) {
    cbiou_config_t config = {30, 30.0f, 0.7f, 2, 0.1f, 0.6f, 0.1f, 0.3f, 0.5f, 1};
    return config;
}

ocsort_config_t ocsort_default_config(void) {
    ocsort_config_t config = {30, 30.0f, 3, 0.3f, 0.2f, 0.6f, 3};
    return config;
}

sort_t* sort_create(const sort_config_t* config_in) {
    const sort_config_t config =
        config_in ? *config_in : sort_default_config();
    int maximum_frames_without_update = 0;
    if (!scaled_lost_buffer_checked(config.lost_track_buffer, config.frame_rate,
                                    &maximum_frames_without_update)) {
        return NULL;
    }
    sort_t* tracker = calloc(1, sizeof(*tracker));
    if (!tracker) {
        return NULL;
    }
    tracker->maximum_frames_without_update = maximum_frames_without_update;
    tracker->minimum_consecutive_frames = config.minimum_consecutive_frames;
    tracker->minimum_iou_threshold = config.minimum_iou_threshold;
    tracker->track_activation_threshold = config.track_activation_threshold;
    return tracker;
}

void sort_destroy(sort_t* tracker) {
    if (!tracker) {
        return;
    }
    free(tracker->tracks);
    free(tracker);
}

void sort_reset(sort_t* tracker) {
    if (!tracker) {
        return;
    }
    tracker->track_count = 0;
    tracker->next_id = 0;
}

static sort_track_t* sort_add_track(sort_t* tracker, box_t box) {
    if (!ensure_capacity((void**)&tracker->tracks, &tracker->track_capacity,
                         sizeof(tracker->tracks[0]), tracker->track_count + 1)) {
        return NULL;
    }
    sort_track_t* track = &tracker->tracks[tracker->track_count++];
    memset(track, 0, sizeof(*track));
    track->tracker_id = -1;
    track->number_of_successful_updates = 1;
    kf_xyxy_init(&track->estimator, box);
    return track;
}

static void sort_retain_alive(sort_t* tracker) {
    size_t out = 0;
    for (size_t i = 0; i < tracker->track_count; ++i) {
        sort_track_t* track = &tracker->tracks[i];
        const int mature = track->number_of_successful_updates >= tracker->minimum_consecutive_frames;
        const int active = track->time_since_update == 0;
        if (track->time_since_update < tracker->maximum_frames_without_update &&
            (mature || active)) {
            if (out != i) {
                tracker->tracks[out] = tracker->tracks[i];
            }
            ++out;
        }
    }
    tracker->track_count = out;
}

tracker_status_t sort_update_ex(
    sort_t* tracker,
    const detection_t* detections,
    size_t detection_count,
    tracked_detection_ex_t* output,
    size_t output_capacity,
    size_t* output_count
) {
    sort_track_t* snapshot = NULL;
    box_t* track_boxes = NULL;
    box_t* detection_boxes = NULL;
    float* iou = NULL;
    assignment_result_t assignments;
    tracker_status_t status = TRACKER_STATUS_OK;
    size_t saved_track_count;
    int saved_next_id;
    size_t required_capacity = 0;
    size_t iou_count = 0;

    memset(&assignments, 0, sizeof(assignments));
    if (output_count) {
        *output_count = 0;
    }
    if (!tracker || !output_count || (!detections && detection_count)) {
        return TRACKER_STATUS_INVALID_ARGUMENT;
    }
    if (detection_count > output_capacity || (detection_count && !output)) {
        return TRACKER_STATUS_CAPACITY;
    }
    if (detection_count > (size_t)INT_MAX || tracker->track_count > (size_t)INT_MAX ||
        tracker->next_id < 0 ||
        detection_count > (size_t)(INT_MAX - tracker->next_id) ||
        !checked_add_size(tracker->track_count, detection_count, &required_capacity) ||
        !checked_multiply_size(tracker->track_count, detection_count, &iou_count)) {
        return TRACKER_STATUS_OVERFLOW;
    }

    status = reserve_capacity((void**)&tracker->tracks, &tracker->track_capacity,
                              sizeof(*tracker->tracks), required_capacity);
    if (status != TRACKER_STATUS_OK) {
        return status;
    }

    saved_track_count = tracker->track_count;
    saved_next_id = tracker->next_id;

    status = allocate_array((void**)&snapshot, saved_track_count, sizeof(*snapshot));
    if (status == TRACKER_STATUS_OK) {
        status = allocate_array((void**)&track_boxes, saved_track_count,
                                sizeof(*track_boxes));
    }
    if (status == TRACKER_STATUS_OK) {
        status = allocate_array((void**)&detection_boxes, detection_count,
                                sizeof(*detection_boxes));
    }
    if (status == TRACKER_STATUS_OK) {
        status = allocate_array((void**)&iou, iou_count, sizeof(*iou));
    }
    if (status != TRACKER_STATUS_OK) {
        goto cleanup;
    }

    if (saved_track_count) {
        memcpy(snapshot, tracker->tracks, sizeof(*snapshot) * saved_track_count);
    }
    for (size_t i = 0; i < detection_count; ++i) {
        detection_boxes[i] = detections[i].box;
        output[i].tracked.detection = detections[i];
        output[i].tracked.tracker_id = -1;
        output[i].detection_index = i;
    }

    for (size_t i = 0; i < tracker->track_count; ++i) {
        status = kf_xyxy_predict(&tracker->tracks[i].estimator);
        if (status != TRACKER_STATUS_OK) {
            goto rollback;
        }
        ++tracker->tracks[i].time_since_update;
        track_boxes[i] = kf_xyxy_box(&tracker->tracks[i].estimator);
    }

    build_iou_matrix_into(iou, track_boxes, tracker->track_count,
                          detection_boxes, detection_count);
    assignments = assign_greedy(iou, tracker->track_count, detection_count,
                                tracker->minimum_iou_threshold);
    if (assignments.status == -1) {
        status = TRACKER_STATUS_ALLOCATION_FAILED;
        goto rollback;
    }

    for (size_t i = 0; i < assignments.match_count; ++i) {
        const int row = assignments.match_rows[i];
        const int col = assignments.match_cols[i];
        sort_track_t* track = &tracker->tracks[row];

        status = kf_xyxy_update(&track->estimator, detection_boxes[col]);
        if (status != TRACKER_STATUS_OK) {
            goto rollback;
        }
        ++track->number_of_successful_updates;
        track->time_since_update = 0;
        if (track->number_of_successful_updates >= tracker->minimum_consecutive_frames) {
            if (track->tracker_id == -1) {
                track->tracker_id = tracker->next_id++;
            }
            output[col].tracked.tracker_id = track->tracker_id;
        }
    }

    for (size_t i = 0; i < assignments.unmatched_col_count; ++i) {
        const int det_idx = assignments.unmatched_cols[i];
        if (confidence_passes(detections[det_idx], tracker->track_activation_threshold)) {
            sort_track_t* track = sort_add_track(tracker, detection_boxes[det_idx]);
            if (!track) {
                status = TRACKER_STATUS_ALLOCATION_FAILED;
                goto rollback;
            }
            if (track->number_of_successful_updates >= tracker->minimum_consecutive_frames) {
                track->tracker_id = tracker->next_id++;
                output[det_idx].tracked.tracker_id = track->tracker_id;
            }
        }
    }

    sort_retain_alive(tracker);
    *output_count = detection_count;
    status = TRACKER_STATUS_OK;
    goto cleanup;

rollback:
    if (saved_track_count) {
        memcpy(tracker->tracks, snapshot, sizeof(*snapshot) * saved_track_count);
    }
    tracker->track_count = saved_track_count;
    tracker->next_id = saved_next_id;
    *output_count = 0;

cleanup:
    free_assignment(&assignments);
    free(iou);
    free(track_boxes);
    free(detection_boxes);
    free(snapshot);
    return status;
}

size_t sort_update(
    sort_t* tracker,
    const detection_t* detections,
    size_t detection_count,
    tracked_detection_t* output,
    size_t output_capacity
) {
    tracked_detection_ex_t* indexed_output = NULL;
    size_t output_count = 0;

    if (detection_count > SIZE_MAX / sizeof(*indexed_output)) {
        return 0;
    }
    indexed_output = detection_count ? malloc(sizeof(*indexed_output) * detection_count) : NULL;
    if (detection_count && !indexed_output) {
        return 0;
    }

    if (sort_update_ex(tracker, detections, detection_count, indexed_output,
                       detection_count, &output_count) != TRACKER_STATUS_OK) {
        free(indexed_output);
        return 0;
    }

    for (size_t i = 0; i < output_count && i < output_capacity; ++i) {
        if (output) {
            output[i] = indexed_output[i].tracked;
        }
    }
    free(indexed_output);
    return output_count;
}

bytetrack_t* bytetrack_create(const bytetrack_config_t* config_in) {
    const bytetrack_config_t config =
        config_in ? *config_in : bytetrack_default_config();
    int maximum_frames_without_update = 0;
    if (!scaled_lost_buffer_checked(config.lost_track_buffer, config.frame_rate,
                                    &maximum_frames_without_update)) {
        return NULL;
    }
    bytetrack_t* tracker = calloc(1, sizeof(*tracker));
    if (!tracker) {
        return NULL;
    }
    tracker->maximum_frames_without_update = maximum_frames_without_update;
    tracker->minimum_consecutive_frames = config.minimum_consecutive_frames;
    tracker->minimum_iou_threshold = config.minimum_iou_threshold;
    tracker->track_activation_threshold = config.track_activation_threshold;
    tracker->high_conf_det_threshold = config.high_conf_det_threshold;
    return tracker;
}

void bytetrack_destroy(bytetrack_t* tracker) {
    if (!tracker) {
        return;
    }
    free(tracker->tracks);
    free(tracker);
}

void bytetrack_reset(bytetrack_t* tracker) {
    if (!tracker) {
        return;
    }
    tracker->track_count = 0;
    tracker->next_id = 0;
}

tracker_status_t bytetrack_clone(const bytetrack_t* source, bytetrack_t** output) {
    bytetrack_t* clone;
    size_t track_bytes = 0;

    if (!source || !output) {
        return TRACKER_STATUS_INVALID_ARGUMENT;
    }
    *output = NULL;
    if (source->track_count > source->track_capacity ||
        (source->track_capacity && !source->tracks) ||
        !checked_multiply_size(source->track_capacity, sizeof(*source->tracks), &track_bytes)) {
        return TRACKER_STATUS_OVERFLOW;
    }

    clone = calloc(1, sizeof(*clone));
    if (!clone) {
        return TRACKER_STATUS_ALLOCATION_FAILED;
    }
    *clone = *source;
    clone->tracks = NULL;
    if (source->track_capacity) {
        clone->tracks = malloc(track_bytes);
        if (!clone->tracks) {
            free(clone);
            return TRACKER_STATUS_ALLOCATION_FAILED;
        }
        if (source->track_count) {
            memcpy(clone->tracks, source->tracks,
                   sizeof(*clone->tracks) * source->track_count);
        }
    }

    *output = clone;
    return TRACKER_STATUS_OK;
}

static void byte_retain_alive(bytetrack_t* tracker) {
    size_t out = 0;
    for (size_t i = 0; i < tracker->track_count; ++i) {
        byte_track_t* track = &tracker->tracks[i];
        const int mature = track->number_of_successful_updates >= tracker->minimum_consecutive_frames;
        const int active = track->time_since_update == 0;
        if (track->time_since_update < tracker->maximum_frames_without_update &&
            (mature || active)) {
            if (out != i) {
                tracker->tracks[out] = tracker->tracks[i];
            }
            ++out;
        }
    }
    tracker->track_count = out;
}

static box_t* gather_boxes(const detection_t* detections, const int* indices, size_t count) {
    box_t* boxes = count ? malloc(sizeof(*boxes) * count) : NULL;
    if (!boxes && count) {
        return NULL;
    }
    for (size_t i = 0; i < count; ++i) {
        boxes[i] = detections[indices[i]].box;
    }
    return boxes;
}

static tracker_status_t byte_reserve_tracks(bytetrack_t* tracker, size_t required_capacity) {
    if (tracker->track_capacity >= required_capacity) {
        return TRACKER_STATUS_OK;
    }

    size_t new_capacity = tracker->track_capacity == 0 ? 4 : tracker->track_capacity;
    while (new_capacity < required_capacity) {
        if (new_capacity > SIZE_MAX / 2) {
            new_capacity = required_capacity;
            break;
        }
        new_capacity *= 2;
    }
    if (new_capacity > SIZE_MAX / sizeof(*tracker->tracks)) {
        return TRACKER_STATUS_OVERFLOW;
    }

    byte_track_t* tracks = realloc(tracker->tracks, sizeof(*tracks) * new_capacity);
    if (!tracks) {
        return TRACKER_STATUS_ALLOCATION_FAILED;
    }
    tracker->tracks = tracks;
    tracker->track_capacity = new_capacity;
    return TRACKER_STATUS_OK;
}

static byte_track_t* byte_add_track_prepared(bytetrack_t* tracker, box_t box) {
    if (tracker->track_count == tracker->track_capacity) {
        return NULL;
    }
    byte_track_t* track = &tracker->tracks[tracker->track_count++];
    memset(track, 0, sizeof(*track));
    track->tracker_id = -1;
    track->number_of_successful_updates = 1;
    kf_xyxy_init(&track->estimator, box);
    return track;
}

static void push_tracked_ex(
    tracked_detection_ex_t* output,
    size_t* count,
    detection_t detection,
    size_t detection_index,
    int tracker_id
) {
    output[*count].tracked.detection = detection;
    output[*count].tracked.tracker_id = tracker_id;
    output[*count].detection_index = detection_index;
    ++(*count);
}

static tracker_status_t byte_allocate_frame_scratch(
    byte_frame_scratch_t* scratch,
    size_t track_count,
    size_t high_count,
    size_t low_count
) {
    const size_t high_matches = track_count < high_count ? track_count : high_count;
    const size_t low_matches = track_count < low_count ? track_count : low_count;
    size_t high_iou_count = 0;
    size_t low_iou_count = 0;
    tracker_status_t status;

    if (!checked_multiply_size(track_count, high_count, &high_iou_count) ||
        !checked_multiply_size(track_count, low_count, &low_iou_count)) {
        return TRACKER_STATUS_OVERFLOW;
    }

#define BYTE_ALLOCATE(member, count) \
    do { \
        status = allocate_array((void**)&scratch->member, (count), sizeof(*scratch->member)); \
        if (status != TRACKER_STATUS_OK) { \
            return status; \
        } \
    } while (0)
    BYTE_ALLOCATE(track_boxes, track_count);
    BYTE_ALLOCATE(high_boxes, high_count);
    BYTE_ALLOCATE(remaining_boxes, track_count);
    BYTE_ALLOCATE(low_boxes, low_count);
    BYTE_ALLOCATE(high_iou, high_iou_count);
    BYTE_ALLOCATE(low_iou, low_iou_count);
    BYTE_ALLOCATE(high_row_used, track_count);
    BYTE_ALLOCATE(high_col_used, high_count);
    BYTE_ALLOCATE(high_match_rows, high_matches);
    BYTE_ALLOCATE(high_match_cols, high_matches);
    BYTE_ALLOCATE(high_unmatched_rows, track_count);
    BYTE_ALLOCATE(high_unmatched_cols, high_count);
    BYTE_ALLOCATE(low_row_used, track_count);
    BYTE_ALLOCATE(low_col_used, low_count);
    BYTE_ALLOCATE(low_match_rows, low_matches);
    BYTE_ALLOCATE(low_match_cols, low_matches);
    BYTE_ALLOCATE(low_unmatched_rows, track_count);
    BYTE_ALLOCATE(low_unmatched_cols, low_count);
#undef BYTE_ALLOCATE

    status = allocate_array(&scratch->track_snapshot, track_count, sizeof(byte_track_t));
    if (status != TRACKER_STATUS_OK) {
        return status;
    }
    return TRACKER_STATUS_OK;
}

static void byte_restore_kalman_snapshot(
    bytetrack_t* tracker,
    const byte_frame_scratch_t* scratch,
    size_t saved_track_count,
    int saved_next_id,
    size_t* output_count
) {
    if (saved_track_count) {
        memcpy(tracker->tracks, scratch->track_snapshot,
               sizeof(*tracker->tracks) * saved_track_count);
    }
    tracker->track_count = saved_track_count;
    tracker->next_id = saved_next_id;
    *output_count = 0;
}

tracker_status_t bytetrack_update_ex(
    bytetrack_t* tracker,
    const detection_t* detections,
    size_t detection_count,
    tracked_detection_ex_t* output,
    size_t output_capacity,
    size_t* output_count
) {
    byte_frame_scratch_t scratch;
    assignment_result_t high;
    assignment_result_t low;
    size_t high_count = 0;
    size_t low_count = 0;
    size_t required_capacity = 0;
    size_t saved_track_count = 0;
    int saved_next_id = 0;
    tracker_status_t status;

    if (output_count) {
        *output_count = 0;
    }
    if (!tracker || !output_count || (!detections && detection_count)) {
        return TRACKER_STATUS_INVALID_ARGUMENT;
    }
    if (detection_count > output_capacity || (detection_count && !output)) {
        return TRACKER_STATUS_CAPACITY;
    }
    if (detection_count > INT_MAX || tracker->track_count > INT_MAX || tracker->next_id < 0 ||
        tracker->track_count > (size_t)INT_MAX - (size_t)tracker->next_id ||
        detection_count > SIZE_MAX / sizeof(int) ||
        detection_count > SIZE_MAX / sizeof(tracked_detection_ex_t) ||
        !checked_add_size(tracker->track_count, detection_count, &required_capacity)) {
        return TRACKER_STATUS_OVERFLOW;
    }

    memset(&scratch, 0, sizeof(scratch));
    memset(&high, 0, sizeof(high));
    memset(&low, 0, sizeof(low));
    status = allocate_array((void**)&scratch.high_indices, detection_count,
                            sizeof(*scratch.high_indices));
    if (status == TRACKER_STATUS_OK) {
        status = allocate_array((void**)&scratch.low_indices, detection_count,
                                sizeof(*scratch.low_indices));
    }
    if (status != TRACKER_STATUS_OK) {
        byte_frame_scratch_release(&scratch);
        return status;
    }

    for (size_t i = 0; i < detection_count; ++i) {
        if (confidence_or(detections[i], 0.0f) >= tracker->high_conf_det_threshold) {
            scratch.high_indices[high_count++] = (int)i;
        } else {
            scratch.low_indices[low_count++] = (int)i;
        }
    }

    status = byte_allocate_frame_scratch(&scratch, tracker->track_count, high_count, low_count);
    if (status == TRACKER_STATUS_OK) {
        status = byte_reserve_tracks(tracker, required_capacity);
    }
    if (status != TRACKER_STATUS_OK) {
        byte_frame_scratch_release(&scratch);
        return status;
    }

    saved_track_count = tracker->track_count;
    saved_next_id = tracker->next_id;
    if (saved_track_count) {
        memcpy(scratch.track_snapshot, tracker->tracks,
               sizeof(*tracker->tracks) * saved_track_count);
    }

    gather_boxes_into(scratch.high_boxes, detections, scratch.high_indices, high_count);
    gather_boxes_into(scratch.low_boxes, detections, scratch.low_indices, low_count);
    for (size_t i = 0; i < tracker->track_count; ++i) {
        status = kf_xyxy_predict(&tracker->tracks[i].estimator);
        if (status != TRACKER_STATUS_OK) {
            byte_restore_kalman_snapshot(tracker, &scratch, saved_track_count,
                                         saved_next_id, output_count);
            byte_frame_scratch_release(&scratch);
            return status;
        }
        ++tracker->tracks[i].time_since_update;
        scratch.track_boxes[i] = kf_xyxy_box(&tracker->tracks[i].estimator);
    }
    build_iou_matrix_into(scratch.high_iou, scratch.track_boxes, tracker->track_count,
                          scratch.high_boxes, high_count);

    high.match_rows = scratch.high_match_rows;
    high.match_cols = scratch.high_match_cols;
    high.unmatched_rows = scratch.high_unmatched_rows;
    high.unmatched_cols = scratch.high_unmatched_cols;
    assign_greedy_into(scratch.high_iou, tracker->track_count, high_count,
                       tracker->minimum_iou_threshold, scratch.high_row_used,
                       scratch.high_col_used, &high);

    for (size_t i = 0; i < high.match_count; ++i) {
        const int row = high.match_rows[i];
        const int det_idx = scratch.high_indices[high.match_cols[i]];
        byte_track_t* track = &tracker->tracks[row];
        status = kf_xyxy_update(&track->estimator, detections[det_idx].box);
        if (status != TRACKER_STATUS_OK) {
            byte_restore_kalman_snapshot(tracker, &scratch, saved_track_count,
                                         saved_next_id, output_count);
            byte_frame_scratch_release(&scratch);
            return status;
        }
        ++track->number_of_successful_updates;
        track->time_since_update = 0;
        if (track->number_of_successful_updates >= tracker->minimum_consecutive_frames &&
            track->tracker_id == -1) {
            track->tracker_id = tracker->next_id++;
        }
        push_tracked_ex(output, output_count, detections[det_idx], (size_t)det_idx,
                        track->tracker_id);
    }

    for (size_t i = 0; i < high.unmatched_row_count; ++i) {
        scratch.remaining_boxes[i] =
            kf_xyxy_box(&tracker->tracks[high.unmatched_rows[i]].estimator);
    }
    build_iou_matrix_into(scratch.low_iou, scratch.remaining_boxes, high.unmatched_row_count,
                          scratch.low_boxes, low_count);

    low.match_rows = scratch.low_match_rows;
    low.match_cols = scratch.low_match_cols;
    low.unmatched_rows = scratch.low_unmatched_rows;
    low.unmatched_cols = scratch.low_unmatched_cols;
    assign_greedy_into(scratch.low_iou, high.unmatched_row_count, low_count,
                       tracker->minimum_iou_threshold, scratch.low_row_used,
                       scratch.low_col_used, &low);

    for (size_t i = 0; i < low.match_count; ++i) {
        const int track_idx = high.unmatched_rows[low.match_rows[i]];
        const int det_idx = scratch.low_indices[low.match_cols[i]];
        byte_track_t* track = &tracker->tracks[track_idx];
        status = kf_xyxy_update(&track->estimator, detections[det_idx].box);
        if (status != TRACKER_STATUS_OK) {
            byte_restore_kalman_snapshot(tracker, &scratch, saved_track_count,
                                         saved_next_id, output_count);
            byte_frame_scratch_release(&scratch);
            return status;
        }
        ++track->number_of_successful_updates;
        track->time_since_update = 0;
        if (track->number_of_successful_updates >= tracker->minimum_consecutive_frames &&
            track->tracker_id == -1) {
            track->tracker_id = tracker->next_id++;
        }
        push_tracked_ex(output, output_count, detections[det_idx], (size_t)det_idx,
                        track->tracker_id);
    }

    for (size_t i = 0; i < low.unmatched_col_count; ++i) {
        const int det_idx = scratch.low_indices[low.unmatched_cols[i]];
        push_tracked_ex(output, output_count, detections[det_idx], (size_t)det_idx, -1);
    }
    for (size_t i = 0; i < high.unmatched_col_count; ++i) {
        const int det_idx = scratch.high_indices[high.unmatched_cols[i]];
        if (confidence_or(detections[det_idx], 0.0f) >= tracker->track_activation_threshold) {
            (void)byte_add_track_prepared(tracker, detections[det_idx].box);
        }
        push_tracked_ex(output, output_count, detections[det_idx], (size_t)det_idx, -1);
    }

    byte_retain_alive(tracker);
    byte_frame_scratch_release(&scratch);
    return TRACKER_STATUS_OK;
}

size_t bytetrack_update(
    bytetrack_t* tracker,
    const detection_t* detections,
    size_t detection_count,
    tracked_detection_t* output,
    size_t output_capacity
) {
    tracked_detection_ex_t* indexed_output;
    size_t output_count = 0;
    if (!tracker || (!detections && detection_count) ||
        detection_count > SIZE_MAX / sizeof(*indexed_output)) {
        return 0;
    }

    indexed_output = detection_count ? malloc(sizeof(*indexed_output) * detection_count) : NULL;
    if (detection_count && !indexed_output) {
        return 0;
    }
    if (bytetrack_update_ex(tracker, detections, detection_count, indexed_output,
                            detection_count, &output_count) != TRACKER_STATUS_OK) {
        free(indexed_output);
        return 0;
    }
    for (size_t i = 0; i < output_count && i < output_capacity; ++i) {
        if (output) {
            output[i] = indexed_output[i].tracked;
        }
    }
    free(indexed_output);
    return output_count;
}

cbiou_t* cbiou_create(const cbiou_config_t* config_in) {
    const cbiou_config_t config =
        config_in ? *config_in : cbiou_default_config();
    int maximum_frames_without_update = 0;
    if (!scaled_lost_buffer_checked(config.lost_track_buffer, config.frame_rate,
                                    &maximum_frames_without_update)) {
        return NULL;
    }
    cbiou_t* tracker = calloc(1, sizeof(*tracker));
    if (!tracker) {
        return NULL;
    }
    tracker->maximum_frames_without_update = maximum_frames_without_update;
    tracker->minimum_consecutive_frames = config.minimum_consecutive_frames;
    tracker->minimum_biou_threshold = config.minimum_biou_threshold;
    tracker->track_activation_threshold = config.track_activation_threshold;
    tracker->high_conf_det_threshold = config.high_conf_det_threshold;
    tracker->low_conf_det_threshold = config.low_conf_det_threshold;
    tracker->first_buffer_ratio = config.first_buffer_ratio;
    tracker->second_buffer_ratio = config.second_buffer_ratio;
    tracker->fuse_detection_score = config.fuse_detection_score;
    return tracker;
}

void cbiou_destroy(cbiou_t* tracker) {
    if (!tracker) {
        return;
    }
    free(tracker->tracks);
    free(tracker);
}

void cbiou_reset(cbiou_t* tracker) {
    if (!tracker) {
        return;
    }
    tracker->track_count = 0;
    tracker->next_id = 0;
}

static cbiou_track_t* cbiou_add_track(cbiou_t* tracker, box_t box) {
    if (!ensure_capacity((void**)&tracker->tracks, &tracker->track_capacity,
                         sizeof(tracker->tracks[0]), tracker->track_count + 1)) {
        return NULL;
    }
    cbiou_track_t* track = &tracker->tracks[tracker->track_count++];
    memset(track, 0, sizeof(*track));
    track->tracker_id = -1;
    track->number_of_successful_updates = 1;
    kf_xyxy_init(&track->estimator, box);
    return track;
}

static void cbiou_retain_alive(cbiou_t* tracker) {
    size_t out = 0;
    for (size_t i = 0; i < tracker->track_count; ++i) {
        cbiou_track_t* track = &tracker->tracks[i];
        const int mature =
            track->number_of_successful_updates >= tracker->minimum_consecutive_frames;
        const int active = track->time_since_update == 0;
        if (track->time_since_update < tracker->maximum_frames_without_update &&
            (mature || active)) {
            if (out != i) {
                tracker->tracks[out] = tracker->tracks[i];
            }
            ++out;
        }
    }
    tracker->track_count = out;
}

tracker_status_t cbiou_update_ex(
    cbiou_t* tracker,
    const detection_t* detections,
    size_t detection_count,
    tracked_detection_ex_t* output,
    size_t output_capacity,
    size_t* output_count
) {
    cbiou_track_t* snapshot = NULL;
    int* high_indices = NULL;
    int* low_indices = NULL;
    box_t* track_boxes = NULL;
    box_t* high_boxes = NULL;
    box_t* remaining_boxes = NULL;
    box_t* low_boxes = NULL;
    float* high_scores = NULL;
    float* low_scores = NULL;
    assignment_result_t high;
    assignment_result_t low;
    tracker_status_t status = TRACKER_STATUS_OK;
    size_t saved_track_count;
    int saved_next_id;
    size_t required_capacity = 0;
    size_t high_count = 0;
    size_t low_count = 0;
    size_t written = 0;

    memset(&high, 0, sizeof(high));
    memset(&low, 0, sizeof(low));
    if (output_count) {
        *output_count = 0;
    }
    if (!tracker || !output_count || (!detections && detection_count)) {
        return TRACKER_STATUS_INVALID_ARGUMENT;
    }
    if (detection_count > output_capacity || (detection_count && !output)) {
        return TRACKER_STATUS_CAPACITY;
    }
    if (detection_count > (size_t)INT_MAX || tracker->track_count > (size_t)INT_MAX ||
        tracker->next_id < 0 ||
        detection_count > (size_t)(INT_MAX - tracker->next_id) ||
        !checked_add_size(tracker->track_count, detection_count, &required_capacity)) {
        return TRACKER_STATUS_OVERFLOW;
    }

    status = reserve_capacity((void**)&tracker->tracks, &tracker->track_capacity,
                              sizeof(*tracker->tracks), required_capacity);
    if (status != TRACKER_STATUS_OK) {
        return status;
    }

    saved_track_count = tracker->track_count;
    saved_next_id = tracker->next_id;

    status = allocate_array((void**)&snapshot, saved_track_count, sizeof(*snapshot));
    if (status == TRACKER_STATUS_OK) {
        status = allocate_array((void**)&high_indices, detection_count, sizeof(*high_indices));
    }
    if (status == TRACKER_STATUS_OK) {
        status = allocate_array((void**)&low_indices, detection_count, sizeof(*low_indices));
    }
    if (status == TRACKER_STATUS_OK) {
        status = allocate_array((void**)&track_boxes, saved_track_count, sizeof(*track_boxes));
    }
    if (status != TRACKER_STATUS_OK) {
        goto cleanup;
    }

    if (saved_track_count) {
        memcpy(snapshot, tracker->tracks, sizeof(*snapshot) * saved_track_count);
    }

    for (size_t i = 0; i < tracker->track_count; ++i) {
        status = kf_xyxy_predict(&tracker->tracks[i].estimator);
        if (status != TRACKER_STATUS_OK) {
            goto rollback;
        }
        ++tracker->tracks[i].time_since_update;
        track_boxes[i] = kf_xyxy_box(&tracker->tracks[i].estimator);
    }

    for (size_t i = 0; i < detection_count; ++i) {
        const float confidence = confidence_or(detections[i], 0.0f);
        if (confidence >= tracker->high_conf_det_threshold) {
            high_indices[high_count++] = (int)i;
        } else if (confidence >= tracker->low_conf_det_threshold) {
            low_indices[low_count++] = (int)i;
        }
    }

    high_boxes = gather_boxes(detections, high_indices, high_count);
    if (high_count && !high_boxes) {
        status = TRACKER_STATUS_ALLOCATION_FAILED;
        goto rollback;
    }
    high_scores = build_biou_score_matrix(
        track_boxes, tracker->track_count, high_boxes, high_indices, detections,
        high_count, tracker->first_buffer_ratio, tracker->fuse_detection_score);
    if (tracker->track_count && high_count && !high_scores) {
        status = TRACKER_STATUS_ALLOCATION_FAILED;
        goto rollback;
    }

    high = assign_greedy(high_scores, tracker->track_count, high_count,
                         tracker->minimum_biou_threshold);
    if (high.status == -1) {
        status = TRACKER_STATUS_ALLOCATION_FAILED;
        goto rollback;
    }

    for (size_t i = 0; i < high.match_count; ++i) {
        const int row = high.match_rows[i];
        const int det_idx = high_indices[high.match_cols[i]];
        cbiou_track_t* track = &tracker->tracks[row];

        status = kf_xyxy_update(&track->estimator, detections[det_idx].box);
        if (status != TRACKER_STATUS_OK) {
            goto rollback;
        }
        ++track->number_of_successful_updates;
        track->time_since_update = 0;
        if (track->number_of_successful_updates >= tracker->minimum_consecutive_frames &&
            track->tracker_id == -1) {
            track->tracker_id = tracker->next_id++;
        }
        push_tracked_ex(output, &written, detections[det_idx],
                        (size_t)det_idx, track->tracker_id);
    }

    remaining_boxes = high.unmatched_row_count
        ? malloc(sizeof(*remaining_boxes) * high.unmatched_row_count) : NULL;
    if (high.unmatched_row_count && !remaining_boxes) {
        status = TRACKER_STATUS_ALLOCATION_FAILED;
        goto rollback;
    }
    for (size_t i = 0; i < high.unmatched_row_count; ++i) {
        remaining_boxes[i] =
            kf_xyxy_box(&tracker->tracks[high.unmatched_rows[i]].estimator);
    }

    low_boxes = gather_boxes(detections, low_indices, low_count);
    if (low_count && !low_boxes) {
        status = TRACKER_STATUS_ALLOCATION_FAILED;
        goto rollback;
    }
    low_scores = build_biou_score_matrix(
        remaining_boxes, high.unmatched_row_count, low_boxes, low_indices,
        detections, low_count, tracker->second_buffer_ratio, 0);
    if (high.unmatched_row_count && low_count && !low_scores) {
        status = TRACKER_STATUS_ALLOCATION_FAILED;
        goto rollback;
    }

    low = assign_greedy(low_scores, high.unmatched_row_count, low_count,
                        tracker->minimum_biou_threshold);
    if (low.status == -1) {
        status = TRACKER_STATUS_ALLOCATION_FAILED;
        goto rollback;
    }

    for (size_t i = 0; i < low.match_count; ++i) {
        const int track_idx = high.unmatched_rows[low.match_rows[i]];
        const int det_idx = low_indices[low.match_cols[i]];
        cbiou_track_t* track = &tracker->tracks[track_idx];

        status = kf_xyxy_update(&track->estimator, detections[det_idx].box);
        if (status != TRACKER_STATUS_OK) {
            goto rollback;
        }
        ++track->number_of_successful_updates;
        track->time_since_update = 0;
        if (track->number_of_successful_updates >= tracker->minimum_consecutive_frames &&
            track->tracker_id == -1) {
            track->tracker_id = tracker->next_id++;
        }
        push_tracked_ex(output, &written, detections[det_idx],
                        (size_t)det_idx, track->tracker_id);
    }

    for (size_t i = 0; i < low.unmatched_col_count; ++i) {
        const int det_idx = low_indices[low.unmatched_cols[i]];
        push_tracked_ex(output, &written, detections[det_idx], (size_t)det_idx, -1);
    }
    for (size_t i = 0; i < high.unmatched_col_count; ++i) {
        const int det_idx = high_indices[high.unmatched_cols[i]];
        if (confidence_or(detections[det_idx], 0.0f) >= tracker->track_activation_threshold) {
            if (!cbiou_add_track(tracker, detections[det_idx].box)) {
                status = TRACKER_STATUS_ALLOCATION_FAILED;
                goto rollback;
            }
        }
        push_tracked_ex(output, &written, detections[det_idx], (size_t)det_idx, -1);
    }

    cbiou_retain_alive(tracker);
    *output_count = written;
    status = TRACKER_STATUS_OK;
    goto cleanup;

rollback:
    if (saved_track_count) {
        memcpy(tracker->tracks, snapshot, sizeof(*snapshot) * saved_track_count);
    }
    tracker->track_count = saved_track_count;
    tracker->next_id = saved_next_id;
    *output_count = 0;

cleanup:
    free_assignment(&low);
    free(low_scores);
    free(low_boxes);
    free(remaining_boxes);
    free_assignment(&high);
    free(high_scores);
    free(high_boxes);
    free(track_boxes);
    free(high_indices);
    free(low_indices);
    free(snapshot);
    return status;
}

size_t cbiou_update(
    cbiou_t* tracker,
    const detection_t* detections,
    size_t detection_count,
    tracked_detection_t* output,
    size_t output_capacity
) {
    tracked_detection_ex_t* indexed_output = NULL;
    size_t output_count = 0;

    if (detection_count > SIZE_MAX / sizeof(*indexed_output)) {
        return 0;
    }
    indexed_output = detection_count ? malloc(sizeof(*indexed_output) * detection_count) : NULL;
    if (detection_count && !indexed_output) {
        return 0;
    }

    if (cbiou_update_ex(tracker, detections, detection_count, indexed_output,
                        detection_count, &output_count) != TRACKER_STATUS_OK) {
        free(indexed_output);
        return 0;
    }

    for (size_t i = 0; i < output_count && i < output_capacity; ++i) {
        if (output) {
            output[i] = indexed_output[i].tracked;
        }
    }
    free(indexed_output);
    return output_count;
}

static void ocsort_track_free(ocsort_track_t* track) {
    free(track->observations);
    memset(track, 0, sizeof(*track));
}

ocsort_t* ocsort_create(const ocsort_config_t* config_in) {
    const ocsort_config_t config =
        config_in ? *config_in : ocsort_default_config();
    int maximum_frames_without_update = 0;
    if (!scaled_lost_buffer_checked(config.lost_track_buffer, config.frame_rate,
                                    &maximum_frames_without_update)) {
        return NULL;
    }
    ocsort_t* tracker = calloc(1, sizeof(*tracker));
    if (!tracker) {
        return NULL;
    }
    tracker->maximum_frames_without_update = maximum_frames_without_update;
    tracker->minimum_consecutive_frames = config.minimum_consecutive_frames;
    tracker->minimum_iou_threshold = config.minimum_iou_threshold;
    tracker->direction_consistency_weight = config.direction_consistency_weight;
    tracker->high_conf_det_threshold = config.high_conf_det_threshold;
    tracker->delta_t = config.delta_t;
    return tracker;
}

void ocsort_destroy(ocsort_t* tracker) {
    if (!tracker) {
        return;
    }
    for (size_t i = 0; i < tracker->track_count; ++i) {
        ocsort_track_free(&tracker->tracks[i]);
    }
    free(tracker->tracks);
    free(tracker);
}

void ocsort_reset(ocsort_t* tracker) {
    if (!tracker) {
        return;
    }
    for (size_t i = 0; i < tracker->track_count; ++i) {
        ocsort_track_free(&tracker->tracks[i]);
    }
    tracker->track_count = 0;
    tracker->frame_count = 0;
    tracker->next_id = 0;
}

static tracker_status_t ocsort_clone_internal(
    const ocsort_t* source,
    ocsort_t** output
) {
    ocsort_t* clone = NULL;

    if (!source || !output) {
        return TRACKER_STATUS_INVALID_ARGUMENT;
    }
    *output = NULL;

    if (source->track_count > source->track_capacity ||
        (source->track_capacity && !source->tracks) ||
        source->track_capacity > SIZE_MAX / sizeof(*source->tracks)) {
        return TRACKER_STATUS_OVERFLOW;
    }

    clone = calloc(1, sizeof(*clone));
    if (!clone) {
        return TRACKER_STATUS_ALLOCATION_FAILED;
    }
    *clone = *source;
    clone->tracks = NULL;

    if (source->track_capacity) {
        clone->tracks = calloc(source->track_capacity, sizeof(*clone->tracks));
        if (!clone->tracks) {
            free(clone);
            return TRACKER_STATUS_ALLOCATION_FAILED;
        }
    }

    for (size_t i = 0; i < source->track_count; ++i) {
        const ocsort_track_t* src = &source->tracks[i];
        ocsort_track_t* dst = &clone->tracks[i];

        if (src->observation_count > src->observation_capacity ||
            (src->observation_count && !src->observations) ||
            src->observation_capacity > SIZE_MAX / sizeof(*src->observations)) {
            ocsort_destroy(clone);
            return TRACKER_STATUS_OVERFLOW;
        }

        *dst = *src;
        dst->observations = NULL;
        if (src->observation_capacity) {
            dst->observations =
                malloc(sizeof(*dst->observations) * src->observation_capacity);
            if (!dst->observations) {
                ocsort_destroy(clone);
                return TRACKER_STATUS_ALLOCATION_FAILED;
            }
            if (src->observation_count) {
                memcpy(dst->observations, src->observations,
                       sizeof(*dst->observations) * src->observation_count);
            }
        }
    }

    *output = clone;
    return TRACKER_STATUS_OK;
}

static void ocsort_commit_clone(ocsort_t* tracker, ocsort_t* working) {
    ocsort_t previous = *tracker;
    *tracker = *working;
    *working = previous;
    ocsort_destroy(working);
}

static tracker_status_t ocsort_add_observation(
    ocsort_track_t* track,
    int age,
    box_t box
) {
    if (track->observation_count == SIZE_MAX) {
        return TRACKER_STATUS_OVERFLOW;
    }
    tracker_status_t status =
        reserve_capacity((void**)&track->observations, &track->observation_capacity,
                         sizeof(track->observations[0]), track->observation_count + 1);
    if (status != TRACKER_STATUS_OK) {
        return status;
    }
    track->observations[track->observation_count].age = age;
    track->observations[track->observation_count].box = box;
    ++track->observation_count;
    return TRACKER_STATUS_OK;
}

static int ocsort_previous_observation(const ocsort_track_t* track, box_t* out) {
    if (track->observation_count == 0) {
        return 0;
    }
    for (int i = 0; i < track->delta_t; ++i) {
        const int dt = track->delta_t - i;
        const int wanted_age = track->age - dt;
        for (size_t j = 0; j < track->observation_count; ++j) {
            if (track->observations[j].age == wanted_age) {
                *out = track->observations[j].box;
                return 1;
            }
        }
    }
    size_t latest = 0;
    for (size_t i = 1; i < track->observation_count; ++i) {
        if (track->observations[i].age > track->observations[latest].age) {
            latest = i;
        }
    }
    *out = track->observations[latest].box;
    return 1;
}

static ocsort_track_t* ocsort_add_track(ocsort_t* tracker, box_t box) {
    if (!ensure_capacity((void**)&tracker->tracks, &tracker->track_capacity,
                         sizeof(tracker->tracks[0]), tracker->track_count + 1)) {
        return NULL;
    }
    ocsort_track_t* track = &tracker->tracks[tracker->track_count++];
    memset(track, 0, sizeof(*track));
    track->tracker_id = -1;
    track->delta_t = tracker->delta_t;
    track->last_observation = box;
    track->observed = 1;
    kf_xcycsr_init(&track->estimator, box);
    return track;
}

static void ocsort_freeze(ocsort_track_t* track) {
    memcpy(track->frozen_state.x, track->estimator.x, sizeof(track->estimator.x));
    memcpy(track->frozen_state.P, track->estimator.P, sizeof(track->estimator.P));
    track->has_frozen_state = 1;
}

static tracker_status_t ocsort_unfreeze(ocsort_track_t* track, box_t bbox) {
    if (!track->has_frozen_state) {
        return TRACKER_STATUS_OK;
    }

    memcpy(track->estimator.x, track->frozen_state.x, sizeof(track->estimator.x));
    memcpy(track->estimator.P, track->frozen_state.P, sizeof(track->estimator.P));

    const int time_gap = track->time_since_update;
    if (time_gap <= 0) {
        track->has_frozen_state = 0;
        return TRACKER_STATUS_OK;
    }

    float from[4];
    float to[4];
    xyxy_to_xcycsr(track->last_observation, from);
    xyxy_to_xcycsr(bbox, to);
    const float w1 = sqrtf(fmaxf(0.0f, from[2] * from[3]));
    const float h1 = from[2] / fmaxf(w1, TRACKERS_EPS);
    const float w2 = sqrtf(fmaxf(0.0f, to[2] * to[3]));
    const float h2 = to[2] / fmaxf(w2, TRACKERS_EPS);
    const float dx = (to[0] - from[0]) / (float)time_gap;
    const float dy = (to[1] - from[1]) / (float)time_gap;
    const float dw = (w2 - w1) / (float)time_gap;
    const float dh = (h2 - h1) / (float)time_gap;

    for (int i = 0; i < time_gap; ++i) {
        const float x = from[0] + (float)(i + 1) * dx;
        const float y = from[1] + (float)(i + 1) * dy;
        const float w = w1 + (float)(i + 1) * dw;
        const float h = h1 + (float)(i + 1) * dh;
        const float measurement[4] = {x, y, w * h, w / fmaxf(h, TRACKERS_EPS)};

        tracker_status_t status =
            kf_xcycsr_update_measurement(&track->estimator, measurement);
        if (status != TRACKER_STATUS_OK) {
            return status;
        }
        if (i < time_gap - 1) {
            status = kf_xcycsr_predict_raw(&track->estimator);
            if (status != TRACKER_STATUS_OK) {
                return status;
            }
        }
    }

    track->has_frozen_state = 0;
    return TRACKER_STATUS_OK;
}

static tracker_status_t ocsort_update_track(ocsort_track_t* track, const box_t* bbox) {
    if (bbox) {
        box_t previous;
        if (ocsort_previous_observation(track, &previous)) {
            compute_velocity(previous, *bbox, track->velocity);
            track->has_velocity = 1;
        }

        if (!track->observed && track->has_frozen_state) {
            tracker_status_t status = ocsort_unfreeze(track, *bbox);
            if (status != TRACKER_STATUS_OK) {
                return status;
            }
        }

        tracker_status_t status = kf_xcycsr_update(&track->estimator, *bbox);
        if (status != TRACKER_STATUS_OK) {
            return status;
        }

        if (track->number_of_successful_updates == INT_MAX) {
            return TRACKER_STATUS_OVERFLOW;
        }
        track->observed = 1;
        track->time_since_update = 0;
        ++track->number_of_successful_updates;
        track->last_observation = *bbox;
        status = ocsort_add_observation(track, track->age, *bbox);
        if (status != TRACKER_STATUS_OK) {
            return status;
        }
        return TRACKER_STATUS_OK;
    }

    if (track->observed) {
        ocsort_freeze(track);
    }
    track->observed = 0;
    return TRACKER_STATUS_OK;
}

static tracker_status_t ocsort_predict_track(ocsort_track_t* track) {
    if (track->age == INT_MAX || track->time_since_update == INT_MAX) {
        return TRACKER_STATUS_OVERFLOW;
    }

    tracker_status_t status = kf_xcycsr_predict(&track->estimator);
    if (status != TRACKER_STATUS_OK) {
        return status;
    }

    ++track->age;
    if (track->time_since_update > 0) {
        track->number_of_successful_updates = 0;
    }
    ++track->time_since_update;
    return TRACKER_STATUS_OK;
}

static int ocsort_resolve_id(ocsort_t* tracker, ocsort_track_t* track) {
    const int mature = track->number_of_successful_updates >= tracker->minimum_consecutive_frames;
    if (tracker->frame_count <= tracker->minimum_consecutive_frames) {
        if (track->time_since_update == 0) {
            if (track->tracker_id == -1) {
                track->tracker_id = tracker->next_id++;
            }
            return track->tracker_id;
        }
    } else if (mature) {
        if (track->tracker_id == -1) {
            track->tracker_id = tracker->next_id++;
        }
        return track->tracker_id;
    }
    return -1;
}

static void ocsort_prune(ocsort_t* tracker) {
    size_t out = 0;
    for (size_t i = 0; i < tracker->track_count; ++i) {
        if (tracker->tracks[i].time_since_update > tracker->maximum_frames_without_update) {
            ocsort_track_free(&tracker->tracks[i]);
            continue;
        }
        if (out != i) {
            tracker->tracks[out] = tracker->tracks[i];
            memset(&tracker->tracks[i], 0, sizeof(tracker->tracks[i]));
        }
        ++out;
    }
    tracker->track_count = out;
}
static float direction_score(const ocsort_track_t* track, box_t reference,
                             box_t detection, float confidence) {
    if (!track->has_velocity) {
        return 0.0f;
    }
    const float ref_cx = (reference.x1 + reference.x2) * 0.5f;
    const float ref_cy = (reference.y1 + reference.y2) * 0.5f;
    const float det_cx = (detection.x1 + detection.x2) * 0.5f;
    const float det_cy = (detection.y1 + detection.y2) * 0.5f;
    float direction[2] = {det_cx - ref_cx, det_cy - ref_cy};
    const float norm = sqrtf(direction[0] * direction[0] + direction[1] * direction[1]) + TRACKERS_EPS;
    direction[0] /= norm;
    direction[1] /= norm;
    float cos_angle = track->velocity[0] * direction[0] + track->velocity[1] * direction[1];
    cos_angle = fminf(1.0f, fmaxf(-1.0f, cos_angle));
    const float angle = acosf(cos_angle);
    return ((TRACKERS_PI * 0.5f - fabsf(angle)) / TRACKERS_PI) * confidence;
}

size_t ocsort_update(ocsort_t* tracker,
                              const detection_t* detections,
                              size_t detection_count,
                              tracked_detection_t* output,
                              size_t output_capacity) {
    if (!tracker || (!detections && detection_count)) {
        return 0;
    }
    if (tracker->track_count == 0 && detection_count == 0) {
        return 0;
    }

    int* kept = detection_count ? malloc(sizeof(int) * detection_count) : NULL;
    detection_t* filtered = detection_count ? malloc(sizeof(*filtered) * detection_count) : NULL;
    box_t* detection_boxes = detection_count ? malloc(sizeof(*detection_boxes) * detection_count) : NULL;
    float* confidences = detection_count ? malloc(sizeof(*confidences) * detection_count) : NULL;
    if (detection_count && (!kept || !filtered || !detection_boxes || !confidences)) {
        free(kept);
        free(filtered);
        free(detection_boxes);
        free(confidences);
        return 0;
    }

    size_t kept_count = 0;
    for (size_t i = 0; i < detection_count; ++i) {
        if (confidence_passes(detections[i], tracker->high_conf_det_threshold)) {
            kept[kept_count] = (int)i;
            filtered[kept_count] = detections[i];
            detection_boxes[kept_count] = detections[i].box;
            confidences[kept_count] = confidence_or(detections[i], 1.0f);
            ++kept_count;
        }
    }

    for (size_t i = 0; i < tracker->track_count; ++i) {
        ocsort_predict_track(&tracker->tracks[i]);
    }

    box_t* predicted = tracker->track_count ? malloc(sizeof(*predicted) * tracker->track_count) : NULL;
    box_t* reference = tracker->track_count ? malloc(sizeof(*reference) * tracker->track_count) : NULL;
    float* combined = tracker->track_count * kept_count ? malloc(sizeof(*combined) * tracker->track_count * kept_count) : NULL;
    float* iou = tracker->track_count * kept_count ? malloc(sizeof(*iou) * tracker->track_count * kept_count) : NULL;
    if ((tracker->track_count && (!predicted || !reference)) ||
        (tracker->track_count * kept_count && (!combined || !iou))) {
        free(predicted);
        free(reference);
        free(combined);
        free(iou);
        free(kept);
        free(filtered);
        free(detection_boxes);
        free(confidences);
        return 0;
    }

    for (size_t row = 0; row < tracker->track_count; ++row) {
        predicted[row] = kf_xcycsr_box(&tracker->tracks[row].estimator);
        if (!ocsort_previous_observation(&tracker->tracks[row], &reference[row])) {
            reference[row] = tracker->tracks[row].last_observation;
        }
    }
    for (size_t row = 0; row < tracker->track_count; ++row) {
        for (size_t col = 0; col < kept_count; ++col) {
            const float iou_value = compute_iou(predicted[row], detection_boxes[col]);
            iou[row * kept_count + col] = iou_value;
            combined[row * kept_count + col] =
                iou_value + tracker->direction_consistency_weight *
                                direction_score(&tracker->tracks[row], reference[row],
                                                detection_boxes[col], confidences[col]);
        }
    }

    assignment_result_t first = assign_greedy(combined, tracker->track_count, kept_count, -INFINITY);
    if (first.status == -1) {
        free_assignment(&first);
        free(predicted);
        free(reference);
        free(combined);
        free(iou);
        free(kept);
        free(filtered);
        free(detection_boxes);
        free(confidences);
        return 0;
    }

    int* track_unmatched = tracker->track_count ? malloc(sizeof(int) * tracker->track_count) : NULL;
    int* det_unmatched = kept_count ? malloc(sizeof(int) * kept_count) : NULL;
    int* track_used = tracker->track_count ? calloc(tracker->track_count, sizeof(int)) : NULL;
    int* det_used = kept_count ? calloc(kept_count, sizeof(int)) : NULL;
    int* out_det = kept_count ? malloc(sizeof(int) * kept_count) : NULL;
    int* out_id = kept_count ? malloc(sizeof(int) * kept_count) : NULL;
    size_t out_count = 0;

    if ((tracker->track_count && (!track_unmatched || !track_used)) ||
        (kept_count && (!det_unmatched || !det_used || !out_det || !out_id))) {
        free_assignment(&first);
        free(track_unmatched);
        free(det_unmatched);
        free(track_used);
        free(det_used);
        free(out_det);
        free(out_id);
        free(predicted);
        free(reference);
        free(combined);
        free(iou);
        free(kept);
        free(filtered);
        free(detection_boxes);
        free(confidences);
        return 0;
    }

    for (size_t i = 0; i < first.match_count; ++i) {
        const int row = first.match_rows[i];
        const int col = first.match_cols[i];
        if (iou[row * kept_count + col] >= tracker->minimum_iou_threshold) {
            track_used[row] = 1;
            det_used[col] = 1;
            ocsort_update_track(&tracker->tracks[row], &detection_boxes[col]);
            out_det[out_count] = col;
            out_id[out_count] = ocsort_resolve_id(tracker, &tracker->tracks[row]);
            ++out_count;
        }
    }

    size_t track_unmatched_count = 0;
    size_t det_unmatched_count = 0;
    for (size_t i = 0; i < tracker->track_count; ++i) {
        if (!track_used[i]) {
            track_unmatched[track_unmatched_count++] = (int)i;
        }
    }
    for (size_t i = 0; i < kept_count; ++i) {
        if (!det_used[i]) {
            det_unmatched[det_unmatched_count++] = (int)i;
        }
    }

    if (track_unmatched_count && det_unmatched_count) {
        box_t* last_boxes = malloc(sizeof(*last_boxes) * track_unmatched_count);
        box_t* unmatched_det_boxes = malloc(sizeof(*unmatched_det_boxes) * det_unmatched_count);
        if (!last_boxes || !unmatched_det_boxes) {
            free(last_boxes);
            free(unmatched_det_boxes);
            free_assignment(&first);
            free(track_unmatched);
            free(det_unmatched);
            free(track_used);
            free(det_used);
            free(out_det);
            free(out_id);
            free(predicted);
            free(reference);
            free(combined);
            free(iou);
            free(kept);
            free(filtered);
            free(detection_boxes);
            free(confidences);
            return 0;
        }
        for (size_t i = 0; i < track_unmatched_count; ++i) {
            last_boxes[i] = tracker->tracks[track_unmatched[i]].last_observation;
        }
        for (size_t i = 0; i < det_unmatched_count; ++i) {
            unmatched_det_boxes[i] = detection_boxes[det_unmatched[i]];
        }
        float* second_iou = build_iou_matrix(last_boxes, track_unmatched_count,
                                             unmatched_det_boxes, det_unmatched_count);
        assignment_result_t second = assign_greedy(second_iou, track_unmatched_count,
                                                   det_unmatched_count,
                                                   tracker->minimum_iou_threshold);
        if (second.status == -1) {
            free_assignment(&second);
            free(second_iou);
            free(last_boxes);
            free(unmatched_det_boxes);
            free_assignment(&first);
            free(track_unmatched);
            free(det_unmatched);
            free(track_used);
            free(det_used);
            free(out_det);
            free(out_id);
            free(predicted);
            free(reference);
            free(combined);
            free(iou);
            free(kept);
            free(filtered);
            free(detection_boxes);
            free(confidences);
            return 0;
        }
        for (size_t i = 0; i < second.match_count; ++i) {
            const int track_idx = track_unmatched[second.match_rows[i]];
            const int det_idx = det_unmatched[second.match_cols[i]];
            ocsort_update_track(&tracker->tracks[track_idx], &detection_boxes[det_idx]);
            out_det[out_count] = det_idx;
            out_id[out_count] = ocsort_resolve_id(tracker, &tracker->tracks[track_idx]);
            ++out_count;
        }
        for (size_t i = 0; i < second.unmatched_row_count; ++i) {
            ocsort_update_track(&tracker->tracks[track_unmatched[second.unmatched_rows[i]]], NULL);
        }
        ocsort_prune(tracker);
        for (size_t i = 0; i < second.unmatched_col_count; ++i) {
            const int det_idx = det_unmatched[second.unmatched_cols[i]];
            (void)ocsort_add_track(tracker, detection_boxes[det_idx]);
            out_det[out_count] = det_idx;
            out_id[out_count] = -1;
            ++out_count;
        }
        free_assignment(&second);
        free(second_iou);
        free(last_boxes);
        free(unmatched_det_boxes);
    } else {
        for (size_t i = 0; i < track_unmatched_count; ++i) {
            ocsort_update_track(&tracker->tracks[track_unmatched[i]], NULL);
        }
        ocsort_prune(tracker);
        for (size_t i = 0; i < det_unmatched_count; ++i) {
            const int det_idx = det_unmatched[i];
            (void)ocsort_add_track(tracker, detection_boxes[det_idx]);
            out_det[out_count] = det_idx;
            out_id[out_count] = -1;
            ++out_count;
        }
    }

    ++tracker->frame_count;
    size_t written = 0;
    for (size_t i = 0; i < out_count; ++i) {
        push_tracked(output, output_capacity, &written, filtered[out_det[i]], out_id[i]);
    }

    free_assignment(&first);
    free(track_unmatched);
    free(det_unmatched);
    free(track_used);
    free(det_used);
    free(out_det);
    free(out_id);
    free(predicted);
    free(reference);
    free(combined);
    free(iou);
    free(kept);
    free(filtered);
    free(detection_boxes);
    free(confidences);
    return written;
}
