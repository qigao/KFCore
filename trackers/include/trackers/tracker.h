#pragma once

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct box {
    float x1;
    float y1;
    float x2;
    float y2;
} box_t;

typedef struct detection {
    box_t box;
    float confidence;
    int has_confidence;
    int class_id;
    int has_class_id;
} detection_t;

typedef struct tracked_detection {
    detection_t detection;
    int tracker_id;
} tracked_detection_t;

typedef enum tracker_status {
    TRACKER_STATUS_OK = 0,
    TRACKER_STATUS_INVALID_ARGUMENT = 1,
    TRACKER_STATUS_CAPACITY = 2,
    TRACKER_STATUS_OVERFLOW = 3,
    TRACKER_STATUS_ALLOCATION_FAILED = 4,
    TRACKER_STATUS_NUMERICAL_FAILURE = 5
} tracker_status_t;

typedef struct tracked_detection_ex {
    tracked_detection_t tracked;
    size_t detection_index;
} tracked_detection_ex_t;

typedef struct sort sort_t;
typedef struct bytetrack bytetrack_t;
typedef struct cbiou cbiou_t;
typedef struct ocsort ocsort_t;

typedef struct sort_config {
    int lost_track_buffer;
    float frame_rate;
    float track_activation_threshold;
    int minimum_consecutive_frames;
    float minimum_iou_threshold;
} sort_config_t;

typedef struct bytetrack_config {
    int lost_track_buffer;
    float frame_rate;
    float track_activation_threshold;
    int minimum_consecutive_frames;
    float minimum_iou_threshold;
    float high_conf_det_threshold;
} bytetrack_config_t;

typedef struct cbiou_config {
    int lost_track_buffer;
    float frame_rate;
    float track_activation_threshold;
    int minimum_consecutive_frames;
    float minimum_biou_threshold;
    float high_conf_det_threshold;
    float low_conf_det_threshold;
    float first_buffer_ratio;
    float second_buffer_ratio;
    int fuse_detection_score;
} cbiou_config_t;

typedef struct ocsort_config {
    int lost_track_buffer;
    float frame_rate;
    int minimum_consecutive_frames;
    float minimum_iou_threshold;
    float direction_consistency_weight;
    float high_conf_det_threshold;
    int delta_t;
} ocsort_config_t;

sort_config_t sort_default_config(void);
bytetrack_config_t bytetrack_default_config(void);
cbiou_config_t cbiou_default_config(void);
ocsort_config_t ocsort_default_config(void);

sort_t* sort_create(const sort_config_t* config);
void sort_destroy(sort_t* tracker);
void sort_reset(sort_t* tracker);
tracker_status_t sort_update_ex(
    sort_t* tracker,
    const detection_t* detections,
    size_t detection_count,
    tracked_detection_ex_t* output,
    size_t output_capacity,
    size_t* output_count
);
size_t sort_update(
    sort_t* tracker,
    const detection_t* detections,
    size_t detection_count,
    tracked_detection_t* output,
    size_t output_capacity
);

bytetrack_t* bytetrack_create(const bytetrack_config_t* config);
void bytetrack_destroy(bytetrack_t* tracker);
void bytetrack_reset(bytetrack_t* tracker);
tracker_status_t bytetrack_clone(const bytetrack_t* source, bytetrack_t** output);
tracker_status_t bytetrack_update_ex(
    bytetrack_t* tracker,
    const detection_t* detections,
    size_t detection_count,
    tracked_detection_ex_t* output,
    size_t output_capacity,
    size_t* output_count
);
size_t bytetrack_update(
    bytetrack_t* tracker,
    const detection_t* detections,
    size_t detection_count,
    tracked_detection_t* output,
    size_t output_capacity
);

cbiou_t* cbiou_create(const cbiou_config_t* config);
void cbiou_destroy(cbiou_t* tracker);
void cbiou_reset(cbiou_t* tracker);
tracker_status_t cbiou_update_ex(
    cbiou_t* tracker,
    const detection_t* detections,
    size_t detection_count,
    tracked_detection_ex_t* output,
    size_t output_capacity,
    size_t* output_count
);
size_t cbiou_update(
    cbiou_t* tracker,
    const detection_t* detections,
    size_t detection_count,
    tracked_detection_t* output,
    size_t output_capacity
);

ocsort_t* ocsort_create(const ocsort_config_t* config);
void ocsort_destroy(ocsort_t* tracker);
void ocsort_reset(ocsort_t* tracker);
tracker_status_t ocsort_update_ex(
    ocsort_t* tracker,
    const detection_t* detections,
    size_t detection_count,
    tracked_detection_ex_t* output,
    size_t output_capacity,
    size_t* output_count
);
size_t ocsort_update(
    ocsort_t* tracker,
    const detection_t* detections,
    size_t detection_count,
    tracked_detection_t* output,
    size_t output_capacity
);

#ifdef __cplusplus
}
#endif
