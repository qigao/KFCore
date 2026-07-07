#include "trackers/tracker.h"

#include <stdio.h>

int main(void) {
    bytetrack_t* tracker = bytetrack_create(NULL);
    if (!tracker) {
        return 1;
    }

    detection_t frame1[2] = {
        {{10.0f, 10.0f, 50.0f, 80.0f}, 0.95f, 1, 0, 0},
        {{100.0f, 20.0f, 140.0f, 90.0f}, 0.88f, 1, 0, 0},
    };
    detection_t frame2[2] = {
        {{12.0f, 10.0f, 52.0f, 80.0f}, 0.93f, 1, 0, 0},
        {{102.0f, 21.0f, 142.0f, 91.0f}, 0.87f, 1, 0, 0},
    };
    tracked_detection_t tracked[2];

    size_t count = bytetrack_update(tracker, frame1, 2, tracked, 2);
    puts("Frame 1:");
    for (size_t i = 0; i < count; ++i) {
        printf(
            "  bbox=(%.1f, %.1f, %.1f, %.1f) tracker_id=%d\n",
            tracked[i].detection.box.x1,
            tracked[i].detection.box.y1,
            tracked[i].detection.box.x2,
            tracked[i].detection.box.y2,
            tracked[i].tracker_id
        );
    }

    count = bytetrack_update(tracker, frame2, 2, tracked, 2);
    puts("Frame 2:");
    for (size_t i = 0; i < count; ++i) {
        printf(
            "  bbox=(%.1f, %.1f, %.1f, %.1f) tracker_id=%d\n",
            tracked[i].detection.box.x1,
            tracked[i].detection.box.y1,
            tracked[i].detection.box.x2,
            tracked[i].detection.box.y2,
            tracked[i].tracker_id
        );
    }

    bytetrack_destroy(tracker);
    return 0;
}
