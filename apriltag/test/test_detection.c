#include <stdarg.h>
#include <apriltag.h>
#include <tag36h11.h>
#include <common/jpeg_loader.h>
#include <math.h>

#include "getline.h"

static const double DETECTION_CORNER_TOLERANCE_PX = 2e-1;


char*
format(const char* fmt, ...)
{
    va_list args1;
    va_list args2;

    va_start(args1, fmt);
    va_copy(args2, args1);

    const int N = vsnprintf(NULL, 0, fmt, args1);
    va_end(args1);

    if (N < 0) {
        va_end(args2);
        return NULL;
    }

    char* res_fmt = calloc(N+1, sizeof(char));
    const int n = vsnprintf(res_fmt, N+1, fmt, args2);
    va_end(args2);

    if (n != N) {
        free(res_fmt);
        return NULL;
    }

    // caller has to free returned string
    return res_fmt;
}

int
detection_compare_function(const apriltag_detection_t *a, const apriltag_detection_t *b)
{
    if (a->id != b->id) {
        return copysign(1, a->id - b->id);
    }

    for (int e = 0; e<4; e++) {
        for (int c = 0; c<2; c++) {
            const double d = a->p[e][c] - b->p[e][c];
            if (fabs(d) > DETECTION_CORNER_TOLERANCE_PX) {
                return copysign(1, d);
            }
        }
    }

    return 0;
}

int
detection_array_element_compare_function(const void *_a, const void *_b)
{
    const apriltag_detection_t * const a = *(apriltag_detection_t**) _a;
    const apriltag_detection_t * const b = *(apriltag_detection_t**) _b;

    return detection_compare_function(a, b);
}

int
main(int argc, char *argv[])
{
    if (argc!=2) {
        return EXIT_FAILURE;
    }

    // load image
    char* const path_img = format("%s.jpg", argv[1]);
    image_u8_t *im = image_u8_create_from_jpeg(path_img);
    if (im == NULL) {
        free(path_img);
        return EXIT_FAILURE;
    }
    free(path_img);

    // load true detection
    char* const path_det_true = format("%s.txt", argv[1]);
    FILE *fp = fopen(path_det_true, "r");
    if (fp == NULL) {
        return EXIT_FAILURE;
    }
    free(path_det_true);

    apriltag_detector_t *td = apriltag_detector_create();
    td->quad_decimate = 1;
    td->refine_edges = false;
    apriltag_family_t *tf = tag36h11_create();
    apriltag_detector_add_family(td, tf);

    const char fmt_det[] = "%i, (%.4lf %.4lf), (%.4lf %.4lf), (%.4lf %.4lf), (%.4lf %.4lf)";
    const char fmt_ref_parse[] = "%i, (%lf %lf), (%lf %lf), (%lf %lf), (%lf %lf)";

    bool ok = true;

    zarray_t *detections = apriltag_detector_detect(td, im);

    // sort detections by detected corners for deterministic sorting order
    zarray_sort(detections, detection_array_element_compare_function);

    int i = 0;
    for (; i < zarray_size(detections); i++) {
        apriltag_detection_t *det;
        zarray_get(detections, i, &det);

        char* const det_fmt = format(fmt_det,
            det->id,
            det->p[0][0], det->p[0][1], det->p[1][0], det->p[1][1],
            det->p[2][0], det->p[2][1], det->p[3][0], det->p[3][1]);

        char* line = NULL;
        size_t len = 0;
        const ssize_t nread = apriltag_test_getline(&line, &len, fp);
        if (nread == -1) {
            free(line);
            return EXIT_FAILURE;
        }

        printf("Got:      %s\n", det_fmt);
        printf("Expected: %s\n", line);

        // parse reference detection
        apriltag_detection_t ref;
        const int nparsed = sscanf(
            line, fmt_ref_parse,
            &ref.id,
            &ref.p[0][0], &ref.p[0][1], &ref.p[1][0], &ref.p[1][1],
            &ref.p[2][0], &ref.p[2][1], &ref.p[3][0], &ref.p[3][1]);

        (void) nparsed;
        assert(nparsed == 9);

        // compare detections
        const bool equ = detection_compare_function(det, &ref) == 0;

        if (!equ || det->id != ref.id) {
            fprintf(stderr, "Mismatch.\nGot:\n  %s\nExpected:\n  %s\n", det_fmt, line);
            ok = false;
        }

        free(det_fmt);
        free(line);
    }

    // check that we compared the expected amount of detections
    // if there are no "true" detections left, we should be at the end of the file
    // if there are no "detected" detections left, we should be at the end of the array
    if ((fgetc(fp) != -1) && (i != zarray_size(detections))) {
        return EXIT_FAILURE;
    }

    fclose(fp);

    apriltag_detections_destroy(detections);
    image_u8_destroy(im);

    apriltag_detector_destroy(td);
    tag36h11_destroy(tf);

    if (!ok) {
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
