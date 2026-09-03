#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

#include "apriltag.h"
#include "tag16h5.h"
#include "tag25h9.h"
#include "tag36h10.h"
#include "tag36h11.h"
#include "tagCircle21h7.h"
#include "tagCircle49h12.h"
#include "tagCustom48h12.h"
#include "tagStandard41h12.h"
#include "tagStandard52h13.h"

#include "../apriltag.c"

enum { MAX_CORRECTION_CODE_SAMPLES = 64 };

static void test_exact_codes(apriltag_family_t *fam)
{
    for (uint32_t i = 0; i < fam->ncodes; i++) {
        struct quick_decode_result res;
        quick_decode_codeword(fam, fam->codes[i], &res);
        if (res.id != i || res.hamming != 0) {
            printf("Failed 0 errors: family %s code %u, got id %d hamming %d\n",
                   fam->name, i, res.id, res.hamming);
            exit(1);
        }
    }
}

static void test_correctable_errors(apriltag_family_t *fam, int limit)
{
    if (limit < 1)
        return;

    const uint32_t code_count = fam->ncodes;
    const uint32_t sample_count =
        code_count < MAX_CORRECTION_CODE_SAMPLES ? code_count : MAX_CORRECTION_CODE_SAMPLES;
    const int nbits = fam->nbits;

    printf("Family %s: testing correction combinations for %u/%u codes\n",
           fam->name, sample_count, code_count);

    // Exact lookups cover every table entry above. Correction behavior is checked on
    // evenly spaced codes while retaining every bit combination through the supported
    // Hamming limit. Time is O(ncodes + samples * nbits^3); extra space is O(1).
    for (uint32_t sample = 0; sample < sample_count; sample++) {
        const uint32_t i = sample_count == 1
                               ? 0
                               : (uint32_t) (((uint64_t) sample * (code_count - 1)) /
                                             (sample_count - 1));
        const uint64_t code = fam->codes[i];
        struct quick_decode_result res;

        for (int b1 = 0; b1 < nbits; b1++) {
            const uint64_t c1 = code ^ (1ULL << b1);
            quick_decode_codeword(fam, c1, &res);
            if (res.id != i || res.hamming != 1) {
                printf("Failed 1 error: family %s code %u bit %d, got id %d hamming %d\n",
                       fam->name, i, b1, res.id, res.hamming);
                exit(1);
            }

            if (limit < 2)
                continue;

            for (int b2 = b1 + 1; b2 < nbits; b2++) {
                const uint64_t c2 = c1 ^ (1ULL << b2);
                quick_decode_codeword(fam, c2, &res);
                if (res.id != i || res.hamming != 2) {
                    printf("Failed 2 errors: family %s code %u bits %d,%d, got id %d hamming %d\n",
                           fam->name, i, b1, b2, res.id, res.hamming);
                    exit(1);
                }

                if (limit < 3)
                    continue;

                for (int b3 = b2 + 1; b3 < nbits; b3++) {
                    const uint64_t c3 = c2 ^ (1ULL << b3);
                    quick_decode_codeword(fam, c3, &res);
                    if (res.id != i || res.hamming != 3) {
                        printf("Failed 3 errors: family %s code %u bits %d,%d,%d, got id %d hamming %d\n",
                               fam->name, i, b1, b2, b3, res.id, res.hamming);
                        exit(1);
                    }
                }
            }
        }
    }
}

void test_family(apriltag_family_t *fam) {
    printf("Testing family %s with %d codes, %d bits\n", fam->name, fam->ncodes, fam->nbits);

    // Calculate theoretical max correction
    int limit = (fam->h - 1) / 2;
    if (limit > 3) limit = 3;
    printf("Family %s: h=%d, max correction tested=%d\n", fam->name, fam->h, limit);

    apriltag_detector_t *td = apriltag_detector_create();
    // Enable up to limit bits correction
    apriltag_detector_add_family_bits(td, fam, limit);

    // Validate initialization
    struct quick_decode *qd = (struct quick_decode*) fam->impl;
    if (!qd) {
        printf("Failed to init quick_decode for %s\n", fam->name);
        exit(1);
    }

    if (qd->maxhamming != limit) {
        printf("Failed to set maxhamming to %d for %s\n", limit, fam->name);
        exit(1);
    }

    test_exact_codes(fam);
    test_correctable_errors(fam, limit);
    
    apriltag_detector_destroy(td);
    printf("Family %s passed.\n", fam->name);
}

int main() {
    apriltag_family_t *fams[] = {
        tag16h5_create(),
        tag25h9_create(),
        tag36h10_create(),
        tag36h11_create(),
        tagCircle21h7_create(),
        tagCircle49h12_create(),
        tagCustom48h12_create(),
        tagStandard41h12_create(),
        tagStandard52h13_create(),
        NULL
    };

    for (int i = 0; fams[i]; i++) {
        test_family(fams[i]);
    }

    tag16h5_destroy(fams[0]);
    tag25h9_destroy(fams[1]);
    tag36h10_destroy(fams[2]);
    tag36h11_destroy(fams[3]);
    tagCircle21h7_destroy(fams[4]);
    tagCircle49h12_destroy(fams[5]);
    tagCustom48h12_destroy(fams[6]);
    tagStandard41h12_destroy(fams[7]);
    tagStandard52h13_destroy(fams[8]);

    printf("All quick_decode tests passed!\n");
    return 0;
}
