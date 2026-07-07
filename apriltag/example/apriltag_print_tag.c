#include <errno.h>
#include <ctype.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

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

#include "common/image_u8.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb_image_write.h>

typedef void (*tag_destroy_fn)(apriltag_family_t *);

static apriltag_family_t *create_family(const char *name, tag_destroy_fn *destroy)
{
    if (strcmp(name, "tag16h5") == 0) {
        *destroy = tag16h5_destroy;
        return tag16h5_create();
    }
    if (strcmp(name, "tag25h9") == 0) {
        *destroy = tag25h9_destroy;
        return tag25h9_create();
    }
    if (strcmp(name, "tag36h10") == 0) {
        *destroy = tag36h10_destroy;
        return tag36h10_create();
    }
    if (strcmp(name, "tag36h11") == 0) {
        *destroy = tag36h11_destroy;
        return tag36h11_create();
    }
    if (strcmp(name, "tagCircle21h7") == 0) {
        *destroy = tagCircle21h7_destroy;
        return tagCircle21h7_create();
    }
    if (strcmp(name, "tagCircle49h12") == 0) {
        *destroy = tagCircle49h12_destroy;
        return tagCircle49h12_create();
    }
    if (strcmp(name, "tagCustom48h12") == 0) {
        *destroy = tagCustom48h12_destroy;
        return tagCustom48h12_create();
    }
    if (strcmp(name, "tagStandard41h12") == 0) {
        *destroy = tagStandard41h12_destroy;
        return tagStandard41h12_create();
    }
    if (strcmp(name, "tagStandard52h13") == 0) {
        *destroy = tagStandard52h13_destroy;
        return tagStandard52h13_create();
    }

    *destroy = NULL;
    return NULL;
}

static int parse_u32(const char *text, uint32_t *out)
{
    char *end = NULL;
    unsigned long value;

    if (!text || !out)
        return -1;

    errno = 0;
    value = strtoul(text, &end, 10);
    if (errno != 0 || end == text || *end != '\0' || value > UINT32_MAX)
        return -1;

    *out = (uint32_t)value;
    return 0;
}

static int parse_double(const char *text, double *out)
{
    char *end = NULL;
    double value;

    if (!text || !out)
        return -1;

    errno = 0;
    value = strtod(text, &end);
    if (errno != 0 || end == text || *end != '\0' || value <= 0.0)
        return -1;

    *out = value;
    return 0;
}

static int has_extension(const char *path, const char *ext)
{
    size_t path_len;
    size_t ext_len;
    const char *start;

    if (!path || !ext)
        return 0;

    path_len = strlen(path);
    ext_len = strlen(ext);
    if (path_len < ext_len)
        return 0;

    start = path + path_len - ext_len;
    for (size_t i = 0; i < ext_len; ++i) {
        if (tolower((unsigned char)start[i]) != tolower((unsigned char)ext[i]))
            return 0;
    }

    return 1;
}

static void usage(const char *argv0)
{
    printf("Usage: %s [-f family] [-i id] [-s scale] [-e edge_px] [--edge-mm mm --dpi dpi] [-m margin] [-o output]\n", argv0);
    printf("Families: tag16h5 tag25h9 tag36h10 tag36h11 tagCircle21h7 tagCircle49h12\n");
    printf("          tagCustom48h12 tagStandard41h12 tagStandard52h13\n");
    printf("Output format is selected by extension: .png writes PNG; anything else writes PNM/PGM.\n");
    printf("The edge size is the AprilTag detection edge used as pose tagsize: the side length at\n");
    printf("the white/black border transition, excluding the outer white printable margin.\n");
}

static image_u8_t *scale_with_margin(const image_u8_t *tag, uint32_t scale, uint32_t margin_cells)
{
    uint32_t margin_px;
    uint32_t out_width;
    image_u8_t *out;

    if (!tag || scale == 0)
        return NULL;

    margin_px = margin_cells * scale;
    out_width = (uint32_t)tag->width * scale + 2 * margin_px;

    out = image_u8_create(out_width, out_width);
    if (!out)
        return NULL;

    memset(out->buf, 255, (size_t)out->height * (size_t)out->stride);

    for (uint32_t y = 0; y < (uint32_t)tag->height; ++y) {
        for (uint32_t x = 0; x < (uint32_t)tag->width; ++x) {
            uint8_t value = tag->buf[y * tag->stride + x];
            uint32_t oy0 = margin_px + y * scale;
            uint32_t ox0 = margin_px + x * scale;

            for (uint32_t oy = 0; oy < scale; ++oy) {
                uint8_t *row = &out->buf[(oy0 + oy) * out->stride + ox0];
                memset(row, value, scale);
            }
        }
    }

    return out;
}

int main(int argc, char *argv[])
{
    const char *family_name = "tag36h11";
    const char *output_path = "tag.png";
    uint32_t id = 0;
    uint32_t scale = 40;
    uint32_t margin = 2;
    uint32_t edge_px = 0;
    double edge_mm = 0.0;
    double dpi = 300.0;
    tag_destroy_fn destroy = NULL;
    apriltag_family_t *family;
    image_u8_t *tag;
    image_u8_t *printable;
    int result;

    for (int i = 1; i < argc; ++i) {
        if ((strcmp(argv[i], "-h") == 0) || (strcmp(argv[i], "--help") == 0)) {
            usage(argv[0]);
            return 0;
        } else if ((strcmp(argv[i], "-f") == 0 || strcmp(argv[i], "--family") == 0) && i + 1 < argc) {
            family_name = argv[++i];
        } else if ((strcmp(argv[i], "-i") == 0 || strcmp(argv[i], "--id") == 0) && i + 1 < argc) {
            if (parse_u32(argv[++i], &id) != 0) {
                fprintf(stderr, "Invalid tag id.\n");
                return 1;
            }
        } else if ((strcmp(argv[i], "-s") == 0 || strcmp(argv[i], "--scale") == 0) && i + 1 < argc) {
            if (parse_u32(argv[++i], &scale) != 0 || scale == 0) {
                fprintf(stderr, "Invalid scale.\n");
                return 1;
            }
        } else if ((strcmp(argv[i], "-e") == 0 || strcmp(argv[i], "--edge-px") == 0) && i + 1 < argc) {
            if (parse_u32(argv[++i], &edge_px) != 0 || edge_px == 0) {
                fprintf(stderr, "Invalid edge size in pixels.\n");
                return 1;
            }
        } else if (strcmp(argv[i], "--edge-mm") == 0 && i + 1 < argc) {
            if (parse_double(argv[++i], &edge_mm) != 0) {
                fprintf(stderr, "Invalid edge size in millimeters.\n");
                return 1;
            }
        } else if (strcmp(argv[i], "--dpi") == 0 && i + 1 < argc) {
            if (parse_double(argv[++i], &dpi) != 0) {
                fprintf(stderr, "Invalid DPI.\n");
                return 1;
            }
        } else if ((strcmp(argv[i], "-m") == 0 || strcmp(argv[i], "--margin") == 0) && i + 1 < argc) {
            if (parse_u32(argv[++i], &margin) != 0) {
                fprintf(stderr, "Invalid margin.\n");
                return 1;
            }
        } else if ((strcmp(argv[i], "-o") == 0 || strcmp(argv[i], "--out") == 0) && i + 1 < argc) {
            output_path = argv[++i];
        } else {
            usage(argv[0]);
            return 1;
        }
    }

    family = create_family(family_name, &destroy);
    if (!family) {
        fprintf(stderr, "Unknown tag family: %s\n", family_name);
        return 1;
    }

    if (id >= family->ncodes) {
        fprintf(stderr, "Tag id %u is out of range for %s; valid range is 0..%u.\n",
                id, family_name, family->ncodes - 1);
        destroy(family);
        return 1;
    }

    if (edge_px != 0 && edge_mm > 0.0) {
        fprintf(stderr, "Use either --edge-px or --edge-mm, not both.\n");
        destroy(family);
        return 1;
    }

    if (edge_px != 0) {
        if (edge_px % family->width_at_border != 0) {
            fprintf(stderr,
                    "--edge-px must be divisible by %d for %s so every tag cell is an integer number of pixels.\n",
                    family->width_at_border, family_name);
            destroy(family);
            return 1;
        }
        scale = edge_px / (uint32_t)family->width_at_border;
    } else if (edge_mm > 0.0) {
        double desired_edge_px = edge_mm * dpi / 25.4;
        scale = (uint32_t)floor(desired_edge_px / family->width_at_border + 0.5);
        if (scale == 0)
            scale = 1;
    }

    tag = apriltag_to_image(family, id);
    printable = scale_with_margin(tag, scale, margin);
    image_u8_destroy(tag);

    if (!printable) {
        fprintf(stderr, "Failed to create printable tag image.\n");
        destroy(family);
        return 1;
    }

    if (has_extension(output_path, ".png"))
        result = stbi_write_png(output_path, printable->width, printable->height, 1,
                                printable->buf, printable->stride) ? 0 : -1;
    else
        result = image_u8_write_pnm(printable, output_path);

    image_u8_destroy(printable);

    if (result != 0) {
        fprintf(stderr, "Failed to write %s.\n", output_path);
        destroy(family);
        return 1;
    }

    printf("Wrote %s family=%s id=%u scale=%u margin=%u edge_px=%u",
           output_path, family_name, id, scale, margin,
           scale * (uint32_t)family->width_at_border);
    if (edge_mm > 0.0 || dpi > 0.0)
        printf(" edge_mm_at_%.3g_dpi=%.6g", dpi,
               (scale * (uint32_t)family->width_at_border) * 25.4 / dpi);
    printf("\n");

    destroy(family);
    return 0;
}
