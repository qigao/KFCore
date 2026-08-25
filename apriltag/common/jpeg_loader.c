/* Copyright (C) 2013-2016, The Regents of The University of Michigan.
All rights reserved.
This software was developed in the APRIL Robotics Lab under the
direction of Edwin Olson, ebolson@umich.edu. This software may be
available under alternative licensing terms; contact the address above.
Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:
1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.
2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.
THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
The views and conclusions contained in the software and documentation are those
of the authors and should not be interpreted as representing official policies,
either expressed or implied, of the Regents of The University of Michigan.
*/

#include "jpeg_loader.h"

#include <stdio.h>
#include <stdlib.h>

#include "pjpeg.h"

#ifdef APRILTAG_USE_LIBJPEG_TURBO
#include <turbojpeg.h>

static unsigned char *read_file(const char *path, size_t *out_size)
{
    FILE *f = fopen(path, "rb");
    if (!f)
        return NULL;

    if (fseek(f, 0, SEEK_END) != 0) {
        fclose(f);
        return NULL;
    }

    long len = ftell(f);
    if (len < 0) {
        fclose(f);
        return NULL;
    }

    if (fseek(f, 0, SEEK_SET) != 0) {
        fclose(f);
        return NULL;
    }

    unsigned char *buf = malloc((size_t) len);
    if (!buf) {
        fclose(f);
        return NULL;
    }

    if (fread(buf, 1, (size_t) len, f) != (size_t) len) {
        free(buf);
        fclose(f);
        return NULL;
    }

    fclose(f);
    *out_size = (size_t) len;
    return buf;
}

static image_u8_t *image_u8_create_from_jpeg_turbo(const char *path)
{
    size_t jpeg_size = 0;
    unsigned char *jpeg = read_file(path, &jpeg_size);
    if (!jpeg)
        return NULL;

    tjhandle handle = tj3Init(TJINIT_DECOMPRESS);
    if (!handle) {
        free(jpeg);
        return NULL;
    }

    image_u8_t *im = NULL;
    if (tj3DecompressHeader(handle, jpeg, jpeg_size) == 0) {
        const int width = tj3Get(handle, TJPARAM_JPEGWIDTH);
        const int height = tj3Get(handle, TJPARAM_JPEGHEIGHT);
        if (width > 0 && height > 0) {
            im = image_u8_create((unsigned int) width, (unsigned int) height);
            if (im && tj3Decompress8(handle, jpeg, jpeg_size, im->buf,
                                     im->stride, TJPF_GRAY) != 0) {
                image_u8_destroy(im);
                im = NULL;
            }
        }
    }

    tj3Destroy(handle);
    free(jpeg);
    return im;
}

static image_u8x3_t *image_u8x3_create_from_jpeg_turbo(const char *path)
{
    size_t jpeg_size = 0;
    unsigned char *jpeg = read_file(path, &jpeg_size);
    if (!jpeg)
        return NULL;

    tjhandle handle = tj3Init(TJINIT_DECOMPRESS);
    if (!handle) {
        free(jpeg);
        return NULL;
    }

    image_u8x3_t *im = NULL;
    if (tj3DecompressHeader(handle, jpeg, jpeg_size) == 0) {
        const int width = tj3Get(handle, TJPARAM_JPEGWIDTH);
        const int height = tj3Get(handle, TJPARAM_JPEGHEIGHT);
        if (width > 0 && height > 0) {
            im = image_u8x3_create((unsigned int) width, (unsigned int) height);
            if (im && tj3Decompress8(handle, jpeg, jpeg_size, im->buf,
                                     im->stride, TJPF_RGB) != 0) {
                image_u8x3_destroy(im);
                im = NULL;
            }
        }
    }

    tj3Destroy(handle);
    free(jpeg);
    return im;
}
#endif

static image_u8_t *image_u8_create_from_jpeg_pjpeg(const char *path)
{
    pjpeg_t *pjpeg = pjpeg_create_from_file(path, 0, NULL);
    if (!pjpeg)
        return NULL;

    image_u8_t *im = pjpeg_to_u8_baseline(pjpeg);
    pjpeg_destroy(pjpeg);
    return im;
}

static image_u8x3_t *image_u8x3_create_from_jpeg_pjpeg(const char *path)
{
    pjpeg_t *pjpeg = pjpeg_create_from_file(path, 0, NULL);
    if (!pjpeg)
        return NULL;

    image_u8x3_t *im = pjpeg_to_u8x3_baseline(pjpeg);
    pjpeg_destroy(pjpeg);
    return im;
}

image_u8_t *image_u8_create_from_jpeg(const char *path)
{
#ifdef APRILTAG_USE_LIBJPEG_TURBO
    image_u8_t *im = image_u8_create_from_jpeg_turbo(path);
    if (im)
        return im;
#endif
    return image_u8_create_from_jpeg_pjpeg(path);
}

image_u8x3_t *image_u8x3_create_from_jpeg(const char *path)
{
#ifdef APRILTAG_USE_LIBJPEG_TURBO
    image_u8x3_t *im = image_u8x3_create_from_jpeg_turbo(path);
    if (im)
        return im;
#endif
    return image_u8x3_create_from_jpeg_pjpeg(path);
}
