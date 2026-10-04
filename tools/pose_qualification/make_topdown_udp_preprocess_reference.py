#!/usr/bin/env python3

import cv2
import numpy as np


SOURCE_W = 13
SOURCE_H = 11
INPUT_W = 6
INPUT_H = 8
BOX = (1.25, 1.5, 5.5, 6.25)
PADDING = 1.25


def source_image() -> np.ndarray:
    y, x = np.indices((SOURCE_H, SOURCE_W), dtype=np.int32)
    b = (17 + x * 7 + y * 3) & 0xFF
    g = (29 + x * 5 + y * 11) & 0xFF
    r = (43 + x * 13 + y * 2) & 0xFF
    return np.stack([b, g, r], axis=-1).astype(np.uint8)


def main() -> None:
    x, y, width, height = BOX
    center_x = np.float32(x + width * 0.5)
    center_y = np.float32(y + height * 0.5)

    aspect = np.float32(INPUT_W / INPUT_H)
    width = np.float32(width)
    height = np.float32(height)
    if width > aspect * height:
        height = width / aspect
    elif width < aspect * height:
        width = height * aspect

    scale_width = np.float32(width * PADDING)
    scale_height = np.float32(height * PADDING)

    # Upstream TopDownAffine(use_udp=true), rotation=0:
    # get_warp_matrix(0, center*2, input_size-1, scale_pixels)
    scale_x = np.float32((INPUT_W - 1) / scale_width)
    scale_y = np.float32((INPUT_H - 1) / scale_height)
    source_to_destination = np.asarray(
        [
            [
                scale_x,
                0.0,
                scale_x * (-center_x + scale_width * 0.5),
            ],
            [
                0.0,
                scale_y,
                scale_y * (-center_y + scale_height * 0.5),
            ],
        ],
        dtype=np.float32,
    )

    crop_bgr = cv2.warpAffine(
        source_image(),
        source_to_destination,
        (INPUT_W, INPUT_H),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )
    rgb_chw = np.transpose(crop_bgr[:, :, ::-1], (2, 0, 1))

    print("center:", float(center_x), float(center_y))
    print("scale:", float(scale_width), float(scale_height))
    print("source_to_destination:")
    print(source_to_destination)
    print("RGB CHW bytes:")
    print(", ".join(str(int(value)) for value in rgb_chw.reshape(-1)))


if __name__ == "__main__":
    main()
