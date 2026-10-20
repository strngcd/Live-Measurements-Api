"""Pure geometry: pixel-to-centimetre scale, silhouette widths and circumference."""
import math

import cv2
import numpy as np


def scale_from_height(top_y_px, bottom_y_px, height_cm):
    """Centimetres per pixel, treating the vertical span top..bottom as the person's height.

    The caller picks the landmarks. The API currently uses nose to lowest ankle,
    which is shorter than full stature, so lengths derived from it are too large.
    """
    span_px = bottom_y_px - top_y_px
    # A non-positive span means an upside-down or rotated photo; abs() used to hide it.
    if not (math.isfinite(span_px) and span_px > 0):
        raise ValueError("calibration span must be a positive number of pixels")
    if not (math.isfinite(height_cm) and height_cm > 0):
        raise ValueError("height_cm must be a positive finite number")
    return height_cm / span_px


def threshold_body_mask(image_bgr, threshold=50):
    """The original silhouette: every pixel brighter than `threshold` counts as body.

    Kept for benchmarking only. It marks dark clothing, hair and shoes as
    background and bright walls as body, so widths taken from it are unreliable.
    """
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    return blur > threshold


def body_width_at_row(mask, row, center_x):
    """Width in pixels of the run of body pixels on `row` that contains `center_x`.

    Returns None rather than a guess when the point is outside the image, the
    centre is not on the body, or the run reaches the image border (so its true
    end is unknown).
    """
    mask = np.asarray(mask, dtype=bool)
    height, width = mask.shape[:2]
    if not (0 <= row < height and 0 <= center_x < width):
        return None
    line = mask[int(row)]
    center_x = int(center_x)
    if not line[center_x]:
        return None
    background = np.flatnonzero(~line)
    left = background[background < center_x]
    right = background[background > center_x]
    if left.size == 0 or right.size == 0:
        return None
    return int(right[0] - left[-1] - 1)


def ellipse_circumference(width, depth):
    """Perimeter of an ellipse with the given full width and depth (same units).

    Uses Ramanujan's approximation, which is within 0.01% of the exact perimeter
    for depth/width ratios of 0.3 and above. The depth must be measured, e.g. from
    a side view; the original code assumed depth = 0.7 * width.
    """
    if not (math.isfinite(width) and math.isfinite(depth)) or width < 0 or depth < 0:
        raise ValueError("width and depth must be non-negative finite numbers")
    a, b = width / 2, depth / 2
    return math.pi * (3 * (a + b) - math.sqrt((3 * a + b) * (a + 3 * b)))
