"""Validation of request inputs: the user's height and the uploaded images."""
import io
import math

import cv2
import numpy as np
from PIL import Image, UnidentifiedImageError

# Sanity bounds that catch typos and unit mix-ups (metres, millimetres),
# not a judgement about which body heights are plausible.
MIN_HEIGHT_CM = 50.0
MAX_HEIGHT_CM = 250.0

MIN_IMAGE_SIDE_PX = 320
MAX_IMAGE_PIXELS = 50_000_000  # checked from the header, before decoding


class InputError(ValueError):
    """A request input that cannot be used, with the API error code to report."""

    def __init__(self, code, message, field=None):
        super().__init__(message)
        self.code = code
        self.message = message
        self.field = field


def parse_height_cm(raw):
    """Return the height in centimetres, or raise InputError."""
    if raw is None or not str(raw).strip():
        raise InputError("MISSING_HEIGHT", "height_cm is required.", "height_cm")
    try:
        height_cm = float(raw)
    except ValueError:
        raise InputError("INVALID_HEIGHT", "height_cm must be a number in centimetres.", "height_cm") from None
    if not math.isfinite(height_cm) or not MIN_HEIGHT_CM <= height_cm <= MAX_HEIGHT_CM:
        raise InputError(
            "INVALID_HEIGHT",
            f"height_cm must be between {MIN_HEIGHT_CM:g} and {MAX_HEIGHT_CM:g} centimetres.",
            "height_cm",
        )
    return height_cm


def decode_image(data, field, min_side=MIN_IMAGE_SIDE_PX, max_pixels=MAX_IMAGE_PIXELS):
    """Decode uploaded image bytes into a BGR array, or raise InputError."""
    if not data:
        raise InputError("INVALID_IMAGE", f"{field} image is empty.", field)

    # Read the size from the header first so a small file that declares a huge
    # image cannot exhaust memory during decoding.
    try:
        with Image.open(io.BytesIO(data)) as header:
            width, height = header.size
    except Image.DecompressionBombError:
        raise InputError("IMAGE_TOO_LARGE", f"{field} image has too many pixels.", field) from None
    except (UnidentifiedImageError, OSError, ValueError):
        raise InputError("INVALID_IMAGE", f"{field} could not be read as an image (use JPEG or PNG).", field) from None
    if width * height > max_pixels:
        raise InputError(
            "IMAGE_TOO_LARGE", f"{field} image must have at most {max_pixels // 1_000_000} megapixels.", field
        )

    try:
        image = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    except cv2.error:
        image = None
    if image is None:
        raise InputError("INVALID_IMAGE", f"{field} could not be read as an image (use JPEG or PNG).", field)
    if min(image.shape[:2]) < min_side:
        raise InputError("IMAGE_TOO_SMALL", f"{field} image must be at least {min_side}px on its shorter side.", field)
    return image
