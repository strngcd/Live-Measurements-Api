import unittest
from unittest import mock

import cv2
import numpy as np

from measurement import inputs
from measurement.inputs import InputError, decode_image, parse_height_cm


def jpeg_bytes(width, height, value=200):
    ok, buf = cv2.imencode(".jpg", np.full((height, width, 3), value, np.uint8))
    assert ok
    return buf.tobytes()


class ParseHeightTest(unittest.TestCase):
    def assert_rejected(self, raw, code):
        with self.assertRaises(InputError) as ctx:
            parse_height_cm(raw)
        self.assertEqual(ctx.exception.code, code)
        self.assertEqual(ctx.exception.field, "height_cm")

    def test_accepts_numbers_in_range(self):
        self.assertEqual(parse_height_cm("170"), 170.0)
        self.assertEqual(parse_height_cm(" 170.5 "), 170.5)
        self.assertEqual(parse_height_cm(str(inputs.MIN_HEIGHT_CM)), inputs.MIN_HEIGHT_CM)
        self.assertEqual(parse_height_cm(str(inputs.MAX_HEIGHT_CM)), inputs.MAX_HEIGHT_CM)

    def test_missing_height_is_rejected_not_defaulted(self):
        # Regression: a missing height used to become 152 cm silently.
        for raw in (None, "", "   "):
            with self.subTest(raw=raw):
                self.assert_rejected(raw, "MISSING_HEIGHT")

    def test_non_numeric_height_is_rejected_not_defaulted(self):
        # Regression: "abc" used to become 152 cm silently.
        self.assert_rejected("abc", "INVALID_HEIGHT")

    def test_non_finite_and_out_of_range_heights_are_rejected(self):
        # Regression: these produced NaN, negative, zero or kilometre-scale output.
        for raw in ("nan", "inf", "-inf", "-170", "0", "1e6", "1.70", "1700"):
            with self.subTest(raw=raw):
                self.assert_rejected(raw, "INVALID_HEIGHT")


class DecodeImageTest(unittest.TestCase):
    def assert_rejected(self, data, code, **kwargs):
        with self.assertRaises(InputError) as ctx:
            decode_image(data, "front", **kwargs)
        self.assertEqual(ctx.exception.code, code)
        self.assertEqual(ctx.exception.field, "front")

    def test_decodes_valid_jpeg(self):
        image = decode_image(jpeg_bytes(400, 600), "front")
        self.assertEqual(image.shape, (600, 400, 3))

    def test_empty_upload_is_rejected(self):
        # Regression: an empty front file used to crash with HTTP 500.
        self.assert_rejected(b"", "INVALID_IMAGE")

    def test_garbage_is_rejected(self):
        self.assert_rejected(b"not an image at all", "INVALID_IMAGE")

    def test_truncated_jpeg_is_rejected(self):
        self.assert_rejected(jpeg_bytes(400, 600)[:200], "INVALID_IMAGE")

    def test_small_image_is_rejected(self):
        self.assert_rejected(jpeg_bytes(200, 600), "IMAGE_TOO_SMALL")

    def test_too_many_pixels_rejected_before_decoding(self):
        with mock.patch.object(inputs.cv2, "imdecode") as imdecode:
            self.assert_rejected(jpeg_bytes(640, 480), "IMAGE_TOO_LARGE", max_pixels=100_000)
        imdecode.assert_not_called()


if __name__ == "__main__":
    unittest.main()
