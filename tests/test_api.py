"""API contract tests. The pose detector is replaced by synthetic landmarks."""
import io
import json
import unittest
from unittest import mock

import cv2
import numpy as np

import app as app_module
from measurement import ALGORITHM_VERSION
from measurement.estimator import MEASUREMENT_NAMES, UNSUPPORTED_REASONS
from measurement.landmarks import PoseLandmark
from tests.helpers import HEIGHT, WIDTH, make_landmarks


def jpeg_bytes(width=WIDTH, height=HEIGHT):
    ok, buf = cv2.imencode(".jpg", np.full((height, width, 3), 200, np.uint8))
    assert ok
    return buf.tobytes()


def strict_json(response):
    """Parse the body the way a browser would: NaN and Infinity are not JSON."""
    def reject(token):
        raise ValueError(f"non-standard JSON token {token}")
    return json.loads(response.get_data(as_text=True), parse_constant=reject)


class UploadImagesTest(unittest.TestCase):
    def setUp(self):
        self.client = app_module.app.test_client()
        self.detect = mock.Mock(return_value=make_landmarks())
        patcher = mock.patch.object(app_module, "detect_pose", self.detect)
        patcher.start()
        self.addCleanup(patcher.stop)

    def post(self, front=None, side=None, height="170", front_name="front.jpg"):
        data = {}
        if front is not None:
            data["front"] = (io.BytesIO(front), front_name)
        if side is not None:
            data["left_side"] = (io.BytesIO(side), "side.jpg")
        if height is not None:
            data["height_cm"] = height
        return self.client.post("/upload_images", data=data, content_type="multipart/form-data")

    def assert_error(self, response, status, code, field=None):
        self.assertEqual(response.status_code, status)
        self.assertEqual(response.content_type, "application/json")
        body = strict_json(response)
        self.assertEqual(body["code"], code)
        self.assertTrue(body["error"])
        if field:
            self.assertEqual(body["field"], field)
        return body

    def test_valid_request_schema(self):
        response = self.post(front=jpeg_bytes())
        self.assertEqual(response.status_code, 200)
        body = strict_json(response)
        self.assertEqual(set(body["measurements"]), set(MEASUREMENT_NAMES))
        self.assertEqual(set(body["measurement_status"]), set(MEASUREMENT_NAMES))
        self.assertEqual(body["units"], "cm")
        self.assertEqual(body["algorithm_version"], ALGORITHM_VERSION)
        self.assertEqual(set(body["debug_info"]), {"scale_factor", "focal_length", "user_height_cm"})
        self.assertEqual(body["debug_info"]["user_height_cm"], 170.0)
        self.assertEqual(
            {w["code"] for w in body["warnings"]}, {"UNVALIDATED_ESTIMATES", "CALIBRATION_BIAS"}
        )

    def test_unsupported_measurements_are_null(self):
        body = strict_json(self.post(front=jpeg_bytes()))
        for name in UNSUPPORTED_REASONS:
            with self.subTest(name=name):
                self.assertIsNone(body["measurements"][name])
                self.assertEqual(body["measurement_status"][name]["status"], "unsupported")
        self.assertAlmostEqual(body["measurements"]["shoulder_width"], 35.06, places=2)

    def test_missing_height_is_rejected(self):
        # Regression: used to fall back to 152 cm silently.
        self.assert_error(self.post(front=jpeg_bytes(), height=None), 400, "MISSING_HEIGHT", "height_cm")
        self.detect.assert_not_called()

    def test_invalid_heights_are_rejected(self):
        # Regression: these returned NaN tokens, negative widths, zeros or kilometre values with 200 OK.
        for height in ("abc", "nan", "inf", "-170", "0", "1e6"):
            with self.subTest(height=height):
                self.assert_error(self.post(front=jpeg_bytes(), height=height), 400, "INVALID_HEIGHT", "height_cm")

    def test_missing_front(self):
        self.assert_error(self.post(side=jpeg_bytes()), 400, "MISSING_FRONT_IMAGE", "front")

    def test_empty_file_input_counts_as_missing(self):
        self.assert_error(self.post(front=b"", front_name=""), 400, "MISSING_FRONT_IMAGE", "front")

    def test_empty_front_file_is_a_json_400(self):
        # Regression: used to crash with an HTML 500 page.
        self.assert_error(self.post(front=b""), 400, "INVALID_IMAGE", "front")

    def test_garbage_front_is_a_json_400(self):
        self.assert_error(self.post(front=b"not an image"), 400, "INVALID_IMAGE", "front")

    def test_corrupt_side_image_is_a_json_400(self):
        # Regression: used to crash with an HTML 500 page.
        response = self.post(front=jpeg_bytes(), side=b"not an image")
        self.assert_error(response, 400, "INVALID_IMAGE", "left_side")

    def test_pose_runs_once_on_front_only(self):
        # Regression: the side photo went through the shared tracker and skewed the next request.
        body = strict_json(self.post(front=jpeg_bytes(), side=jpeg_bytes(800, 1600)))
        self.assertEqual(self.detect.call_count, 1)
        self.assertEqual(self.detect.call_args.args[0].shape, (HEIGHT, WIDTH, 3))
        self.assertIn("SIDE_IMAGE_UNUSED", {w["code"] for w in body["warnings"]})

    def test_no_person_detected(self):
        self.detect.return_value = None
        body = self.assert_error(self.post(front=jpeg_bytes()), 400, "INVALID_POSE")
        self.assertEqual(body["pose"], "front")

    def test_feet_out_of_frame_rejected(self):
        # Regression: validation checked the knees but not the ankles that set the scale.
        moved = {PoseLandmark.LEFT_ANKLE: (0.56, 1.04), PoseLandmark.RIGHT_ANKLE: (0.44, 1.04)}
        self.detect.return_value = make_landmarks(moved)
        body = self.assert_error(self.post(front=jpeg_bytes()), 400, "INVALID_POSE")
        self.assertEqual(body["missing_landmarks"], ["LEFT_ANKLE", "RIGHT_ANKLE"])

    def test_oversized_upload_is_a_json_413(self):
        with mock.patch.dict(app_module.app.config, {"MAX_CONTENT_LENGTH": 1024}):
            self.assert_error(self.post(front=jpeg_bytes()), 413, "PAYLOAD_TOO_LARGE")

    def test_unexpected_error_is_a_json_500(self):
        self.detect.side_effect = RuntimeError("boom")
        with mock.patch.dict(app_module.app.config, {"PROPAGATE_EXCEPTIONS": False}), \
                self.assertLogs(app_module.app.logger, "ERROR"):
            self.assert_error(self.post(front=jpeg_bytes()), 500, "INTERNAL_ERROR")


if __name__ == "__main__":
    unittest.main()
