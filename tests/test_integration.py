"""End-to-end tests with the real MediaPipe model on the bundled sample photos.

They check determinism and input handling, not accuracy: the person in the
samples has no reference tape measurements.
"""
import io
import os
import unittest

import cv2
import numpy as np

import app as app_module

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Values from the original code's first request after startup, before its
# tracker state could leak between images (see the Phase 1 audit). They pin
# the refactor to the same arithmetic; they are not known-correct sizes.
UNCONTAMINATED_BASELINE = {"shoulder_width": 42.68, "arm_length": 59.68, "trouser_length": 84.81}


def read(name):
    with open(os.path.join(REPO, name), "rb") as f:
        return f.read()


def jpeg(image):
    ok, buf = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 95])
    assert ok
    return buf.tobytes()


class SamplePhotoTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.front = read("front.jpeg")
        cls.side = read("left_side.jpeg")
        cls.client = app_module.app.test_client()

    def post(self, front, side=None, height="170"):
        data = {"front": (io.BytesIO(front), "front.jpg"), "height_cm": height}
        if side is not None:
            data["left_side"] = (io.BytesIO(side), "side.jpg")
        return self.client.post("/upload_images", data=data, content_type="multipart/form-data")

    def test_identical_requests_give_identical_responses(self):
        # Regression: three identical requests used to return hips of 230, 278 and 285 cm.
        bodies = [self.post(self.front, self.side).get_json() for _ in range(3)]
        self.assertEqual(bodies[0], bodies[1])
        self.assertEqual(bodies[0], bodies[2])

    def test_side_photo_does_not_affect_later_requests(self):
        # Regression: after a side photo the next front shoulder span dropped 221px -> 162px.
        alone = self.post(self.front).get_json()["measurements"]
        self.post(self.front, self.side)
        self.post(self.side)  # rejected, but still must not leave state behind
        self.assertEqual(self.post(self.front).get_json()["measurements"], alone)

    def test_matches_uncontaminated_baseline(self):
        body = self.post(self.front, self.side).get_json()
        for name, value in UNCONTAMINATED_BASELINE.items():
            with self.subTest(name=name):
                self.assertAlmostEqual(body["measurements"][name], value, delta=0.05)
                self.assertEqual(body["measurement_status"][name]["status"], "approximate")

    def test_feet_cropped_out_is_rejected(self):
        # Regression: this photo was accepted and measured with extrapolated ankles.
        image = cv2.imread(os.path.join(REPO, "front.jpeg"))
        cropped = image[: int(image.shape[0] * 0.78)]
        response = self.post(jpeg(cropped))
        self.assertEqual(response.status_code, 400)
        body = response.get_json()
        self.assertEqual(body["code"], "INVALID_POSE")
        self.assertIn("LEFT_ANKLE", body["missing_landmarks"])
        self.assertIn("RIGHT_ANKLE", body["missing_landmarks"])

    def test_side_photo_submitted_as_front_is_rejected(self):
        response = self.post(self.side)
        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.get_json()["code"], "INVALID_POSE")

    def test_photo_without_a_person_is_rejected(self):
        response = self.post(jpeg(np.full((1600, 900, 3), 200, np.uint8)))
        self.assertEqual(response.status_code, 400)
        self.assertIn("No person", response.get_json()["error"])


if __name__ == "__main__":
    unittest.main()
