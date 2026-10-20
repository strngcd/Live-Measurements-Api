import math
import unittest

from measurement.estimator import (
    APPROXIMATE, MEASUREMENT_NAMES, UNAVAILABLE, UNSUPPORTED, UNSUPPORTED_REASONS, estimate_measurements,
)
from measurement.landmarks import PoseLandmark, validate_front_pose
from tests.helpers import HEIGHT, UPRIGHT_FRONT, WIDTH, make_landmarks


class EstimateMeasurementsTest(unittest.TestCase):
    def estimate(self, landmarks=None, width=WIDTH, height=HEIGHT, height_cm=170.0):
        return estimate_measurements(landmarks or make_landmarks(), width, height, height_cm)

    def test_scale_uses_nose_to_lowest_ankle(self):
        moved = {PoseLandmark.RIGHT_ANKLE: (0.44, 0.92)}
        result = self.estimate(make_landmarks(moved))
        self.assertAlmostEqual(result["scale_factor"], 170 / (0.92 * HEIGHT - 0.10 * HEIGHT))

    def test_approximate_measurements(self):
        result = self.estimate()
        scale = 170 / 1600  # nose y=200px, ankles y=1800px
        expected = {
            "shoulder_width": 300 * 1.1 * scale,
            "arm_length": 600 * scale,
            "trouser_length": 800 * scale,
        }
        for name, value in expected.items():
            with self.subTest(name=name):
                self.assertAlmostEqual(result["measurements"][name], value, places=2)
                self.assertEqual(result["measurement_status"][name]["status"], APPROXIMATE)

    def test_every_original_key_is_present_with_a_status(self):
        result = self.estimate()
        self.assertEqual(tuple(result["measurements"]), MEASUREMENT_NAMES)
        self.assertEqual(set(result["measurement_status"]), set(MEASUREMENT_NAMES))

    def test_unsupported_measurements_are_null_not_fabricated(self):
        result = self.estimate()
        for name in UNSUPPORTED_REASONS:
            with self.subTest(name=name):
                self.assertIsNone(result["measurements"][name])
                self.assertEqual(result["measurement_status"][name]["status"], UNSUPPORTED)
                self.assertTrue(result["measurement_status"][name]["reason"])

    def test_hidden_landmark_makes_only_its_measurements_unavailable(self):
        result = self.estimate(make_landmarks(hidden={PoseLandmark.LEFT_WRIST}))
        self.assertIsNone(result["measurements"]["arm_length"])
        status = result["measurement_status"]["arm_length"]
        self.assertEqual(status["status"], UNAVAILABLE)
        self.assertEqual(status["missing_landmarks"], ["LEFT_WRIST"])
        self.assertIsNotNone(result["measurements"]["shoulder_width"])

    def test_padding_the_photo_does_not_change_results(self):
        # Regression: the waist used to be 10% of the photo width, so extra
        # background at the sides changed it (54 cm -> 112 cm in the audit).
        pad = 500
        padded = {lm: ((x * WIDTH + pad) / (WIDTH + 2 * pad), y) for lm, (x, y) in UPRIGHT_FRONT.items()}
        original = self.estimate()["measurements"]
        self.assertEqual(self.estimate(make_landmarks(padded), width=WIDTH + 2 * pad)["measurements"], original)

    def test_ankles_out_of_frame_are_rejected(self):
        # Regression: a photo with the feet cropped off was measured anyway.
        moved = {PoseLandmark.LEFT_ANKLE: (0.56, 1.05), PoseLandmark.RIGHT_ANKLE: (0.44, 1.05)}
        with self.assertRaises(ValueError):
            self.estimate(make_landmarks(moved))

    def test_upside_down_photo_is_rejected(self):
        moved = {PoseLandmark.NOSE: (0.5, 0.95)}
        with self.assertRaises(ValueError):
            self.estimate(make_landmarks(moved))

    def test_invalid_height_is_rejected(self):
        for height_cm in (math.nan, 0, -170):
            with self.subTest(height_cm=height_cm), self.assertRaises(ValueError):
                self.estimate(height_cm=height_cm)

    def test_warnings_are_not_shared_between_calls(self):
        self.estimate()["warnings"].clear()
        self.assertTrue(self.estimate()["warnings"])


class ValidateFrontPoseTest(unittest.TestCase):
    def test_upright_full_body_passes(self):
        self.assertEqual(validate_front_pose(make_landmarks(), WIDTH, HEIGHT)[0], True)

    def test_no_person(self):
        is_valid, message, _ = validate_front_pose(None, WIDTH, HEIGHT)
        self.assertFalse(is_valid)
        self.assertIn("No person", message)

    def test_hidden_ankles_fail_and_are_named(self):
        hidden = {PoseLandmark.LEFT_ANKLE, PoseLandmark.RIGHT_ANKLE}
        is_valid, _, missing = validate_front_pose(make_landmarks(hidden=hidden), WIDTH, HEIGHT)
        self.assertFalse(is_valid)
        self.assertEqual(missing, ["LEFT_ANKLE", "RIGHT_ANKLE"])

    def test_selfie_framing_fails(self):
        moved = {PoseLandmark.LEFT_SHOULDER: (0.52, 0.20), PoseLandmark.RIGHT_SHOULDER: (0.48, 0.20)}
        is_valid, message, _ = validate_front_pose(make_landmarks(moved), WIDTH, HEIGHT)
        self.assertFalse(is_valid)
        self.assertIn("step back", message)


if __name__ == "__main__":
    unittest.main()
