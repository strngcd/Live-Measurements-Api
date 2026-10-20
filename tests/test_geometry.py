import math
import unittest

import numpy as np

from measurement.geometry import body_width_at_row, ellipse_circumference, scale_from_height, threshold_body_mask


def exact_ellipse_perimeter(width, depth, steps=20000):
    a, b = width / 2, depth / 2
    t = np.linspace(0, 2 * np.pi, steps, endpoint=False)
    return float(np.mean(np.sqrt((a * np.sin(t)) ** 2 + (b * np.cos(t)) ** 2)) * 2 * np.pi)


class ScaleFromHeightTest(unittest.TestCase):
    def test_scale_is_height_over_span(self):
        self.assertAlmostEqual(scale_from_height(200, 1800, 170), 170 / 1600)

    def test_upside_down_or_zero_span_is_rejected(self):
        for top, bottom in ((1800, 200), (500, 500)):
            with self.subTest(top=top, bottom=bottom), self.assertRaises(ValueError):
                scale_from_height(top, bottom, 170)

    def test_non_finite_inputs_are_rejected(self):
        for args in ((0, math.nan, 170), (0, 1000, math.nan), (0, 1000, -170)):
            with self.subTest(args=args), self.assertRaises(ValueError):
                scale_from_height(*args)


class BodyWidthAtRowTest(unittest.TestCase):
    def setUp(self):
        self.mask = np.zeros((10, 100), bool)
        self.mask[5, 40:60] = True

    def test_width_of_run_containing_centre(self):
        self.assertEqual(body_width_at_row(self.mask, 5, 50), 20)

    def test_centre_off_body_returns_none_not_a_floor(self):
        # Regression: the original returned 10% of the image width here.
        self.assertIsNone(body_width_at_row(self.mask, 5, 10))

    def test_narrow_run_is_reported_as_measured(self):
        mask = np.zeros((10, 1000), bool)
        mask[5, 495:505] = True
        # Regression: anything under 10% of the width (100px) used to become 100px.
        self.assertEqual(body_width_at_row(mask, 5, 500), 10)

    def test_run_reaching_border_returns_none_not_one_sided_width(self):
        # Regression: a missing left edge used to collapse to the centre.
        self.mask[5, 0:60] = True
        self.assertIsNone(body_width_at_row(self.mask, 5, 50))

    def test_out_of_image_points_return_none(self):
        # Regression: negative indices wrapped to the other side; x == width raised IndexError.
        for row, x in ((5, -1), (5, 100), (-1, 50), (10, 50)):
            with self.subTest(row=row, x=x):
                self.assertIsNone(body_width_at_row(self.mask, row, x))

    def test_threshold_mask_misses_dark_clothing(self):
        # Documents the failure seen in the audit: a dark torso on a bright wall
        # is "background" under the original threshold, so no width is reported.
        image = np.full((100, 100, 3), 200, np.uint8)
        image[:, 40:60] = 30
        self.assertIsNone(body_width_at_row(threshold_body_mask(image), 50, 50))


class EllipseCircumferenceTest(unittest.TestCase):
    def test_circle(self):
        self.assertAlmostEqual(ellipse_circumference(30, 30), math.pi * 30)

    def test_matches_exact_perimeter_for_body_like_ratios(self):
        for ratio in (0.3, 0.5, 0.7, 0.85, 1.0):
            with self.subTest(ratio=ratio):
                exact = exact_ellipse_perimeter(40, 40 * ratio)
                self.assertAlmostEqual(ellipse_circumference(40, 40 * ratio) / exact, 1, delta=1e-4)

    def test_depth_is_an_input_not_derived_from_width(self):
        self.assertNotAlmostEqual(ellipse_circumference(40, 20), ellipse_circumference(40, 30))

    def test_invalid_inputs_are_rejected(self):
        for args in ((-1, 10), (10, math.nan), (math.inf, 10)):
            with self.subTest(args=args), self.assertRaises(ValueError):
                ellipse_circumference(*args)


if __name__ == "__main__":
    unittest.main()
