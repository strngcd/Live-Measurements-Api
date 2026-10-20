import unittest

import numpy as np

try:
    import torch
    from measurement.depth import estimate_depth
except ImportError:  # torch is only needed for this kept-but-unused component
    torch = None


@unittest.skipIf(torch is None, "torch is not installed")
class EstimateDepthTest(unittest.TestCase):
    def test_feeds_model_a_384_square_rgb_tensor_and_returns_2d_map(self):
        seen = {}

        def fake_model(tensor):
            seen["shape"] = tuple(tensor.shape)
            seen["first_pixel"] = tensor[0, :, 0, 0].tolist()
            return torch.ones((1, 384, 384))

        image = np.zeros((600, 400, 3), np.uint8)
        image[:, :] = (255, 0, 0)  # pure blue in BGR
        depth = estimate_depth(image, model=fake_model)

        self.assertEqual(seen["shape"], (1, 3, 384, 384))
        self.assertEqual(seen["first_pixel"], [0.0, 0.0, 1.0])  # converted to RGB, scaled to 0..1
        self.assertEqual(depth.shape, (384, 384))


if __name__ == "__main__":
    unittest.main()
