import unittest

import torch

from nodes import (
    ArtemKo7v_DetailedRegionInpaint_PrepareInpaintRegion as Prepare,
    ArtemKo7v_DetailedRegionInpaint_RestoreInpaintRegion as Restore,
)


class RegionNodesTests(unittest.TestCase):
    def setUp(self):
        self.prepare = Prepare()
        self.restore = Restore()
        self.image = torch.rand(1, 40, 60, 3)
        self.mask = torch.zeros(1, 40, 60)

    def test_center_bounds_and_padding(self):
        self.mask[:, 12:18, 20:28] = 1
        crop, crop_mask, data = self.prepare.prepare(
            self.image, self.mask, padding=3, scale=1, max_size=1024, mask_blur=0
        )
        self.assertEqual((data["crop_x"], data["crop_y"], data["crop_width"], data["crop_height"]), (17, 9, 14, 12))
        self.assertEqual(tuple(crop.shape), (1, 12, 14, 3))
        self.assertTrue(torch.equal(crop, self.image[:, 9:21, 17:31]))
        self.assertTrue(torch.equal(crop_mask, self.mask[:, 9:21, 17:31]))

    def test_each_edge_and_corner(self):
        cases = [
            (0, 10), (59, 10), (20, 0), (20, 39),
            (0, 0), (59, 0), (0, 39), (59, 39),
        ]
        for x, y in cases:
            with self.subTest(x=x, y=y):
                mask = torch.zeros_like(self.mask)
                mask[:, y, x] = 1
                crop, _, data = self.prepare.prepare(
                    self.image, mask, padding=32, scale=2, max_size=1024, mask_blur=0
                )
                self.assertGreaterEqual(data["crop_x"], 0)
                self.assertGreaterEqual(data["crop_y"], 0)
                self.assertLessEqual(data["crop_x"] + data["crop_width"], 60)
                self.assertLessEqual(data["crop_y"] + data["crop_height"], 40)
                self.assertLessEqual(data["crop_x"], x)
                self.assertLessEqual(data["crop_y"], y)
                self.assertGreater(data["crop_x"] + data["crop_width"], x)
                self.assertGreater(data["crop_y"] + data["crop_height"], y)
                self.assertEqual(tuple(crop.shape[1:3]), (data["target_height"], data["target_width"]))

    def test_tiny_mask_and_max_size(self):
        mask = torch.zeros(1, 256, 384)
        image = torch.rand(1, 256, 384, 3)
        mask[:, 100:104, 150:154] = 1
        crop, _, data = self.prepare.prepare(image, mask, padding=0, scale=8, max_size=64, mask_blur=0)
        self.assertEqual((data["crop_width"], data["crop_height"]), (4, 4))
        self.assertEqual(tuple(crop.shape[1:3]), (32, 32))
        mask[:, 40:200, 60:340] = 1
        crop, _, data = self.prepare.prepare(image, mask, padding=0, scale=4, max_size=512, mask_blur=0)
        self.assertLessEqual(max(crop.shape[1:3]), 512)
        self.assertLessEqual(abs(crop.shape[2] / crop.shape[1] - data["crop_width"] / data["crop_height"]), 0.02)
        self.assertLessEqual(data["effective_scale"], 4)

    def test_blur_does_not_change_geometry_or_compositing_mask(self):
        self.mask[:, 12:18, 20:28] = 1
        plain, _, plain_data = self.prepare.prepare(
            self.image, self.mask, padding=4, scale=1, max_size=1024, mask_blur=0
        )
        _, blurred, blur_data = self.prepare.prepare(
            self.image, self.mask, padding=4, scale=1, max_size=1024, mask_blur=2
        )
        self.assertEqual(plain_data, blur_data)
        self.assertTrue(bool(((blurred > 0) & (blurred < 1)).any()))
        white = torch.ones_like(plain)
        restored, = self.restore.restore(self.image, white, self.mask, blur_data)
        self.assertTrue(torch.equal(restored[self.mask[..., None].expand_as(self.image) == 0], self.image[self.mask[..., None].expand_as(self.image) == 0]))

    def test_soft_mask_blend_and_outside_preservation(self):
        self.mask[:, 15, 25] = 0.25
        self.mask[:, 15, 26] = 1
        _, _, data = self.prepare.prepare(self.image, self.mask, padding=2, scale=1, max_size=1024, mask_blur=0)
        white = torch.ones(1, data["crop_height"], data["crop_width"], 3)
        restored, = self.restore.restore(self.image, white, self.mask, data)
        expected = self.image[:, 15, 25] * 0.75 + 0.25
        self.assertTrue(torch.allclose(restored[:, 15, 25], expected, atol=1e-6))
        self.assertTrue(torch.equal(restored[:, 15, 26], torch.ones_like(restored[:, 15, 26])))
        self.assertTrue(torch.equal(restored[self.mask[..., None].expand_as(self.image) == 0], self.image[self.mask[..., None].expand_as(self.image) == 0]))

    def test_empty_mask_and_batch_errors(self):
        with self.assertRaisesRegex(ValueError, "input mask is empty"):
            self.prepare.prepare(self.image, self.mask)
        with self.assertRaisesRegex(ValueError, "only batch size 1"):
            self.prepare.prepare(self.image.repeat(2, 1, 1, 1), self.mask.repeat(2, 1, 1))

    def test_metadata_and_source_dimensions(self):
        self.mask[:, 15:20, 25:30] = 1
        crop, _, data = self.prepare.prepare(self.image, self.mask, scale=1)
        with self.assertRaisesRegex(ValueError, "dimensions do not match region_data"):
            self.restore.restore(self.image[:, :-1], crop, self.mask[:, :-1], data)
        bad_data = dict(data, crop_x=60)
        with self.assertRaisesRegex(ValueError, "outside original_image"):
            self.restore.restore(self.image, crop, self.mask, bad_data)

    def test_identity_round_trip(self):
        y = torch.linspace(0, 1, 80)[None, :, None, None]
        x = torch.linspace(0, 1, 96)[None, None, :, None]
        image = ((x + y) / 2).expand(1, 80, 96, 3).contiguous()
        mask = torch.zeros(1, 80, 96)
        mask[:, 20:55, 30:70] = 1
        crop, _, data = self.prepare.prepare(image, mask, padding=8, scale=3, max_size=256, mask_blur=3)
        restored, = self.restore.restore(image, crop, mask, data)
        self.assertEqual(tuple(restored.shape), tuple(image.shape))
        self.assertLess((restored - image).abs().max().item(), 0.005)
        self.assertTrue(torch.equal(restored[mask[..., None].expand_as(image) == 0], image[mask[..., None].expand_as(image) == 0]))


    def test_scale_one_keeps_dimensions_and_cap_can_reduce(self):
        self.mask[:, 5:35, 10:50] = 1
        crop, _, data = self.prepare.prepare(self.image, self.mask, padding=0, scale=1, max_size=1024)
        self.assertEqual(tuple(crop.shape[1:3]), (30, 40))
        self.assertEqual(data["effective_scale"], 1)
        crop, _, data = self.prepare.prepare(self.image, self.mask, padding=0, scale=1, max_size=32)
        self.assertLessEqual(max(crop.shape[1:3]), 32)
        self.assertLessEqual(data["effective_scale"], 1)

    def test_half_precision_cpu_round_trip(self):
        self.mask[:, 10:20, 20:30] = 1
        image = self.image.half()
        crop, prepared_mask, data = self.prepare.prepare(image, self.mask, padding=4, scale=2, mask_blur=1)
        restored, = self.restore.restore(image, crop, self.mask, data)
        self.assertEqual(crop.dtype, torch.float16)
        self.assertEqual(prepared_mask.dtype, torch.float32)
        self.assertEqual(restored.dtype, torch.float16)
        self.assertTrue(torch.equal(restored[self.mask[..., None].expand_as(image) == 0], image[self.mask[..., None].expand_as(image) == 0]))
if __name__ == "__main__":
    unittest.main()


