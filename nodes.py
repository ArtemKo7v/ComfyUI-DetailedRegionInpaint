"""Mask-guided crop and restore nodes for ComfyUI."""

import math

import torch
import torch.nn.functional as F


_MASK_EPSILON = 1e-6


def _check_image(image, name):
    if not isinstance(image, torch.Tensor) or image.ndim != 4:
        raise ValueError(f"DetailedRegionInpaint: {name} must be an IMAGE [B, H, W, C].")
    if image.shape[0] != 1:
        raise ValueError("DetailedRegionInpaint: only batch size 1 is supported.")
    if min(image.shape[1:]) < 1 or not image.is_floating_point():
        raise ValueError(f"DetailedRegionInpaint: {name} must have nonempty dimensions and a floating dtype.")


def _check_mask(mask, height, width, name):
    if not isinstance(mask, torch.Tensor) or mask.ndim != 3:
        raise ValueError(f"DetailedRegionInpaint: {name} must be a MASK [B, H, W].")
    if mask.shape != (1, height, width):
        raise ValueError(f"DetailedRegionInpaint: {name} must match the image's batch and spatial dimensions.")
    if not mask.is_floating_point() or not bool(torch.isfinite(mask).all()):
        raise ValueError(f"DetailedRegionInpaint: {name} must contain finite floating-point values.")


def _resize_image(image, width, height):
    if image.shape[2] == width and image.shape[1] == height:
        return image
    channels_first = image.permute(0, 3, 1, 2)
    work = channels_first.float() if image.dtype in (torch.float16, torch.bfloat16) else channels_first
    result = F.interpolate(work, size=(height, width), mode="bicubic", align_corners=False, antialias=True)
    return result.to(image.dtype).permute(0, 2, 3, 1).clamp(0, 1)


def _resize_mask(mask, width, height):
    if mask.shape[2] == width and mask.shape[1] == height:
        return mask
    result = F.interpolate(mask[:, None].float(), size=(height, width), mode="bilinear", align_corners=False, antialias=True)
    return result[:, 0].clamp(0, 1)


def _blur_mask(mask, sigma):
    if sigma == 0:
        return mask
    radius = max(1, math.ceil(3 * sigma))
    offsets = torch.arange(-radius, radius + 1, device=mask.device, dtype=torch.float32)
    kernel = torch.exp(-0.5 * (offsets / sigma).square())
    kernel /= kernel.sum()
    work = mask[:, None].float()
    work = F.conv2d(F.pad(work, (radius, radius, 0, 0), mode="replicate"), kernel.view(1, 1, 1, -1))
    work = F.conv2d(F.pad(work, (0, 0, radius, radius), mode="replicate"), kernel.view(1, 1, -1, 1))
    return work[:, 0].clamp(0, 1)


def _target_size(width, height, scale, max_size):
    effective_scale = min(scale, max_size / width, max_size / height)
    target_width = max(1, min(max_size, math.floor(width * effective_scale)))
    target_height = max(1, min(max_size, math.floor(height * effective_scale)))
    # Keep exact source dimensions at scale 1. Use multiples of 8 only when
    # alignment preserves aspect ratio and most of the requested resolution.
    if target_width != width or target_height != height:
        aligned_width = target_width // 8 * 8
        aligned_height = target_height // 8 * 8
        if aligned_width and aligned_height:
            area_fraction = aligned_width * aligned_height / (target_width * target_height)
            aspect_error = abs((aligned_width / aligned_height) / (width / height) - 1)
            avoids_downscale = effective_scale < 1 or (aligned_width >= width and aligned_height >= height)
            if area_fraction >= 0.9 and aspect_error <= 0.01 and avoids_downscale:
                target_width, target_height = aligned_width, aligned_height
    return target_width, target_height, effective_scale


class ArtemKo7v_DetailedRegionInpaint_PrepareInpaintRegion:
    CATEGORY = "ArtemKo7v/inpaint"
    FUNCTION = "prepare"
    RETURN_TYPES = ("IMAGE", "MASK", "DETAILED_REGION")
    RETURN_NAMES = ("image", "mask", "region_data")

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "image": ("IMAGE",),
            "mask": ("MASK",),
            "padding": ("INT", {"default": 32, "min": 0, "max": 1024, "step": 1}),
            "scale": ("FLOAT", {"default": 2.0, "min": 1.0, "max": 8.0, "step": 0.1}),
            "max_size": ("INT", {"default": 1024, "min": 64, "max": 8192, "step": 8}),
            "mask_blur": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 256.0, "step": 0.1}),
        }}

    def prepare(self, image, mask, padding=32, scale=2.0, max_size=1024, mask_blur=0.0):
        _check_image(image, "image")
        _, height, width, _ = image.shape
        _check_mask(mask, height, width, "mask")
        if padding < 0 or not math.isfinite(scale) or scale < 1 or max_size < 1 or not math.isfinite(mask_blur) or mask_blur < 0:
            raise ValueError("DetailedRegionInpaint: invalid padding, scale, max_size, or mask_blur.")

        # Calculate half-open bounds from the unblurred source mask.
        ys, xs = torch.where(mask[0] > _MASK_EPSILON)
        if xs.numel() == 0:
            raise ValueError("DetailedRegionInpaint: input mask is empty.")
        x0 = max(0, int(xs.min()) - padding)
        y0 = max(0, int(ys.min()) - padding)
        x1 = min(width, int(xs.max()) + 1 + padding)
        y1 = min(height, int(ys.max()) + 1 + padding)
        crop_width, crop_height = x1 - x0, y1 - y0

        crop_image = image[:, y0:y1, x0:x1, :]
        crop_mask = mask[:, y0:y1, x0:x1].to(device=image.device, dtype=torch.float32).clamp(0, 1)
        target_width, target_height, effective_scale = _target_size(crop_width, crop_height, scale, max_size)
        crop_mask = _resize_mask(crop_mask, target_width, target_height)
        # Sigma is measured in output mask pixels, regardless of scale or max_size.
        crop_mask = _blur_mask(crop_mask, mask_blur)
        region_data = {
            "source_width": width,
            "source_height": height,
            "crop_x": x0,
            "crop_y": y0,
            "crop_width": crop_width,
            "crop_height": crop_height,
            "target_width": target_width,
            "target_height": target_height,
            "effective_scale": effective_scale,
        }
        return (
            _resize_image(crop_image, target_width, target_height),
            crop_mask,
            region_data,
        )


class ArtemKo7v_DetailedRegionInpaint_RestoreInpaintRegion:
    CATEGORY = "ArtemKo7v/inpaint"
    FUNCTION = "restore"
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "original_image": ("IMAGE",),
            "inpainted_image": ("IMAGE",),
            "original_mask": ("MASK",),
            "region_data": ("DETAILED_REGION",),
        }}

    def restore(self, original_image, inpainted_image, original_mask, region_data):
        _check_image(original_image, "original_image")
        _check_image(inpainted_image, "inpainted_image")
        _, height, width, channels = original_image.shape
        _check_mask(original_mask, height, width, "original_mask")
        if inpainted_image.shape[3] != channels:
            raise ValueError("DetailedRegionInpaint: inpainted_image channel count must match original_image.")

        keys = ("source_width", "source_height", "crop_x", "crop_y", "crop_width", "crop_height")
        if not isinstance(region_data, dict) or any(type(region_data.get(key)) is not int for key in keys):
            raise ValueError("DetailedRegionInpaint: invalid region_data from Prepare Inpaint Region.")
        if (region_data["source_width"], region_data["source_height"]) != (width, height):
            raise ValueError("DetailedRegionInpaint: original_image dimensions do not match region_data.")
        x0, y0 = region_data["crop_x"], region_data["crop_y"]
        crop_width, crop_height = region_data["crop_width"], region_data["crop_height"]
        if x0 < 0 or y0 < 0 or crop_width < 1 or crop_height < 1 or x0 + crop_width > width or y0 + crop_height > height:
            raise ValueError("DetailedRegionInpaint: region_data crop is outside original_image.")

        resized = _resize_image(inpainted_image.to(original_image.device), crop_width, crop_height)
        resized = resized.to(original_image.dtype)
        mask = original_mask[:, y0:y0 + crop_height, x0:x0 + crop_width]
        mask = mask.to(device=original_image.device, dtype=torch.float32).clamp(0, 1)[..., None]
        original_crop = original_image[:, y0:y0 + crop_height, x0:x0 + crop_width, :]
        blend_dtype = torch.float64 if original_image.dtype == torch.float64 else torch.float32
        blended = original_crop.to(blend_dtype) * (1 - mask) + resized.to(blend_dtype) * mask
        # An exactly zero mask retains the source pixel even if the processed
        # crop contains NaNs.
        blended = torch.where(mask == 0, original_crop.to(blend_dtype), blended).to(original_image.dtype)
        result = original_image.clone()
        result[:, y0:y0 + crop_height, x0:x0 + crop_width, :] = blended
        return (result,)


NODE_CLASS_MAPPINGS = {
    "ArtemKo7v_DetailedRegionInpaint_PrepareInpaintRegion": ArtemKo7v_DetailedRegionInpaint_PrepareInpaintRegion,
    "ArtemKo7v_DetailedRegionInpaint_RestoreInpaintRegion": ArtemKo7v_DetailedRegionInpaint_RestoreInpaintRegion,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ArtemKo7v_DetailedRegionInpaint_PrepareInpaintRegion": "Prepare Inpaint Region",
    "ArtemKo7v_DetailedRegionInpaint_RestoreInpaintRegion": "Restore Inpaint Region",
}


