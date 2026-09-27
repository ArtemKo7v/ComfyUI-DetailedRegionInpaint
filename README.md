# ComfyUI Detailed Region Inpaint

Two ComfyUI nodes crop a masked area, enlarge it for a normal inpainting workflow, and composite the processed crop back into the source image.

## Install

Place this folder in `ComfyUI/custom_nodes/` and restart ComfyUI. PyTorch is supplied by ComfyUI; this package has no additional Python dependencies.

## Workflow

1. Connect the source `IMAGE` and `MASK` to **Prepare Inpaint Region**.
2. Connect its enlarged `image` and `mask` outputs to your preferred inpainting nodes.
3. Connect the resulting image to **Restore Inpaint Region** as `inpainted_image`.
4. Also connect the original image, original mask, and Prepare's `region_data` output to Restore.

The two nodes are in **ArtemKo7v/inpaint**. The `region_data` socket carries the exact crop geometry, so no coordinates need to be entered manually.

## Prepare inputs

| Input | Default | Meaning |
| --- | ---: | --- |
| `padding` | 32 | Context pixels around the nonzero mask bounds, clipped to the source image. |
| `scale` | 2.0 | Requested enlargement factor; 1.0 keeps the crop at its original size unless `max_size` requires a reduction. |
| `max_size` | 1024 | Maximum width or height of the prepared crop. |
| `mask_blur` | 0.0 | Gaussian blur sigma in output mask pixels, applied after resizing. The specified value is independent of `scale` and `max_size`. |

The bounds are calculated from the **original, unblurred** mask. Prepare uses bicubic image resizing and bilinear mask resizing. It favors dimensions divisible by 8 when doing so keeps nearly the same aspect ratio and resolution.

## Restore behavior

Restore resizes the processed image to the stored crop dimensions, then blends it into a copy of the original image using the **original mask**. Mask values from 0 to 1 are preserved for soft compositing; pixels where the original mask is zero remain unchanged. `mask_blur` affects the mask sent to the inpainting workflow, but does not feather the final composite.

The current version supports one image and one mask per call. Larger batches, empty masks, incompatible source dimensions, and invalid crop metadata raise a clear error.

## Tests

Run `python -m unittest discover -s tests -v` from this folder. The tests cover crop bounds, edges, corners, tiny and large regions, soft masks, empty masks, metadata validation, and an unchanged-image round trip.
