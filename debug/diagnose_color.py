"""
Diagnostic tool for debugging weak color detection.

Run this against a saved snapshot or a manually-cropped image of a single
person to see EXACTLY what the color pipeline is doing:
    1. The original crop
    2. Where the sample box lands (upper/lower/long region)
    3. Which pixels get excluded by skin masking
    4. Which pixels get excluded by edge/background masking
    5. The final pixels k-means actually clusters on
    6. The extracted dominant color (RGB) and its mapped name

Usage:
    python debug/diagnose_color.py path/to/crop.jpg --region upper

Saves an annotated comparison image to debug/output/ so you can inspect
everything visually in one place instead of guessing from printed numbers.
"""

import argparse
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vision.color import (
    _sample_box, _build_skin_mask, _build_edge_exclusion_mask,
    _crop_to_pixels, extract_dominant_color, _nearest_named_color,
)


REGION_CONFIGS = {
    "upper": {"width_pct": 0.40, "height_pct": 0.30, "vertical_anchor": "bottom"},
    "lower": {"width_pct": 0.40, "height_pct": 0.25, "vertical_anchor": "top"},
    "long":  {"width_pct": 0.30, "height_pct": 0.30, "vertical_anchor": "center"},
}


def make_mask_overlay(crop_bgr, mask, color_bgr):
    """Returns a copy of crop_bgr with mask pixels tinted the given color."""
    overlay = crop_bgr.copy()
    overlay[mask] = color_bgr
    return overlay


def diagnose(crop_path, region):
    crop = cv2.imread(crop_path)
    if crop is None:
        print(f"ERROR: could not read image at {crop_path}")
        return

    print(f"Loaded crop: {crop_path}, shape={crop.shape}")

    cfg = REGION_CONFIGS[region]

    # NEW correct pattern (matches the fixed extract_color_for_region):
    # edge exclusion is computed on the FULL crop, then sliced down to the
    # sample box region — not recomputed on the small sample box itself.
    full_edge_mask = _build_edge_exclusion_mask(crop)

    sample_crop, box_coords = _sample_box(crop, **cfg)
    bx1, by1, bx2, by2 = box_coords
    print(f"Sample box (region='{region}'): {box_coords}")
    print(f"Sample crop shape: {sample_crop.shape}")

    edge_mask = full_edge_mask[by1:by2, bx1:bx2]  # sliced from full-crop analysis
    skin_mask = _build_skin_mask(sample_crop)
    combined_mask = skin_mask | edge_mask

    print(f"Skin mask: {round(100 * skin_mask.mean(), 1)}% of sample box flagged as skin")
    print(f"Edge mask (from full-crop analysis): {round(100 * edge_mask.mean(), 1)}% of sample box flagged as background")
    print(f"Combined excluded: {round(100 * combined_mask.mean(), 1)}%")
    print(f"Remaining for k-means: {round(100 * (~combined_mask).mean(), 1)}%")

    # what k-means actually sees right now, matching production behavior in
    # the FIXED extract_color_for_region (precomputed mask, not recomputed
    # edge exclusion on the sample box itself)
    pixels = _crop_to_pixels(sample_crop, exclude_skin=True, exclude_edge_bg=False,
                              precomputed_exclude_mask=edge_mask)
    print(f"Pixel count fed to k-means: {pixels.shape[0]}")

    if pixels.shape[0] > 0:
        mean_kept_rgb = pixels.mean(axis=0)
        print(f"Mean RGB of kept (non-excluded) pixels: {tuple(int(c) for c in mean_kept_rgb)}")

    result = extract_dominant_color(sample_crop, exclude_skin=True, exclude_edge_bg=False,
                                     precomputed_exclude_mask=edge_mask)
    print(f"\nFINAL RESULT: {result}")

    # also compute with NO exclusion at all, for comparison
    result_no_exclusion = extract_dominant_color(sample_crop, exclude_skin=False, exclude_edge_bg=False)
    print(f"(for comparison) result with NO exclusion: {result_no_exclusion}")

    # build visual comparison: original full crop with box drawn, sample box
    # alone, skin mask overlay, edge mask overlay, combined exclusion overlay
    os.makedirs("debug/output", exist_ok=True)

    full_with_box = crop.copy()
    bx1, by1, bx2, by2 = box_coords
    cv2.rectangle(full_with_box, (bx1, by1), (bx2, by2), (0, 0, 255), 2)

    skin_overlay = make_mask_overlay(sample_crop, skin_mask, (0, 0, 255))      # red = skin
    edge_overlay = make_mask_overlay(sample_crop, edge_mask, (0, 255, 255))    # yellow = background
    combined_overlay = make_mask_overlay(sample_crop, combined_mask, (255, 0, 255))  # magenta = excluded

    # resize everything to the same height for a clean side-by-side
    target_h = 300
    def resize_to_h(img, h=target_h):
        scale = h / img.shape[0]
        w = max(1, int(img.shape[1] * scale))
        return cv2.resize(img, (w, h))

    panels = [
        resize_to_h(full_with_box),
        resize_to_h(sample_crop),
        resize_to_h(skin_overlay),
        resize_to_h(edge_overlay),
        resize_to_h(combined_overlay),
    ]
    labels = ["full crop + box", "sample box", "skin mask (red)", "edge mask (yellow)", "combined excl. (magenta)"]

    # pad each panel with a label strip on top
    label_h = 24
    labeled_panels = []
    for panel, label in zip(panels, labels):
        strip = np.full((label_h, panel.shape[1], 3), (30, 30, 30), dtype=np.uint8)
        cv2.putText(strip, label, (4, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
        labeled_panels.append(np.vstack([strip, panel]))

    comparison = np.hstack(labeled_panels)

    # add a final color swatch + text summary at the bottom
    swatch_h = 60
    swatch = np.zeros((swatch_h, comparison.shape[1], 3), dtype=np.uint8)
    if result is not None:
        r, g, b = result["rgb"]
        swatch[:, :150] = (b, g, r)  # cv2 is BGR
        cv2.putText(swatch, f"name={result['name']}  rgb={result['rgb']}",
                    (160, swatch_h // 2 + 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

    final = np.vstack([comparison, swatch])

    out_path = os.path.join("debug/output", f"diagnose_{region}_{os.path.basename(crop_path)}")
    cv2.imwrite(out_path, final)
    print(f"\nSaved visual comparison to: {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Diagnose color extraction on a single crop.")
    parser.add_argument("crop_path", help="Path to a cropped image (person, upper, or lower crop)")
    parser.add_argument("--region", choices=["upper", "lower", "long"], default="upper",
                         help="Which region config to use for sample box placement")
    args = parser.parse_args()

    diagnose(args.crop_path, args.region)
