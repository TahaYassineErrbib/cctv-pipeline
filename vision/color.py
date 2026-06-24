"""
Color extraction for garment crops. Kept deliberately simple:

    1. Take a small sample box from the relevant region (upper/lower/long)
    2. Trim a fixed border off that box (cheap, avoids edge contamination)
    3. Mask out likely-skin pixels (YCrCb range)
    4. K-means on what's left, take the largest cluster as "dominant"
    5. Map the dominant RGB to the nearest named color (HSV-based distance,
       so dim/dark lighting doesn't collapse everything into "black"/"gray")
"""

import cv2
import numpy as np


# A reasonably compact named-color palette (RGB).
NAMED_COLORS_RGB = {
    "black":  (0, 0, 0),
    "white":  (255, 255, 255),
    "gray":   (128, 128, 128),
    "red":    (220, 20, 60),
    "orange": (255, 140, 0),
    "yellow": (255, 215, 0),
    "green":  (34, 139, 34),
    "blue":   (30, 60, 200),
    "navy":   (0, 0, 128),
    "purple": (128, 0, 128),
    "pink":   (255, 105, 180),
    "brown":  (139, 69, 19),
    "beige":  (222, 196, 160),
    "khaki":  (189, 183, 107),
}

# Region-specific sample box settings: how big the box is (as a % of the
# crop's own size) and where it's anchored vertically.
REGION_CONFIGS = {
    "upper": {"width_pct": 0.40, "height_pct": 0.30, "vertical_anchor": "bottom"},
    "lower": {"width_pct": 0.40, "height_pct": 0.25, "vertical_anchor": "top"},
    "long":  {"width_pct": 0.30, "height_pct": 0.30, "vertical_anchor": "center"},
}

BORDER_TRIM_PCT = 0.15  # shrink the sample box by this much on each side


# ---------------------------------------------------------------------
# Sample box
# ---------------------------------------------------------------------

def _sample_box(crop_bgr, width_pct, height_pct, vertical_anchor="center"):
    """
    Computes a centered sub-box from a crop, sized as a percentage of the
    crop's own width/height (scales naturally with the YOLO bbox size).

    vertical_anchor: "center", "top", or "bottom" — where the box sits
    vertically within the crop.

    Returns (sub_crop, box_coords) where box_coords = (x1, y1, x2, y2) are
    relative to the input crop, for debug drawing.
    """
    h, w = crop_bgr.shape[:2]
    box_w = max(1, int(w * width_pct))
    box_h = max(1, int(h * height_pct))

    x_start = max(0, (w - box_w) // 2)
    x_end = min(w, x_start + box_w)

    margin_pct = 0.08  # small margin so we don't land right on a seam/edge

    if vertical_anchor == "top":
        y_start = max(0, int(h * margin_pct))
        y_end = min(h, y_start + box_h)
    elif vertical_anchor == "bottom":
        y_end = min(h, h - int(h * margin_pct))
        y_start = max(0, y_end - box_h)
    else:  # "center"
        y_start = max(0, (h - box_h) // 2)
        y_end = min(h, y_start + box_h)

    sub_crop = crop_bgr[y_start:y_end, x_start:x_end]
    box_coords = (x_start, y_start, x_end, y_end)

    if sub_crop.size == 0:
        return crop_bgr, (0, 0, w, h)

    return sub_crop, box_coords


def _trim_border(crop_bgr, trim_pct=BORDER_TRIM_PCT):
    """Shrinks the crop inward by trim_pct on each side."""
    h, w = crop_bgr.shape[:2]
    by, bx = int(h * trim_pct), int(w * trim_pct)

    if h - 2 * by <= 0 or w - 2 * bx <= 0:
        return crop_bgr  # too small to trim, use as-is

    return crop_bgr[by:h - by, bx:w - bx]


# ---------------------------------------------------------------------
# Skin masking
# ---------------------------------------------------------------------

def _build_skin_mask(crop_bgr):
    """
    Boolean mask (True = likely skin) using a standard YCrCb skin range.
    YCrCb is more lighting-robust than HSV here, since luma (Y) is
    separated from the chroma channels that actually distinguish skin tone.
    """
    ycrcb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2YCrCb)
    lower = np.array([0, 135, 85], dtype=np.uint8)
    upper = np.array([255, 180, 135], dtype=np.uint8)
    return cv2.inRange(ycrcb, lower, upper).astype(bool)


# ---------------------------------------------------------------------
# Nearest named color (HSV-based, so dim colors keep their true hue)
# ---------------------------------------------------------------------

def _rgb_to_hsv_single(rgb_value):
    r, g, b = [max(0, min(255, int(c))) for c in rgb_value]
    bgr_pixel = np.array([[[b, g, r]]], dtype=np.uint8)
    return cv2.cvtColor(bgr_pixel, cv2.COLOR_BGR2HSV)[0, 0]


_NAMED_COLORS_HSV = {name: _rgb_to_hsv_single(rgb) for name, rgb in NAMED_COLORS_RGB.items()}


def _nearest_named_color(rgb_value):
    """
    Maps an RGB value to the closest named color using HSV distance
    (hue weighted heavily, value/brightness barely) instead of raw RGB
    distance — so a dim/dark version of a color still matches its true
    hue instead of collapsing into black/brown/gray under low light.
    """
    h, s, v = _rgb_to_hsv_single(rgb_value)

    # genuinely neutral pixels (low saturation or extreme value): hue is
    # meaningless/noisy here, match directly by brightness instead
    if s < 30 or v < 25:
        neutrals = {"black": 0, "gray": 128, "white": 255}
        return min(neutrals, key=lambda name: abs(int(v) - neutrals[name]))

    best_name, best_dist = None, float("inf")
    for name, (ref_h, ref_s, ref_v) in _NAMED_COLORS_HSV.items():
        hue_diff = min(abs(int(h) - int(ref_h)), 180 - abs(int(h) - int(ref_h)))
        dist = (hue_diff * 3.0) ** 2 + (abs(int(s) - int(ref_s)) * 0.5) ** 2 + (abs(int(v) - int(ref_v)) * 0.1) ** 2
        if dist < best_dist:
            best_dist = dist
            best_name = name

    return best_name


# ---------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------

def extract_dominant_color(crop_bgr, k=3):
    """
    Runs k-means on the crop's pixels (after skin masking), picks the
    largest cluster as dominant, maps it to the nearest named color.
    Returns {"name": str, "rgb": (r, g, b)}, or None for empty crops.
    """
    if crop_bgr is None or crop_bgr.size == 0:
        return None

    skin_mask = _build_skin_mask(crop_bgr)
    rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)

    keep_mask = ~skin_mask
    if keep_mask.sum() > 0:
        pixels = rgb[keep_mask].astype(np.float32)
    else:
        # skin masking removed everything — fall back to using all pixels
        # rather than returning nothing
        pixels = rgb.reshape(-1, 3).astype(np.float32)

    if pixels.shape[0] < k:
        mean_rgb = pixels.mean(axis=0) if pixels.shape[0] > 0 else np.array([0, 0, 0])
        return {"name": _nearest_named_color(mean_rgb), "rgb": tuple(int(c) for c in mean_rgb)}

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 0.5)
    _, labels, centers = cv2.kmeans(pixels, k, None, criteria, attempts=3, flags=cv2.KMEANS_PP_CENTERS)

    counts = np.bincount(labels.flatten())
    dominant_rgb = centers[int(np.argmax(counts))]

    return {"name": _nearest_named_color(dominant_rgb), "rgb": tuple(int(c) for c in dominant_rgb)}


def extract_color_for_region(crop_bgr, region):
    """
    Region-aware color extraction:
        1. Take the sample box for this region (upper/lower/long)
        2. Trim a fixed border off it
        3. Run extract_dominant_color (skin masking + k-means) on what's left

    Returns (color_dict, box_coords) — box_coords is the sample box's
    location relative to crop_bgr, for debug drawing.
    """
    if crop_bgr is None or crop_bgr.size == 0:
        return None, None

    cfg = REGION_CONFIGS.get(region)
    if cfg is None:
        raise ValueError(f"Unknown region '{region}', expected one of {list(REGION_CONFIGS)}")

    sample_crop, box_coords = _sample_box(crop_bgr, **cfg)
    trimmed = _trim_border(sample_crop)

    color = extract_dominant_color(trimmed)
    return color, box_coords

# ---------------------------------------------------------------------
# Multi-region, pose-anchored color extraction (additive -- the original
# extract_color_for_region() above is untouched and still usable).
#
# Runs k-means independently in each pose-anchored sub-region, then
# combines them via confidence-weighted voting where the weight comes
# from TWO signals multiplied together:
#   1. how dominant the winning k-means cluster was in that sub-region
#      (cluster pixel count / total pixels)
#   2. an HSV-based trust factor -- a sub-region that's mostly dark/
#      desaturated (shadow, washed-out highlight) is down-weighted, since
#      a "dominant" cluster there is more likely noise than true garment
#      color, even if it technically won by pixel count.
# ---------------------------------------------------------------------

def _hsv_trust_factor(rgb_value):
    """
    Returns a 0-1 trust score for a color sample based on its HSV
    saturation and value. Low saturation (washed out / grayish) or
    very low/high value (deep shadow / blown-out highlight) reduce
    trust, since these are the conditions where a k-means "dominant"
    cluster is most likely to be lighting artifact rather than true
    garment color.
    """
    h, s, v = _rgb_to_hsv_single(rgb_value)

    s_norm = s / 255.0
    v_norm = v / 255.0

    sat_trust = 0.3 + 0.7 * s_norm

    if v_norm < 0.15 or v_norm > 0.95:
        val_trust = 0.3
    else:
        val_trust = 1.0

    return sat_trust * val_trust


def _kmeans_dominant_with_fraction(crop_bgr, k=3):
    """
    Like extract_dominant_color(), but also returns what FRACTION of
    pixels belonged to the winning cluster -- needed as one half of the
    multi-region vote weight (see _hsv_trust_factor for the other half).
    Returns (color_dict, dominance_fraction) or (None, 0.0) for empty crops.
    """
    if crop_bgr is None or crop_bgr.size == 0:
        return None, 0.0

    skin_mask = _build_skin_mask(crop_bgr)
    rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)

    keep_mask = ~skin_mask
    if keep_mask.sum() > 0:
        pixels = rgb[keep_mask].astype(np.float32)
    else:
        pixels = rgb.reshape(-1, 3).astype(np.float32)

    if pixels.shape[0] == 0:
        return None, 0.0

    if pixels.shape[0] < k:
        mean_rgb = pixels.mean(axis=0)
        color = {"name": _nearest_named_color(mean_rgb), "rgb": tuple(int(c) for c in mean_rgb)}
        return color, 1.0

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 0.5)
    _, labels, centers = cv2.kmeans(pixels, k, None, criteria, attempts=3, flags=cv2.KMEANS_PP_CENTERS)

    counts = np.bincount(labels.flatten())
    winning_idx = int(np.argmax(counts))
    dominant_rgb = centers[winning_idx]
    dominance_fraction = float(counts[winning_idx]) / float(len(labels))

    color = {"name": _nearest_named_color(dominant_rgb), "rgb": tuple(int(c) for c in dominant_rgb)}
    return color, dominance_fraction


def extract_color_multi_region(crop_bgr, subregion_boxes):
    """
    Pose-anchored multi-region color extraction.

    subregion_boxes: list of (x1, y1, x2, y2) crop-local boxes, as returned
    by vision.pose_color_regions.compute_color_subregions(). Each box is
    independently k-means'd, then all sub-region results are combined via
    confidence-weighted voting (weight = cluster dominance x HSV trust).

    Returns {"name": str, "rgb": (r, g, b), "num_subregions_used": int}
    or None if no sub-region produced a usable color.
    """
    if not subregion_boxes:
        return None

    name_votes = {}
    rgb_by_name = {}

    for box in subregion_boxes:
        x1, y1, x2, y2 = box
        sub_crop = crop_bgr[y1:y2, x1:x2]
        sub_crop = _trim_border(sub_crop)

        color, dominance_fraction = _kmeans_dominant_with_fraction(sub_crop)
        if color is None:
            continue

        trust = _hsv_trust_factor(color["rgb"])
        weight = dominance_fraction * trust

        name = color["name"]
        name_votes[name] = name_votes.get(name, 0.0) + weight
        rgb_by_name.setdefault(name, []).append((color["rgb"], weight))

    if not name_votes:
        return None

    winning_name = max(name_votes, key=name_votes.get)

    agreeing = rgb_by_name[winning_name]
    total_weight = sum(w for _, w in agreeing)
    if total_weight > 0:
        avg_rgb = tuple(
            int(sum(rgb[c] * w for rgb, w in agreeing) / total_weight)
            for c in range(3)
        )
    else:
        avg_rgb = agreeing[0][0]

    return {
        "name": winning_name,
        "rgb": avg_rgb,
        "num_subregions_used": len(agreeing),
    }
