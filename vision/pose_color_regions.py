"""
Computes pose-anchored sample regions for MULTI-region color extraction.

This is deliberately separate from vision/pose_splitting.py, which handles
upper/lower GARMENT-TYPE splitting (no longer used now that
vision/splitting_yolo.py is the settled method for that). This module's
only job is finding WHERE to sample color from, using pose keypoints --
independent of whichever method produced the upper/lower garment boxes.

Three quads are computed:
    upper (torso):       shoulder-L, shoulder-R, hip-R, hip-L
    lower (thigh):        hip-L, hip-R, knee-R, knee-L
        (stops at the knee, not the ankle, to stay clear of exposed lower
        leg/skin)
    long (full garment): shoulder-L, shoulder-R, knee-R, knee-L
        (for jellaba/manteau -- full-body garments that don't have a
        separate upper/lower split. Spans shoulder to knee, same
        knee-not-ankle reasoning as "lower".)

Each quad is split into 3 vertical sub-regions (left / center / right) for
multi-region color voting.

COCO-17 keypoint indices:
    5  = left_shoulder    6  = right_shoulder
    11 = left_hip         12 = right_hip
    13 = left_knee        14 = right_knee
"""

import numpy as np

import config


LEFT_SHOULDER, RIGHT_SHOULDER = 5, 6
LEFT_HIP, RIGHT_HIP = 11, 12
LEFT_KNEE, RIGHT_KNEE = 13, 14

KEYPOINT_CONF_THRESHOLD = getattr(config, "POSE_KEYPOINT_CONF_THRESHOLD", 0.5)
NUM_COLOR_SUBREGIONS = getattr(config, "COLOR_NUM_SUBREGIONS", 3)


def _kpt_ok(keypoints_conf, idx, conf_threshold):
    return keypoints_conf is None or keypoints_conf[idx] >= conf_threshold


def _quad_bounds(local_xy, keypoints_conf, top_left_idx, top_right_idx,
                  bottom_right_idx, bottom_left_idx, conf_threshold, crop_h, crop_w):
    """
    Reduces a 4-keypoint quad to an axis-aligned bounding box. Requires at
    least 3 of the 4 keypoints to be confident; with 2 or fewer the box
    would be a guess in disguise, so returns None for the caller to fall
    back on.
    """
    idxs = [top_left_idx, top_right_idx, bottom_right_idx, bottom_left_idx]
    ok_flags = [_kpt_ok(keypoints_conf, i, conf_threshold) for i in idxs]

    if sum(ok_flags) < 3:
        return None

    points = [local_xy[i] for i, ok in zip(idxs, ok_flags) if ok]
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]

    x1, x2 = max(0, int(min(xs))), min(crop_w, int(max(xs)))
    y1, y2 = max(0, int(min(ys))), min(crop_h, int(max(ys)))

    if x2 <= x1 or y2 <= y1:
        return None

    return (x1, y1, x2, y2)


def _split_into_vertical_subregions(box, num_subregions):
    """Splits an (x1, y1, x2, y2) box into N equal-width vertical strips."""
    x1, y1, x2, y2 = box
    width = x2 - x1
    strip_w = max(1, width // num_subregions)

    strips = []
    for i in range(num_subregions):
        sx1 = x1 + i * strip_w
        sx2 = x2 if i == num_subregions - 1 else x1 + (i + 1) * strip_w
        if sx2 > sx1:
            strips.append((sx1, y1, sx2, y2))

    return strips


def compute_color_subregions(person_crop_bgr, keypoints_xy, keypoints_conf,
                               crop_origin, region, num_subregions=NUM_COLOR_SUBREGIONS,
                               conf_threshold=KEYPOINT_CONF_THRESHOLD):
    """
    Returns a list of (x1, y1, x2, y2) crop-local sub-region boxes for
    color sampling, for region="upper" (torso quad), region="lower"
    (hip-to-knee quad), or region="long" (shoulder-to-knee quad, for
    full-body garments like jellaba/manteau). Returns None if keypoints
    weren't usable for this sample.
    """
    if keypoints_xy is None:
        return None

    h, w = person_crop_bgr.shape[:2]
    if h == 0 or w == 0:
        return None

    ox, oy = crop_origin
    local_xy = np.asarray(keypoints_xy, dtype=np.float32).copy()
    local_xy[:, 0] -= ox
    local_xy[:, 1] -= oy

    if region == "upper":
        box = _quad_bounds(local_xy, keypoints_conf,
                            LEFT_SHOULDER, RIGHT_SHOULDER, RIGHT_HIP, LEFT_HIP,
                            conf_threshold, h, w)
    elif region == "lower":
        box = _quad_bounds(local_xy, keypoints_conf,
                            LEFT_HIP, RIGHT_HIP, RIGHT_KNEE, LEFT_KNEE,
                            conf_threshold, h, w)
    elif region == "long":
        box = _quad_bounds(local_xy, keypoints_conf,
                            LEFT_SHOULDER, RIGHT_SHOULDER, RIGHT_KNEE, LEFT_KNEE,
                            conf_threshold, h, w)
    else:
        raise ValueError(f"Unknown region '{region}', expected 'upper', 'lower', or 'long'")

    if box is None:
        return None

    return _split_into_vertical_subregions(box, num_subregions)
