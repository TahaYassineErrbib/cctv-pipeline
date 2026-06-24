"""
Pose-based upper/lower body splitting, using YOLOv8-pose keypoints instead
of the fixed 55%/45% geometric ratio in vision/splitting.py.

Why: the geometric split assumes a person is standing upright, centered,
and fully visible in the crop. CCTV footage breaks all three assumptions
routinely (bent poses, partial crops, crouching, walking mid-stride).
Anchoring the split to actual shoulder/hip/ankle keypoints adapts to the
real pose instead of guessing a fixed ratio.

COCO-17 keypoint indices used:
    5  = left_shoulder    6  = right_shoulder
    11 = left_hip         12 = right_hip
    15 = left_ankle       16 = right_ankle

Boundary logic (mirrors the geometric version's intent, just anatomy-
anchored instead of ratio-anchored):
    upper region: shoulders -> hips   (with a small margin above/below)
    lower region: hips -> ankles      (falls back to crop bottom if ankles
                                        aren't visible, e.g. legs cut off
                                        by the camera angle or crop edge)

This module never crashes on bad input — it returns None when keypoints
aren't trustworthy enough to use, so callers fall back to the geometric
split (vision/splitting.py) instead of breaking the pipeline.
"""

import numpy as np

import config


LEFT_SHOULDER, RIGHT_SHOULDER = 5, 6
LEFT_HIP, RIGHT_HIP = 11, 12
LEFT_ANKLE, RIGHT_ANKLE = 15, 16

# Below this per-keypoint confidence, treat the point as "not visible"
# rather than trusting a noisy/guessed coordinate.
KEYPOINT_CONF_THRESHOLD = getattr(config, "POSE_KEYPOINT_CONF_THRESHOLD", 0.5)

# Small vertical margins so the upper/lower boxes keep a sliver of overlap
# around the shoulder/hip line and hip/ankle line, same motivation as the
# original geometric split's intentional overlap around the waist.
SHOULDER_MARGIN_PCT = 0.05
HIP_MARGIN_PCT = 0.05


def _avg_point(kpt_xy, kpt_conf, idx_a, idx_b, conf_threshold):
    """
    Averages a left/right keypoint pair (e.g. left+right shoulder) when
    both are confident enough. Falls back to whichever single side is
    confident if only one is usable. Returns None if neither is usable.
    """
    pa, ca = kpt_xy[idx_a], kpt_conf[idx_a]
    pb, cb = kpt_xy[idx_b], kpt_conf[idx_b]

    a_ok = ca >= conf_threshold
    b_ok = cb >= conf_threshold

    if a_ok and b_ok:
        return (pa + pb) / 2.0
    if a_ok:
        return pa
    if b_ok:
        return pb
    return None


def compute_pose_boxes(person_crop_bgr, keypoints_xy, keypoints_conf,
                        crop_origin, conf_threshold=KEYPOINT_CONF_THRESHOLD):
    """
    Computes upper/lower boxes for ONE person from their pose keypoints.

    Args:
        person_crop_bgr: the person's cropped image (only used for shape).
        keypoints_xy: (17, 2) array of (x, y) pixel coords, in FULL FRAME
            coordinates (this is how ultralytics returns them).
        keypoints_conf: (17,) array of per-keypoint confidence.
        crop_origin: (x1, y1) of this person's bbox in the full frame, so
            frame-space keypoints can be converted to crop-local coords.
        conf_threshold: minimum confidence to trust a keypoint pair.

    Returns:
        (upper_box, lower_box) as (y_start, y_end) tuples relative to the
        person crop -- same contract as vision.splitting.split_upper_lower
        -- or None if keypoints weren't usable (caller should fall back to
        the geometric split).
    """
    if keypoints_xy is None or keypoints_conf is None:
        return None

    h, w = person_crop_bgr.shape[:2]
    if h == 0 or w == 0:
        return None

    ox, oy = crop_origin

    # convert frame-space keypoints to crop-local coords
    local_xy = np.asarray(keypoints_xy, dtype=np.float32).copy()
    local_xy[:, 0] -= ox
    local_xy[:, 1] -= oy

    shoulder = _avg_point(local_xy, keypoints_conf, LEFT_SHOULDER, RIGHT_SHOULDER, conf_threshold)
    hip = _avg_point(local_xy, keypoints_conf, LEFT_HIP, RIGHT_HIP, conf_threshold)
    ankle = _avg_point(local_xy, keypoints_conf, LEFT_ANKLE, RIGHT_ANKLE, conf_threshold)

    # shoulders + hips are the minimum we need; without both there's no
    # reliable upper/lower boundary at all, so bail out to the geometric
    # fallback rather than guessing.
    if shoulder is None or hip is None:
        return None

    shoulder_y = float(shoulder[1])
    hip_y = float(hip[1])

    if hip_y <= shoulder_y:
        # degenerate pose (upside down, bad detection, etc.) -- don't trust it
        return None

    margin = (hip_y - shoulder_y) * SHOULDER_MARGIN_PCT

    upper_start = max(0, int(shoulder_y - margin))
    upper_end = min(h, int(hip_y + margin))

    # lower region: hip -> ankle if ankle is visible and below the hip,
    # otherwise fall back to the crop's bottom edge (legs cut off by the
    # camera angle or crop boundary -- common in CCTV footage)
    if ankle is not None and float(ankle[1]) > hip_y:
        ankle_y = float(ankle[1])
        hip_margin = (ankle_y - hip_y) * HIP_MARGIN_PCT
        lower_start = max(0, int(hip_y - hip_margin))
        lower_end = min(h, int(ankle_y))
    else:
        lower_start = max(0, int(hip_y - margin))
        lower_end = h

    if upper_end <= upper_start or lower_end <= lower_start:
        return None

    return (upper_start, upper_end), (lower_start, lower_end)


def split_upper_lower_pose(person_crop_bgr, keypoints_xy, keypoints_conf, crop_origin):
    """
    Drop-in pose-based replacement for vision.splitting.split_upper_lower.

    Returns (upper_crop, lower_crop, upper_box, lower_box) on success, or
    None if pose keypoints weren't usable -- caller should then call
    vision.splitting.split_upper_lower(person_crop_bgr) as the fallback.
    """
    boxes = compute_pose_boxes(person_crop_bgr, keypoints_xy, keypoints_conf, crop_origin)
    if boxes is None:
        return None

    upper_box, lower_box = boxes
    uy1, uy2 = upper_box
    ly1, ly2 = lower_box

    upper_crop = person_crop_bgr[uy1:uy2, :]
    lower_crop = person_crop_bgr[ly1:ly2, :]

    if upper_crop.size == 0 or lower_crop.size == 0:
        return None

    return upper_crop, lower_crop, upper_box, lower_box
