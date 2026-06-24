"""
Splits a full-person crop into upper and lower body regions.

Uses a trained YOLOv8n detector (upper_lower_detector.pt) to find the
actual upper/lower regions instead of a fixed 55/45 ratio. Falls back to
the old fixed-ratio split if the detector doesn't fire confidently on a
given crop (e.g. heavy occlusion, unusual pose, low-confidence noise) --
better to get a usable region than no region, or a bad one.
"""

import config
from ultralytics import YOLO

from vision.splitting import split_upper_lower as _fixed_ratio_split

_detector = None  # lazy-loaded singleton, avoid reloading per-call

MIN_DETECTION_CONF = 0.4  # below this, don't trust the box -- fall back instead


def _get_detector():
    global _detector
    if _detector is None:
        _detector = YOLO(config.UPPER_LOWER_DETECTOR_PATH)
    return _detector


def _best_box_per_class(results, min_conf):
    """
    Returns {0: xyxy_or_None, 1: xyxy_or_None} -- the highest-confidence box
    for each class, ignoring any box below min_conf.
    """
    best = {0: (None, 0.0), 1: (None, 0.0)}

    if results.boxes is None:
        return {0: None, 1: None}

    for box in results.boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        if cls_id not in best or conf < min_conf:
            continue
        if conf > best[cls_id][1]:
            best[cls_id] = (box.xyxy[0].tolist(), conf)

    return {cls_id: xyxy for cls_id, (xyxy, _) in best.items()}


def split_upper_lower_yolo(person_crop_bgr):
    """
    Returns (upper_crop, lower_crop, upper_box, lower_box) -- SAME shape as
    the fixed-ratio version, so callers need zero changes if they switch
    to this function instead.
    """
    h, w = person_crop_bgr.shape[:2]

    detector = _get_detector()
    results = detector.predict(person_crop_bgr, verbose=False)[0]

    boxes = _best_box_per_class(results, MIN_DETECTION_CONF)
    upper_box_xyxy = boxes.get(0)
    lower_box_xyxy = boxes.get(1)

    if upper_box_xyxy is None or lower_box_xyxy is None:
        return _fixed_ratio_split(person_crop_bgr)

    upper_y_start = max(0, int(upper_box_xyxy[1]))
    upper_y_end = min(h, int(upper_box_xyxy[3]))
    lower_y_start = max(0, int(lower_box_xyxy[1]))
    lower_y_end = min(h, int(lower_box_xyxy[3]))

    upper_crop = person_crop_bgr[upper_y_start:upper_y_end, :]
    lower_crop = person_crop_bgr[lower_y_start:lower_y_end, :]

    if upper_crop.size == 0 or lower_crop.size == 0:
        return _fixed_ratio_split(person_crop_bgr)

    upper_box = (upper_y_start, upper_y_end)
    lower_box = (lower_y_start, lower_y_end)

    return upper_crop, lower_crop, upper_box, lower_box
