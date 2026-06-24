"""
Wraps Ultralytics' built-in tracking (.track() with ByteTrack) so the rest
of the pipeline just asks for "detections with track IDs for this frame"
without caring about the underlying tracker config.

Also flags occlusion: when two tracked people's bboxes overlap significantly
in the same frame (e.g. one walking behind/across another), both are marked
"occluded" so the attribute-sampling stage can skip them for that frame
without breaking tracking continuity. ByteTrack keeps following the track ID
regardless — this flag only affects whether we trust the crop for
classification right now.
"""

import config


OCCLUSION_IOU_THRESHOLD = 0.15  # bbox overlap above this counts as "merging"


def _iou(box_a, box_b):
    """Standard intersection-over-union between two (x1, y1, x2, y2) boxes."""
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    if inter_area == 0:
        return 0.0

    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union_area = area_a + area_b - inter_area

    if union_area == 0:
        return 0.0

    return inter_area / union_area


def _flag_occlusions(detections, iou_threshold=OCCLUSION_IOU_THRESHOLD):
    """
    Marks each detection dict with "occluded": True/False based on whether
    its bbox overlaps any OTHER detection's bbox above iou_threshold.
    Mutates and returns the same list.
    """
    n = len(detections)
    for det in detections:
        det["occluded"] = False

    for i in range(n):
        for j in range(i + 1, n):
            iou = _iou(detections[i]["bbox"], detections[j]["bbox"])
            if iou > iou_threshold:
                detections[i]["occluded"] = True
                detections[j]["occluded"] = True

    return detections


def track_frame(yolo_model, frame):
    """
    Runs detection + tracking on a single frame.

    Returns a list of dicts:
        [{"track_id": int, "bbox": (x1, y1, x2, y2), "conf": float,
          "occluded": bool}, ...]

    Only "person" class detections above the configured confidence
    threshold are returned. Detections without a track ID yet (can happen
    on the very first frame or right after a track is lost/reacquired) are
    skipped — they'll get an ID on a subsequent frame.

    "occluded" is True if this person's bbox overlaps another tracked
    person's bbox above OCCLUSION_IOU_THRESHOLD in this same frame —
    signals to the rest of the pipeline that this crop probably contains a
    blend of two people and shouldn't be trusted for classification right
    now, even though tracking itself continues normally.
    """
    results = yolo_model.track(
        frame,
        persist=True,
        tracker=config.TRACKER_CONFIG,
        verbose=False,
    )[0]

    detections = []

    if results.boxes is None or results.boxes.id is None:
        return detections

    boxes = results.boxes
    for i in range(len(boxes)):
        cls_id = int(boxes.cls[i])
        conf = float(boxes.conf[i])

        if cls_id != config.PERSON_CLASS_ID or conf < config.PERSON_CONF_THRESHOLD:
            continue

        track_id = int(boxes.id[i])
        x1, y1, x2, y2 = map(int, boxes.xyxy[i])

        detections.append({
            "track_id": track_id,
            "bbox": (x1, y1, x2, y2),
            "conf": conf,
        })

    detections = _flag_occlusions(detections)

    return detections