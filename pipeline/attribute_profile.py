"""
Runs the full hierarchical classification tree on one person crop and
builds an attribute profile dict, including the upper/lower sub-region
boxes (so the UI layer can draw them) and pose-anchored multi-region
color extraction.

Two INDEPENDENT pose-related concerns, kept deliberately separate:

  1. UPPER/LOWER GARMENT-TYPE SPLITTING -- controlled by config.SPLIT_METHOD
     ("fixed_ratio" / "yolo_detector" / "pose"). Currently settled on
     "yolo_detector" (vision.splitting_yolo), trained specifically for
     this task.

  2. COLOR SAMPLING -- ALWAYS pose-anchored now, regardless of
     SPLIT_METHOD. Uses vision.pose_color_regions to find a shoulder-hip
     quad (upper) and hip-knee quad (lower) from keypoints, splits each
     into sub-regions, and votes across them via
     vision.color.extract_color_multi_region. If keypoints aren't usable
     for a given sample, color silently comes back None for that sample
     rather than falling back to the old fixed-box method -- the
     aggregator's confidence-weighted voting (pipeline/track_aggregator.py)
     already tolerates missing per-sample color gracefully.

keypoints_xy/keypoints_conf come from pipeline.tracker.track_frame(),
which already runs the pose model every frame as part of normal person
tracking -- so this does NOT add a second model load. It just uses
keypoints that were already being computed.
"""

import config
from models.classify import classify
from vision.splitting import split_upper_lower as _split_fixed_ratio
from vision.color import extract_color_multi_region
from vision.pose_color_regions import compute_color_subregions


def _split_upper_lower(person_crop_bgr, keypoints_xy, keypoints_conf, crop_origin):
    """
    Dispatches to whichever GARMENT-TYPE split method config.SPLIT_METHOD
    selects. Returns (upper_crop, lower_crop, upper_box, lower_box,
    split_method_used).
    """
    method = config.SPLIT_METHOD

    if method == "yolo_detector":
        from vision.splitting_yolo import split_upper_lower_yolo
        upper_crop, lower_crop, upper_box, lower_box = split_upper_lower_yolo(person_crop_bgr)
        return upper_crop, lower_crop, upper_box, lower_box, "yolo_detector"

    if method == "pose":
        from vision.pose_splitting import split_upper_lower_pose
        split_result = None
        if keypoints_xy is not None:
            split_result = split_upper_lower_pose(
                person_crop_bgr, keypoints_xy, keypoints_conf, crop_origin
            )
        if split_result is not None:
            upper_crop, lower_crop, upper_box, lower_box = split_result
            return upper_crop, lower_crop, upper_box, lower_box, "pose"
        upper_crop, lower_crop, upper_box, lower_box = _split_fixed_ratio(person_crop_bgr)
        return upper_crop, lower_crop, upper_box, lower_box, "fixed_ratio_fallback"

    upper_crop, lower_crop, upper_box, lower_box = _split_fixed_ratio(person_crop_bgr)
    return upper_crop, lower_crop, upper_box, lower_box, "fixed_ratio"


def _extract_pose_color(person_crop_bgr, keypoints_xy, keypoints_conf, crop_origin, region):
    """
    Always-pose-anchored color extraction for one region ("upper" or
    "lower"). Returns the color dict from extract_color_multi_region(),
    or None if keypoints weren't usable for this sample.
    """
    subregion_boxes = compute_color_subregions(
        person_crop_bgr, keypoints_xy, keypoints_conf, crop_origin, region=region
    )
    if subregion_boxes is None:
        return None

    return extract_color_multi_region(person_crop_bgr, subregion_boxes)


def build_attribute_profile(person_crop_bgr, models_dict, keypoints_xy=None,
                             keypoints_conf=None, crop_origin=(0, 0)):
    """
    models_dict is the dict returned by models.loaders.load_all_models():
        {"c1": (model, classes), "c2": (...), "c3": (...), "c4": (...)}

    keypoints_xy / keypoints_conf: this person's pose keypoints from
    pipeline.tracker.track_frame(), in FULL-FRAME coordinates, or None if
    pose data isn't available for this detection. crop_origin is the
    (x1, y1) of person_crop_bgr's bbox in the full frame.

    Returns a profile dict. For STANDARD garments, includes "upper_box" /
    "lower_box" (from whichever SPLIT_METHOD is active) plus a "color"
    field per region (always pose-anchored, independent of SPLIT_METHOD).
    "split_method" records which GARMENT-TYPE split method actually ran.
    """
    c1_model, c1_classes = models_dict["c1"]
    c2_model, c2_classes = models_dict["c2"]
    c3_model, c3_classes = models_dict["c3"]
    c4_model, c4_classes = models_dict["c4"]

    profile = {}

    garment_type, c1_conf = classify(c1_model, c1_classes, person_crop_bgr)
    profile["garment_type"] = garment_type
    profile["garment_type_conf"] = round(c1_conf, 3)

    is_long = garment_type is not None and "long" in garment_type.lower()

    if is_long:
        long_type, c2_conf = classify(c2_model, c2_classes, person_crop_bgr)
        long_color = _extract_pose_color(
            person_crop_bgr, keypoints_xy, keypoints_conf, crop_origin, region="long"
        )
        profile["long_type"] = {
            "class": long_type,
            "confidence": round(c2_conf, 3),
            "color": long_color,
        }
    else:
        upper_crop, lower_crop, upper_box, lower_box, split_method = _split_upper_lower(
            person_crop_bgr, keypoints_xy, keypoints_conf, crop_origin
        )
        profile["split_method"] = split_method

        upper_class, c3_conf = classify(c3_model, c3_classes, upper_crop)
        lower_class, c4_conf = classify(c4_model, c4_classes, lower_crop)

        upper_color = _extract_pose_color(
            person_crop_bgr, keypoints_xy, keypoints_conf, crop_origin, region="upper"
        )
        lower_color = _extract_pose_color(
            person_crop_bgr, keypoints_xy, keypoints_conf, crop_origin, region="lower"
        )

        profile["upper"] = {
            "class": upper_class,
            "confidence": round(c3_conf, 3),
            "color": upper_color,
        }
        profile["lower"] = {
            "class": lower_class,
            "confidence": round(c4_conf, 3),
            "color": lower_color,
        }
        profile["upper_box"] = upper_box
        profile["lower_box"] = lower_box

    return profile
