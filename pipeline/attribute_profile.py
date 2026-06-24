"""
Runs the full hierarchical classification tree on one person crop and
builds an attribute profile dict, including the upper/lower sub-region
boxes (so the UI layer can draw them).

Color extraction has been removed entirely for now -- no color fields
in the profile, no calls into vision.color. Re-add later if needed; the
color module itself is untouched on disk.

Splitting method is selected via config.SPLIT_METHOD ("fixed_ratio",
"yolo_detector", or "pose"), so each can be tested on the live feed one
at a time without juggling multiple models in memory at once. All three
return the same (upper_crop, lower_crop, upper_box, lower_box) shape, so
nothing downstream needs to know which one ran -- except "pose", which
additionally needs this detection's keypoints to do anything pose-based;
without them it silently behaves like "fixed_ratio" for that frame.
"""

import config
from models.classify import classify
from vision.splitting import split_upper_lower as _split_fixed_ratio


def _split_upper_lower(person_crop_bgr, keypoints_xy, keypoints_conf, crop_origin):
    """
    Dispatches to whichever split method config.SPLIT_METHOD selects.
    Returns (upper_crop, lower_crop, upper_box, lower_box, split_method_used)
    -- the last element records what ACTUALLY ran this time (useful for
    "pose" mode, which can silently fall back per-frame).
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


def build_attribute_profile(person_crop_bgr, models_dict, keypoints_xy=None,
                             keypoints_conf=None, crop_origin=(0, 0)):
    """
    models_dict is the dict returned by models.loaders.load_all_models():
        {"c1": (model, classes), "c2": (...), "c3": (...), "c4": (...)}

    keypoints_xy / keypoints_conf: this person's pose keypoints from
    pipeline.tracker.track_frame(), in FULL-FRAME coordinates, or None if
    pose data isn't available for this detection (or config.SPLIT_METHOD
    isn't "pose", in which case these are simply unused). crop_origin is
    the (x1, y1) of person_crop_bgr's bbox in the full frame, needed to
    convert those keypoints into crop-local coordinates.

    Returns a profile dict. For STANDARD garments, also includes
    "upper_box" / "lower_box" (y_start, y_end) tuples relative to the
    person crop, so the drawing layer can render sub-region rectangles,
    plus "split_method" recording which method actually produced this
    sample's boxes.
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
        profile["long_type"] = {
            "class": long_type,
            "confidence": round(c2_conf, 3),
        }
    else:
        upper_crop, lower_crop, upper_box, lower_box, split_method = _split_upper_lower(
            person_crop_bgr, keypoints_xy, keypoints_conf, crop_origin
        )
        profile["split_method"] = split_method

        upper_class, c3_conf = classify(c3_model, c3_classes, upper_crop)
        lower_class, c4_conf = classify(c4_model, c4_classes, lower_crop)

        profile["upper"] = {
            "class": upper_class,
            "confidence": round(c3_conf, 3),
        }
        profile["lower"] = {
            "class": lower_class,
            "confidence": round(c4_conf, 3),
        }
        profile["upper_box"] = upper_box
        profile["lower_box"] = lower_box

    return profile
