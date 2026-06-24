"""
Runs the full hierarchical classification tree on one person crop and
builds an attribute profile dict, including color extraction and the
upper/lower sub-region boxes (so the UI layer can draw them).

CHANGED: the standard-garment branch now tries a pose-based upper/lower
split (vision.pose_splitting, anchored to shoulder/hip/ankle keypoints)
before falling back to the old fixed 55%/45% geometric split
(vision.splitting). Pose-based splitting adapts to the person's actual
posture; the geometric split is kept as a safety net for frames where
pose estimation didn't return usable keypoints (low confidence, person
partially out of frame, etc.) so the pipeline never breaks.
"""

from models.classify import classify
from vision.color import extract_color_for_region
from vision.splitting import split_upper_lower
from vision.pose_splitting import split_upper_lower_pose


def build_attribute_profile(person_crop_bgr, models_dict, keypoints_xy=None,
                             keypoints_conf=None, crop_origin=(0, 0)):
    """
    models_dict is the dict returned by models.loaders.load_all_models():
        {"c1": (model, classes), "c2": (...), "c3": (...), "c4": (...)}

    keypoints_xy / keypoints_conf: this person's pose keypoints from
    pipeline.tracker.track_frame(), in FULL-FRAME coordinates, or None if
    pose data isn't available for this detection. crop_origin is the
    (x1, y1) of person_crop_bgr's bbox in the full frame, needed to convert
    those keypoints into crop-local coordinates. All three are optional --
    omitting them just means the geometric split is used directly, same as
    before this change.

    Returns a profile dict. For STANDARD garments, also includes
    "upper_box" / "lower_box" (y_start, y_end) tuples relative to the
    person crop, so the drawing layer can render sub-region rectangles,
    plus "split_method" ("pose" or "geometric") so it's visible in the
    saved JSON which path was used for this sample.
    """
    c1_model, c1_classes = models_dict["c1"]
    c2_model, c2_classes = models_dict["c2"]
    c3_model, c3_classes = models_dict["c3"]
    c4_model, c4_classes = models_dict["c4"]

    profile = {}

    garment_type, c1_conf = classify(c1_model, c1_classes, person_crop_bgr)
    profile["garment_type"] = garment_type
    profile["garment_type_conf"] = round(c1_conf, 3)

    # NOTE: assumes C1's "long" class label literally contains "long".
    # Confirmed via runtime logs: classes=['long', 'standard'].
    is_long = garment_type is not None and "long" in garment_type.lower()

    if is_long:
        long_type, c2_conf = classify(c2_model, c2_classes, person_crop_bgr)
        long_color, long_color_box = extract_color_for_region(person_crop_bgr, region="long")
        profile["long_type"] = {
            "class": long_type,
            "confidence": round(c2_conf, 3),
            "color": long_color,
        }
        # color_box here is relative to the full person crop (since "long"
        # samples directly from person_crop_bgr, not a sub-crop)
        profile["color_sample_box"] = long_color_box
    else:
        split_result = None
        if keypoints_xy is not None:
            split_result = split_upper_lower_pose(
                person_crop_bgr, keypoints_xy, keypoints_conf, crop_origin
            )

        if split_result is not None:
            upper_crop, lower_crop, upper_box, lower_box = split_result
            profile["split_method"] = "pose"
        else:
            upper_crop, lower_crop, upper_box, lower_box = split_upper_lower(person_crop_bgr)
            profile["split_method"] = "geometric"

        upper_class, c3_conf = classify(c3_model, c3_classes, upper_crop)
        lower_class, c4_conf = classify(c4_model, c4_classes, lower_crop)

        upper_color, upper_color_box = extract_color_for_region(upper_crop, region="upper")
        lower_color, lower_color_box = extract_color_for_region(lower_crop, region="lower")

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

        # color_box coords are relative to upper_crop/lower_crop respectively
        # (sub-crops of person_crop_bgr), NOT relative to person_crop_bgr
        # directly. draw.py must offset by the region's own box origin, not
        # just the person bbox origin, when rendering these.
        profile["upper_color_box"] = upper_color_box
        profile["lower_color_box"] = lower_color_box

    return profile
