"""
All cv2 drawing logic lives here: the full-person box, the upper/lower
sub-region boxes (color-coded per predicted class), and label overlays.
Keeping this separate from pipeline logic means UI changes never touch
the classification code.

Color sample box drawing has been removed -- the profile no longer
contains color fields or color_sample_box / upper_color_box / lower_color_box
keys, since color extraction was stripped from attribute_profile.py.
"""

import cv2

import config


def _color_for_class(class_name):
    if class_name is None:
        return config.CLASS_COLORS_BGR["unknown"]
    return config.CLASS_COLORS_BGR.get(class_name, config.CLASS_COLORS_BGR["unknown"])


def _put_label(frame, text, x, y, color_bgr, font_scale=0.45):
    """Draws a small filled background rect behind text for readability."""
    (text_w, text_h), baseline = cv2.getTextSize(
        text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1
    )
    cv2.rectangle(frame, (x, y - text_h - 4), (x + text_w + 4, y + baseline), color_bgr, -1)
    cv2.putText(frame, text, (x + 2, y - 2),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), 1, cv2.LINE_AA)


def draw_person_annotation(frame, bbox, track_id, profile):
    """
    Draws:
      - the full person bbox (color = garment_type class color)
      - sub-region boxes for upper/lower (color = predicted class), if present
      - text labels for track ID, garment type, and each sub-classification
    """
    x1, y1, x2, y2 = bbox

    garment_type = profile.get("garment_type")
    main_color = _color_for_class(garment_type)

    cv2.rectangle(frame, (x1, y1), (x2, y2), main_color, 2)
    _put_label(frame, f"ID {track_id} | {garment_type}", x1, y1 - 6, main_color)

    if "long_type" in profile:
        lt = profile["long_type"]
        label = f"{lt['class']} ({lt['confidence']:.2f})"
        _put_label(frame, label, x1, y2 + 18, _color_for_class(lt["class"]))

    else:
        upper = profile.get("upper", {})
        lower = profile.get("lower", {})
        upper_box = profile.get("upper_box")
        lower_box = profile.get("lower_box")

        if upper_box is not None:
            uy1, uy2 = upper_box
            upper_color = _color_for_class(upper.get("class"))
            cv2.rectangle(frame, (x1, y1 + uy1), (x2, y1 + uy2), upper_color, 1)
            upper_label = f"U: {upper.get('class')} ({upper.get('confidence', 0):.2f})"
            _put_label(frame, upper_label, x1, y1 + uy2 - 4, upper_color, font_scale=0.4)

        if lower_box is not None:
            ly1, ly2 = lower_box
            lower_color = _color_for_class(lower.get("class"))
            cv2.rectangle(frame, (x1, y1 + ly1), (x2, y1 + ly2), lower_color, 1)
            lower_label = f"L: {lower.get('class')} ({lower.get('confidence', 0):.2f})"
            _put_label(frame, lower_label, x1, y2 + 18, lower_color, font_scale=0.4)

    return frame


def draw_fps(frame, fps):
    cv2.putText(frame, f"FPS: {fps:.1f}", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
    return frame
