"""
All cv2 drawing logic lives here: the full-person box, the upper/lower
sub-region boxes (color-coded per predicted class), and label overlays.
Keeping this separate from pipeline logic means UI changes never touch
the classification code.
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


def _draw_sample_box(frame, origin_x, origin_y, box_coords, label=""):
    """
    Draws a small debug rectangle for a color sample box.
    box_coords = (x1, y1, x2, y2) relative to some local crop; origin_x/origin_y
    is where that local crop's (0,0) lands in the full frame, so we just add.
    """
    if box_coords is None:
        return
    bx1, by1, bx2, by2 = box_coords
    cv2.rectangle(
        frame,
        (origin_x + bx1, origin_y + by1),
        (origin_x + bx2, origin_y + by2),
        (0, 0, 255),  # fixed bright red, distinct from any class color, easy to spot
        1,
    )
    if label:
        cv2.putText(frame, label, (origin_x + bx1, origin_y + by1 - 3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1, cv2.LINE_AA)


def draw_person_annotation(frame, bbox, track_id, profile, show_color_sample_boxes=True):
    """
    Draws:
      - the full person bbox (color = garment_type class color)
      - sub-region boxes for upper/lower (color = predicted class), if present
      - text labels for track ID, garment type, and each sub-classification
      - (optional) the exact color sample box used for k-means, in red, so
        you can visually verify/tune whether it's landing in the right spot
    """
    x1, y1, x2, y2 = bbox

    garment_type = profile.get("garment_type")
    main_color = _color_for_class(garment_type)

    # full person box
    cv2.rectangle(frame, (x1, y1), (x2, y2), main_color, 2)
    _put_label(frame, f"ID {track_id} | {garment_type}", x1, y1 - 6, main_color)

    if "long_type" in profile:
        lt = profile["long_type"]
        label = f"{lt['class']} ({lt['confidence']:.2f}) {lt['color']['name'] if lt['color'] else ''}"
        _put_label(frame, label, x1, y2 + 18, _color_for_class(lt["class"]))

        # color_sample_box is relative to the FULL person crop, so origin is
        # simply the person bbox's own (x1, y1) — no extra sub-crop offset.
        if show_color_sample_boxes:
            _draw_sample_box(frame, x1, y1, profile.get("color_sample_box"), label="color")

    else:
        upper = profile.get("upper", {})
        lower = profile.get("lower", {})
        upper_box = profile.get("upper_box")
        lower_box = profile.get("lower_box")

        # upper sub-region box, offset by the person bbox's y1
        if upper_box is not None:
            uy1, uy2 = upper_box
            upper_color = _color_for_class(upper.get("class"))
            cv2.rectangle(frame, (x1, y1 + uy1), (x2, y1 + uy2), upper_color, 1)
            upper_label = f"U: {upper.get('class')} ({upper.get('confidence', 0):.2f})"
            if upper.get("color"):
                upper_label += f" {upper['color']['name']}"
            _put_label(frame, upper_label, x1, y1 + uy2 - 4, upper_color, font_scale=0.4)

            # upper_color_box is relative to upper_crop, which itself starts
            # at (x1, y1 + uy1) in the full frame — so that's our origin here.
            if show_color_sample_boxes:
                _draw_sample_box(frame, x1, y1 + uy1, profile.get("upper_color_box"))

        # lower sub-region box, offset by the person bbox's y1
        if lower_box is not None:
            ly1, ly2 = lower_box
            lower_color = _color_for_class(lower.get("class"))
            cv2.rectangle(frame, (x1, y1 + ly1), (x2, y1 + ly2), lower_color, 1)
            lower_label = f"L: {lower.get('class')} ({lower.get('confidence', 0):.2f})"
            if lower.get("color"):
                lower_label += f" {lower['color']['name']}"
            _put_label(frame, lower_label, x1, y2 + 18, lower_color, font_scale=0.4)

            # lower_color_box is relative to lower_crop, which starts at
            # (x1, y1 + ly1) in the full frame — that's our origin here.
            if show_color_sample_boxes:
                _draw_sample_box(frame, x1, y1 + ly1, profile.get("lower_color_box"))

    return frame


def draw_fps(frame, fps):
    cv2.putText(frame, f"FPS: {fps:.1f}", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
    return frame