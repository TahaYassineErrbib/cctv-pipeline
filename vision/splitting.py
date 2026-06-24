"""
Splits a full-person crop into upper and lower body regions using a simple
geometric ratio (no pose model needed). A slight overlap around the waist
is intentional — classifiers are robust to a bit of slack, and this avoids
hard-cutting through a garment right at the boundary.
"""

import config


def split_upper_lower(person_crop_bgr):
    """
    Returns (upper_crop, lower_crop, upper_box, lower_box).

    upper_box / lower_box are (y_start, y_end) tuples relative to the
    person crop, so callers can draw sub-region rectangles on the original
    frame if they offset by the person's bbox origin.
    """
    h, w = person_crop_bgr.shape[:2]
    upper_end = int(h * config.UPPER_SPLIT_RATIO)
    lower_start = int(h * (1 - config.LOWER_SPLIT_RATIO))

    upper_crop = person_crop_bgr[0:upper_end, :]
    lower_crop = person_crop_bgr[lower_start:h, :]

    upper_box = (0, upper_end)
    lower_box = (lower_start, h)

    return upper_crop, lower_crop, upper_box, lower_box
