"""
Active zone: a centered rectangular region of the frame, expressed as
percentage margins, used to ignore detections near the camera borders
(people entering/leaving frame, partially visible, lens-edge distortion).

The zone is computed fresh per frame from the frame's actual dimensions,
so it automatically adapts to any video resolution without hardcoding
pixel values.
"""

import cv2

import config


def compute_active_zone(frame_width, frame_height,
                         margin_x=None, margin_y=None):
    """
    Returns (x1, y1, x2, y2) bounding the active zone in pixel coordinates
    for a frame of the given size.
    """
    margin_x = config.ACTIVE_ZONE_MARGIN_X if margin_x is None else margin_x
    margin_y = config.ACTIVE_ZONE_MARGIN_Y if margin_y is None else margin_y

    x1 = int(frame_width * margin_x)
    x2 = int(frame_width * (1 - margin_x))
    y1 = int(frame_height * margin_y)
    y2 = int(frame_height * (1 - margin_y))

    return (x1, y1, x2, y2)


def is_inside_zone(bbox, zone):
    """
    Returns True if the CENTER of bbox (x1, y1, x2, y2) falls inside zone
    (zx1, zy1, zx2, zy2). Using the center rather than requiring the full
    bbox to be inside means a person partially crossing the zone boundary
    (e.g. half their body still outside) is still counted once their
    midpoint has crossed in — avoids flickering in/out right at the edge.
    """
    bx1, by1, bx2, by2 = bbox
    zx1, zy1, zx2, zy2 = zone

    cx = (bx1 + bx2) / 2
    cy = (by1 + by2) / 2

    return zx1 <= cx <= zx2 and zy1 <= cy <= zy2


def draw_active_zone(frame, zone):
    """Draws the active zone boundary on the frame for visual confirmation."""
    x1, y1, x2, y2 = zone
    cv2.rectangle(frame, (x1, y1), (x2, y2), config.ACTIVE_ZONE_COLOR_BGR, 2)
    cv2.putText(frame, "active zone", (x1 + 4, y1 + 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, config.ACTIVE_ZONE_COLOR_BGR, 1, cv2.LINE_AA)
    return frame
