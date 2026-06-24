"""
Main entrypoint for the CCTV attribute pipeline.

Per frame:
    - run tracker every frame (cheap, keeps IDs consistent)
    - filter detections to the active zone (ignore people near borders)
    - every PROCESS_EVERY_N_FRAMES frames, for each in-zone, non-occluded,
      not-yet-locked tracked person:
        - build attribute profile (the expensive part)
        - feed it to the TrackAggregator (confidence-weighted majority vote)
        - if this sample causes the track to lock, the aggregator returns
          the final frozen profile
    - locked tracks skip classification entirely from then on — just reuse
      the frozen profile (this is a real speedup on a weak GPU, since most
      of a video's runtime is spent on tracks that have already stabilized)
    - draw color-coded boxes + labels, save snapshot, append JSON record

Video source can be a local file, webcam index (0), or a YouTube URL
(set config.USE_YOUTUBE_STREAM = True and put the URL in config.VIDEO_SOURCE).

CHANGED: pipeline.tracker now runs a yolov8-pose model, so each detection
also carries pose keypoints. Those are passed into build_attribute_profile
so the standard-garment branch can use pose-based upper/lower splitting
(vision/pose_splitting.py) instead of always using the fixed 55%/45%
geometric split. No other control flow changes.

Run with: python main.py
"""

import os
import time

import cv2

import config
from models.loaders import load_all_models
from pipeline.tracker import track_frame
from pipeline.attribute_profile import build_attribute_profile
from pipeline.track_aggregator import TrackAggregator
from pipeline.active_zone import compute_active_zone, is_inside_zone, draw_active_zone
from pipeline.youtube_source import resolve_video_source
from storage.json_store import append_record
from ui.draw import draw_person_annotation, draw_fps


def save_snapshot(frame, frame_idx):
    if not config.SAVE_SNAPSHOTS:
        return
    path = os.path.join(config.SNAPSHOT_DIR, f"frame_{frame_idx:06d}.jpg")
    cv2.imwrite(path, frame)


def main():
    models_dict = load_all_models()
    aggregator = TrackAggregator(samples_required=10)

    video_source = resolve_video_source(config.VIDEO_SOURCE)

    print("Opening video source:", video_source)
    cap = cv2.VideoCapture(video_source)

    if not cap.isOpened():
        print(f"ERROR: could not open video source: {video_source}")
        return

    frame_idx = 0
    t_start = time.time()

    # cache of last-known (pre-lock) profile per track_id, so non-sampled
    # frames can still draw something reasonable instead of a bare box
    last_known_profiles = {}

    # active zone is computed once we see the first real frame, since it
    # depends on the actual frame dimensions (works for any resolution)
    active_zone = None

    while True:
        ret, frame = cap.read()
        if not ret:
            print("End of video stream or cannot read frame.")
            break

        frame_idx += 1

        if config.USE_ACTIVE_ZONE and active_zone is None:
            h, w = frame.shape[:2]
            active_zone = compute_active_zone(w, h)
            print(f"Active zone computed for {w}x{h} frame: {active_zone}")

        detections = track_frame(models_dict["yolo"], frame)

        if config.USE_ACTIVE_ZONE:
            detections = [d for d in detections if is_inside_zone(d["bbox"], active_zone)]

        do_classify = (frame_idx % config.PROCESS_EVERY_N_FRAMES == 0)

        for det in detections:
            track_id = det["track_id"]
            occluded = det.get("occluded", False)
            x1, y1, x2, y2 = det["bbox"]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)

            person_crop = frame[y1:y2, x1:x2]
            if person_crop.size == 0:
                continue

            if aggregator.is_locked(track_id):
                # identity is settled — skip the expensive classifier calls
                # entirely and just reuse the frozen profile
                profile = aggregator.get_locked_profile(track_id)

            elif do_classify and not occluded:
                # only sample when this person is NOT overlapping another
                # tracked person right now — an occluded crop is likely a
                # blend of two people and would poison the majority vote.
                # Tracking itself (the track_id) is completely unaffected by
                # this skip; we just don't trust this particular frame's
                # crop enough to classify it.
                #
                # CHANGED: pass this detection's pose keypoints (full-frame
                # coords) + crop_origin through, so the standard-garment
                # branch can attempt a pose-based upper/lower split before
                # falling back to the geometric one. keypoints_xy/_conf are
                # None if pose data wasn't available for this detection —
                # build_attribute_profile handles that fallback internally.
                profile = build_attribute_profile(
                    person_crop,
                    models_dict,
                    keypoints_xy=det.get("keypoints_xy"),
                    keypoints_conf=det.get("keypoints_conf"),
                    crop_origin=(x1, y1),
                )
                last_known_profiles[track_id] = profile

                locked_result = aggregator.add_sample(track_id, profile)
                if locked_result is not None:
                    profile = locked_result  # use the frozen profile from now on
                    print(f"[track {track_id}] LOCKED after "
                          f"{aggregator.samples_required} classified samples: "
                          f"garment_type={profile.get('garment_type')}")

                record = {
                    "track_id": track_id,
                    "frame_idx": frame_idx,
                    "timestamp": round(time.time() - t_start, 3),
                    "camera_id": "cam_0",  # placeholder until multi-camera setup
                    "bbox": [x1, y1, x2, y2],
                    "profile": profile,
                    "locked": aggregator.is_locked(track_id),
                }
                append_record(record)

            else:
                # either not a sampled frame, OR this person IS occluded
                # right now (do_classify True but occluded True falls here
                # too) — reuse last-known profile, don't advance progress
                profile = last_known_profiles.get(track_id)

            if profile is not None:
                draw_person_annotation(frame, (x1, y1, x2, y2), track_id, profile)
            elif occluded:
                # no profile yet AND currently occluded — draw a distinct
                # dashed-style indicator so it's visually clear why this
                # person has no label yet, instead of looking like a bug
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 1)
                cv2.putText(frame, "occluded", (x1, y1 - 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1, cv2.LINE_AA)
            else:
                # no classification yet for this track (first sighting,
                # not-yet-sampled) — draw a plain box so it's still visible
                cv2.rectangle(frame, (x1, y1), (x2, y2), config.DEFAULT_BOX_COLOR_BGR, 2)

        if do_classify:
            save_snapshot(frame, frame_idx)

        elapsed = time.time() - t_start
        fps = frame_idx / elapsed if elapsed > 0 else 0.0
        draw_fps(frame, fps)

        cv2.imshow("CCTV Attribute Pipeline", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            print("Quit requested.")
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"\nDone. Profiles saved to: {config.PROFILES_JSON_PATH}")
    print(f"Snapshots saved to: {config.SNAPSHOT_DIR}")


if __name__ == "__main__":
    main()
