"""
debug/test_on_images.py

Runs the full pipeline (person detection -> garment-type split via
config.SPLIT_METHOD -> C1-C4 classification -> pose-anchored multi-region
color) on a folder of still images, instead of a live/video feed.

Usage:
    python debug/test_on_images.py --images_dir debug/sample_crops

Each image is treated as a FULL FRAME (not a pre-cropped person) -- person
detection runs on it first, same as main.py would on a video frame.

For each detected person, saves:
    debug/output/test_images/<image_name>_annotated.jpg

And prints the full attribute profile dict per person to the console.
"""

import argparse
import glob
import json
import os
import sys

import cv2

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from models.loaders import load_all_models
from pipeline.attribute_profile import build_attribute_profile
from ui.draw import draw_person_annotation


def run(images_dir, out_dir, person_conf_threshold):
    os.makedirs(out_dir, exist_ok=True)

    print("Loading models (this loads YOLO-pose + box detector + C1-C4)...")
    models_dict = load_all_models()
    print(f"Models loaded. SPLIT_METHOD = '{config.SPLIT_METHOD}'\n")

    image_paths = sorted(
        glob.glob(os.path.join(images_dir, "*.jpg")) +
        glob.glob(os.path.join(images_dir, "*.png")) +
        glob.glob(os.path.join(images_dir, "*.jpeg"))
    )

    if not image_paths:
        print(f"No images found in {images_dir} (looked for .jpg/.jpeg/.png)")
        return

    all_profiles = []

    for path in image_paths:
        frame = cv2.imread(path)
        if frame is None:
            print(f"  Could not read {path}, skipping.")
            continue

        name = os.path.splitext(os.path.basename(path))[0]
        print(f"--- {name} ---")

        yolo_model = models_dict["yolo"]
        results = yolo_model.predict(frame, verbose=False)[0]

        if results.boxes is None or len(results.boxes) == 0:
            print("  No persons detected.")
            continue

        boxes = results.boxes
        keypoints = getattr(results, "keypoints", None)

        person_count = 0
        for i in range(len(boxes)):
            cls_id = int(boxes.cls[i])
            conf = float(boxes.conf[i])
            if cls_id != config.PERSON_CLASS_ID or conf < person_conf_threshold:
                continue

            x1, y1, x2, y2 = map(int, boxes.xyxy[i])
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)

            person_crop = frame[y1:y2, x1:x2]
            if person_crop.size == 0:
                continue

            kpt_xy, kpt_conf = None, None
            if keypoints is not None and keypoints.xy is not None and i < len(keypoints.xy):
                kpt_xy = keypoints.xy[i].cpu().numpy()
                kpt_conf = keypoints.conf[i].cpu().numpy() if keypoints.conf is not None else None

            profile = build_attribute_profile(
                person_crop, models_dict,
                keypoints_xy=kpt_xy, keypoints_conf=kpt_conf, crop_origin=(x1, y1),
            )

            person_count += 1
            print(f"  Person {person_count} (bbox={x1},{y1},{x2},{y2}, det_conf={conf:.2f}):")
            print(f"    {json.dumps(profile, indent=4, default=str)}")

            draw_person_annotation(frame, (x1, y1, x2, y2), person_count, profile)

            all_profiles.append({
                "image": os.path.basename(path),
                "person_idx": person_count,
                "bbox": [x1, y1, x2, y2],
                "profile": profile,
            })

        if person_count == 0:
            print("  No persons above confidence threshold.")

        out_path = os.path.join(out_dir, f"{name}_annotated.jpg")
        cv2.imwrite(out_path, frame)
        print(f"  Saved: {out_path}\n")

    summary_path = os.path.join(out_dir, "all_profiles.json")
    with open(summary_path, "w") as f:
        json.dump(all_profiles, f, indent=2, default=str)

    print(f"Done. {len(image_paths)} images processed, {len(all_profiles)} person profiles built.")
    print(f"Annotated images + summary JSON in: {out_dir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test the full attribute pipeline on a folder of still images.")
    parser.add_argument("--images_dir", default="debug/sample_images",
                         help="Folder of FULL-FRAME images (not pre-cropped persons).")
    parser.add_argument("--out_dir", default="debug/output/test_images")
    parser.add_argument("--person_conf", type=float, default=None,
                         help="Override config.PERSON_CONF_THRESHOLD for this run.")
    args = parser.parse_args()

    threshold = args.person_conf if args.person_conf is not None else config.PERSON_CONF_THRESHOLD

    if not os.path.isdir(args.images_dir):
        print(f"ERROR: images_dir not found: {args.images_dir}")
        sys.exit(1)

    run(args.images_dir, args.out_dir, threshold)
