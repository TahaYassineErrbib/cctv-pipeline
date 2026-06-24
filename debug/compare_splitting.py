"""
debug/compare_splitting.py

Compares the three upper/lower splitting strategies WITHOUT needing all
three loaded in memory at once -- run this once per --mode, on the SAME
folder of test crops, then compare the saved results across runs.

Why this exists: running the person-detector (YOLO) + pose model + box
detector + 4 garment classifiers all at once is too much for a weak GPU.
This script only loads what one mode actually needs:
    geometric -> no model at all (vision.splitting)
    yolo      -> just the upper/lower box detector (vision.splitting_yolo)
    pose      -> just a YOLOv8-pose model (vision.pose_splitting)

Usage:
    python debug/compare_splitting.py --mode geometric --crops_dir my_crops/
    python debug/compare_splitting.py --mode yolo       --crops_dir my_crops/
    python debug/compare_splitting.py --mode pose       --crops_dir my_crops/

Each run writes:
    debug/output/compare_<mode>/<crop_name>_annotated.jpg   (visual)
    debug/output/compare_<mode>/results.json                (boxes + timing)

After running all the modes you care about, run:
    python debug/compare_splitting.py --summarize

...which reads every debug/output/compare_*/results.json that exists and
prints a side-by-side table -- no models loaded for this step, just JSON.

crops_dir should contain plain person-crop images (.jpg/.png) -- e.g. a
folder of saved snapshots, or crops you pulled out by hand. This script
does NOT run person detection itself; it assumes you already have
person crops, consistent with how build_attribute_profile() receives them
downstream in the real pipeline.
"""

import argparse
import glob
import json
import os
import sys
import time

import cv2

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

OUTPUT_ROOT = "debug/output"


def _draw_boxes(crop, upper_box, lower_box, mode_label):
    """Draws upper box in orange, lower box in green, full width assumed."""
    annotated = crop.copy()
    h, w = annotated.shape[:2]

    if upper_box is not None:
        uy1, uy2 = upper_box
        cv2.rectangle(annotated, (0, uy1), (w, uy2), (0, 165, 255), 2)
        cv2.putText(annotated, "upper", (4, uy1 + 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 165, 255), 1, cv2.LINE_AA)

    if lower_box is not None:
        ly1, ly2 = lower_box
        cv2.rectangle(annotated, (0, ly1), (w, ly2), (0, 200, 0), 2)
        cv2.putText(annotated, "lower", (4, ly1 + 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 200, 0), 1, cv2.LINE_AA)

    cv2.putText(annotated, mode_label, (4, 16),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return annotated


def run_geometric(crops_dir, out_dir):
    from vision.splitting import split_upper_lower

    results = []
    for path in sorted(glob.glob(os.path.join(crops_dir, "*.jpg")) +
                        glob.glob(os.path.join(crops_dir, "*.png"))):
        crop = cv2.imread(path)
        if crop is None:
            continue

        t0 = time.time()
        _, _, upper_box, lower_box = split_upper_lower(crop)
        elapsed_ms = round((time.time() - t0) * 1000, 2)

        name = os.path.splitext(os.path.basename(path))[0]
        annotated = _draw_boxes(crop, upper_box, lower_box, "geometric")
        cv2.imwrite(os.path.join(out_dir, f"{name}_annotated.jpg"), annotated)

        results.append({
            "crop": os.path.basename(path),
            "upper_box": list(upper_box) if upper_box else None,
            "lower_box": list(lower_box) if lower_box else None,
            "used_fallback": False,
            "elapsed_ms": elapsed_ms,
        })

    return results


def run_yolo(crops_dir, out_dir):
    from vision.splitting_yolo import split_upper_lower_yolo

    results = []
    for path in sorted(glob.glob(os.path.join(crops_dir, "*.jpg")) +
                        glob.glob(os.path.join(crops_dir, "*.png"))):
        crop = cv2.imread(path)
        if crop is None:
            continue

        t0 = time.time()
        _, _, upper_box, lower_box = split_upper_lower_yolo(crop)
        elapsed_ms = round((time.time() - t0) * 1000, 2)

        name = os.path.splitext(os.path.basename(path))[0]
        annotated = _draw_boxes(crop, upper_box, lower_box, "yolo-detector")
        cv2.imwrite(os.path.join(out_dir, f"{name}_annotated.jpg"), annotated)

        results.append({
            "crop": os.path.basename(path),
            "upper_box": list(upper_box) if upper_box else None,
            "lower_box": list(lower_box) if lower_box else None,
            "elapsed_ms": elapsed_ms,
        })

    return results


def run_pose(crops_dir, out_dir):
    """
    Pose mode needs keypoints, which in the real pipeline come from
    pipeline.tracker (running on the FULL FRAME, not a person crop) --
    a standalone crop has no full-frame context to run pose detection on
    sensibly. So for a fair, self-contained comparison here, we run
    YOLOv8-pose directly ON the crop itself (treating it as if it were
    the full frame), which is the same thing pose_splitting.py needs:
    keypoints in "frame" coordinates with crop_origin=(0, 0).
    """
    from ultralytics import YOLO
    from vision.pose_splitting import split_upper_lower_pose

    import config
    pose_model = YOLO(config.YOLO_WEIGHTS)

    results = []
    for path in sorted(glob.glob(os.path.join(crops_dir, "*.jpg")) +
                        glob.glob(os.path.join(crops_dir, "*.png"))):
        crop = cv2.imread(path)
        if crop is None:
            continue

        t0 = time.time()
        pose_results = pose_model.predict(crop, verbose=False)[0]

        keypoints_xy, keypoints_conf = None, None
        if pose_results.keypoints is not None and len(pose_results.keypoints.xy) > 0:
            keypoints_xy = pose_results.keypoints.xy[0].cpu().numpy()
            keypoints_conf = (pose_results.keypoints.conf[0].cpu().numpy()
                               if pose_results.keypoints.conf is not None else None)

        split_result = None
        if keypoints_xy is not None:
            split_result = split_upper_lower_pose(crop, keypoints_xy, keypoints_conf, (0, 0))

        elapsed_ms = round((time.time() - t0) * 1000, 2)

        name = os.path.splitext(os.path.basename(path))[0]

        if split_result is not None:
            _, _, upper_box, lower_box = split_result
            used_fallback = False
        else:
            from vision.splitting import split_upper_lower as fixed_ratio_split
            _, _, upper_box, lower_box = fixed_ratio_split(crop)
            used_fallback = True

        annotated = _draw_boxes(crop, upper_box, lower_box,
                                 "pose" + (" (fallback)" if used_fallback else ""))
        cv2.imwrite(os.path.join(out_dir, f"{name}_annotated.jpg"), annotated)

        results.append({
            "crop": os.path.basename(path),
            "upper_box": list(upper_box) if upper_box else None,
            "lower_box": list(lower_box) if lower_box else None,
            "used_fallback": used_fallback,
            "elapsed_ms": elapsed_ms,
        })

    return results


def summarize():
    """
    Reads every debug/output/compare_*/results.json found and prints a
    side-by-side table per crop. Loads NO models -- pure JSON comparison.
    """
    mode_dirs = sorted(glob.glob(os.path.join(OUTPUT_ROOT, "compare_*")))
    if not mode_dirs:
        print(f"No results found under {OUTPUT_ROOT}/compare_*/. Run with --mode first.")
        return

    all_results = {}
    for d in mode_dirs:
        mode_name = os.path.basename(d).replace("compare_", "")
        results_path = os.path.join(d, "results.json")
        if not os.path.exists(results_path):
            continue
        with open(results_path) as f:
            data = json.load(f)
        all_results[mode_name] = {r["crop"]: r for r in data}
        print(f"Loaded {len(data)} results for mode='{mode_name}' from {results_path}")

    if not all_results:
        print("No results.json files found inside compare_*/ folders.")
        return

    modes = sorted(all_results.keys())
    all_crop_names = sorted(set().union(*[set(v.keys()) for v in all_results.values()]))

    print("\n" + "=" * 100)
    print(f"{'crop':<30}", end="")
    for m in modes:
        print(f"{m + ' upper':<18}{m + ' lower':<18}{m + ' ms':<10}", end="")
    print()
    print("-" * 100)

    for crop_name in all_crop_names:
        print(f"{crop_name:<30}", end="")
        for m in modes:
            r = all_results[m].get(crop_name)
            if r is None:
                print(f"{'--':<18}{'--':<18}{'--':<10}", end="")
            else:
                up = str(r.get("upper_box"))
                lo = str(r.get("lower_box"))
                ms = str(r.get("elapsed_ms"))
                print(f"{up:<18}{lo:<18}{ms:<10}", end="")
        print()

    print("\nAverage elapsed_ms per mode:")
    for m in modes:
        times = [r["elapsed_ms"] for r in all_results[m].values() if "elapsed_ms" in r]
        if times:
            print(f"  {m}: {sum(times) / len(times):.2f} ms/crop")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare upper/lower splitting strategies, one mode per run.")
    parser.add_argument("--mode", choices=["geometric", "yolo", "pose"],
                         help="Which splitting strategy to run this time.")
    parser.add_argument("--crops_dir", default="debug/sample_crops",
                         help="Folder of person-crop images to test against (same folder for all modes).")
    parser.add_argument("--summarize", action="store_true",
                         help="Skip running anything -- just read existing results.json files and print a comparison table.")
    args = parser.parse_args()

    if args.summarize:
        summarize()
        sys.exit(0)

    if not args.mode:
        parser.error("Must pass --mode (geometric/yolo/pose) unless using --summarize")

    out_dir = os.path.join(OUTPUT_ROOT, f"compare_{args.mode}")
    os.makedirs(out_dir, exist_ok=True)

    if not os.path.isdir(args.crops_dir):
        print(f"ERROR: crops_dir not found: {args.crops_dir}")
        sys.exit(1)

    print(f"Running mode='{args.mode}' on crops in '{args.crops_dir}'...")

    if args.mode == "geometric":
        results = run_geometric(args.crops_dir, out_dir)
    elif args.mode == "yolo":
        results = run_yolo(args.crops_dir, out_dir)
    elif args.mode == "pose":
        results = run_pose(args.crops_dir, out_dir)

    results_path = os.path.join(out_dir, "results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Done. {len(results)} crops processed.")
    print(f"Annotated images: {out_dir}/*_annotated.jpg")
    print(f"Results JSON: {results_path}")
    print(f"\nRun again with a different --mode on the same --crops_dir, then use --summarize to compare.")
