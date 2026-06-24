# CCTV person re-identification via clothing attributes

A surveillance pipeline that re-identifies people across CCTV footage using
**clothing appearance** (garment type + color) instead of face recognition —
built for conditions where faces are unclear, distant, or occluded.

## How it works

```
video frame
     |
     v
YOLOv8-pose  --> person detection + tracking (ByteTrack) + pose keypoints
     |
     v
C1: long vs. standard garment
     |                    |
  (jellaba route)    (standard route)
     |                    |
C2: jellaba vs.      box detector (YOLOv8n, trained) --> upper/lower crops
    manteau               |
     |              C3: upper body classifier (tshirt / Longmanche / Jacket)
     |              C4: lower body classifier (pontalon / jupe / short)
     |                    |
     +---------- pose-anchored multi-region color extraction ----------+
                           |
                           v
              confidence-weighted track aggregator
                  (locks after 10 clean samples)
                           |
                           v
               JSON record + annotated frame
```

Two things deliberately don't share a model:

- **Garment-type splitting** (where the upper/lower crop boundary goes) is
  controlled by `config.SPLIT_METHOD`, currently set to `yolo_detector` — a
  YOLOv8n model trained specifically on a small hand-labeled upper/lower
  bounding-box dataset. `fixed_ratio` (a simple 55/45 geometric cut) and
  `pose` (YOLOv8-pose keypoints) exist as alternates for comparison.
- **Color sampling** always uses pose keypoints, regardless of
  `SPLIT_METHOD`. Shoulder/hip/knee keypoints define a torso quad (upper),
  hip-knee quad (lower), or shoulder-knee quad (long garments), each split
  into vertical sub-regions and combined via confidence-weighted voting
  (k-means cluster dominance x HSV-based trust). This was a deliberate
  choice to keep "where to look for color" independent of "which method
  drew the garment-type boundary."

## Project layout

| Path | What's in it |
|---|---|
| `main.py` | Live/video entrypoint — the loop described above |
| `config.py` | All tunables: paths, thresholds, which split method is active, class colors |
| `local_config.py` | **Not tracked in git.** Real camera credentials and machine-specific overrides go here — see "Local setup" below |
| `models/` | Model loading (`loaders.py`) and the shared classifier inference helper (`classify.py`) |
| `pipeline/` | Per-frame orchestration: `tracker.py` (detection+tracking+occlusion), `attribute_profile.py` (runs C1-C4 + color, the core per-person logic), `track_aggregator.py` (confidence-weighted voting + locking), `active_zone.py` (ignore border detections), `youtube_source.py` (YouTube live-stream resolution) |
| `vision/` | `splitting.py` (fixed-ratio split), `splitting_yolo.py` (trained box detector, current default), `pose_splitting.py` (pose-based split, alternate), `pose_color_regions.py` (pose quads for color sampling), `color.py` (k-means + HSV color extraction, single-region and multi-region) |
| `ui/draw.py` | All `cv2` drawing — boxes, labels, FPS counter |
| `storage/json_store.py` | Append-only JSON record store (one record per track_id/frame) |
| `debug/` | Standalone test/diagnostic scripts, not part of the live pipeline (see below) |
| `checkpoints/` | Trained model weights (`.pth`/`.pt`) — **gitignored**, not in the repo |

## Classifier status

| Stage | Task | Classes | Status |
|---|---|---|---|
| C1 | long vs. standard garment | `long`, `standard` | trained, integrated |
| C2 | jellaba vs. manteau | 2 | 99.1% accuracy/F1 |
| C3 | upper body | tshirt / Longmanche / Jacket | 96.4% accuracy |
| C4 | lower body | pontalon / jupe / short | 98.6% accuracy |
| Upper/lower box detector | localizes upper/lower regions for splitting | 2 (box classes: upper, lower) | trained YOLOv8n, current default `SPLIT_METHOD` |

Class naming is French/mixed (`pontalon`, `jupe`, `jellaba`, `manteau`,
`Longmanche`, `Jacket`, `tshirt`, `short`) — this is the canonical naming
used throughout the dataset, training notebooks, and code.

## Local setup

1. Clone the repo, create a virtualenv, `pip install -r requirements.txt`
   (`pip install --break-system-packages` if needed depending on your
   environment).
2. Place trained checkpoints in `checkpoints/` — filenames are listed in
   `config.py` (`C1_CKPT_PATH` through `C4_CKPT_PATH`,
   `UPPER_LOWER_DETECTOR_PATH`). These are gitignored and must be obtained
   separately (Drive/Colab training outputs).
3. Create `local_config.py` in the project root for any real camera
   credentials — this file is gitignored and never committed:
   ```python
   VIDEO_SOURCE_OVERRIDE = "URL GOES HERE" 
   ```
   `config.py` automatically picks this up if the file exists, and falls
   back to the placeholder `VIDEO_SOURCE` otherwise.
4. Run `python main.py` for the live pipeline, or see `debug/` for
   lighter-weight testing options below.

## Debug / test scripts

These are standalone, not invoked by `main.py`:

- `debug/test_on_images.py` — runs the full pipeline on a folder of still
  images instead of live video. Much cheaper for checking correctness
  before testing on real footage. `python debug/test_on_images.py --images_dir path/to/images`
- `debug/compare_splitting.py` — runs one splitting method at a time
  (geometric / yolo / pose) against a folder of person crops and saves
  annotated results + timing, so methods can be compared without loading
  all three models into memory simultaneously.
- `debug/diagnose_color.py` — visualizes exactly what the single-region
  color pipeline (`vision.color.extract_color_for_region`) is doing on one
  crop: sample box placement, skin mask, edge mask, final k-means pixels.

## Known limitations / open items

- The upper/lower box detector was trained on a small dataset (~530 images);
  accuracy on real CCTV footage at scale hasn't been formally evaluated yet.
- Multi-region color voting currently uses 3 vertical sub-regions per quad
  (`config.COLOR_NUM_SUBREGIONS`) — not yet validated against ground truth.
- `SPLIT_METHOD` and pose-based color sampling both depend on
  `yolov8n-pose.pt` keypoints; CCTV footage with heavy occlusion or extreme
  angles can produce unusable keypoints, in which case color extraction
  silently returns `None` for that sample rather than falling back to a
  fixed-box method.
