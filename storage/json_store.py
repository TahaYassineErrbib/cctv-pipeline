"""
Simple JSON-file storage for attribute profile records.

One record per (track_id, frame) — full history, not overwritten. This is
intentional: for re-ID groundwork later, we want to see how a tracked
person's attributes vary across frames/cameras, not just their latest state.

Each record:
{
  "track_id": int,
  "frame_idx": int,
  "timestamp": float,        # seconds since pipeline start
  "camera_id": str,
  "bbox": [x1, y1, x2, y2],
  "profile": {...}            # output of build_attribute_profile()
}
"""

import json
import os
import threading

import config


_lock = threading.Lock()


def _load_existing():
    if not os.path.exists(config.PROFILES_JSON_PATH):
        return []
    with open(config.PROFILES_JSON_PATH, "r") as f:
        try:
            return json.load(f)
        except json.JSONDecodeError:
            print(f"[storage] WARNING: {config.PROFILES_JSON_PATH} was corrupt/empty, starting fresh.")
            return []


def append_record(record):
    """
    Appends one record to the JSON store. Uses a simple read-modify-write
    with a lock — fine for single-process pipelines like this one. If this
    ever needs to handle concurrent writers, switch to one-file-per-record
    or a real database instead.
    """
    with _lock:
        records = _load_existing()
        records.append(record)
        with open(config.PROFILES_JSON_PATH, "w") as f:
            json.dump(records, f, indent=2)


def load_all_records():
    """Returns the full list of stored records."""
    with _lock:
        return _load_existing()


def records_for_track(track_id):
    """Convenience filter: all records for a given track_id, in order."""
    return [r for r in load_all_records() if r["track_id"] == track_id]
