"""
Aggregates classification results per track_id across multiple sampled
frames, using confidence-weighted majority voting, then permanently locks
the profile once enough samples have been collected.

Why this matters: a single sampled frame can be wrong (bad lighting, odd
pose, partial occlusion). Voting across N samples before committing to a
final identity profile gives much more stable attributes for re-ID, and
once locked, the expensive classifier calls can be skipped entirely for
that track — a meaningful speedup on a weak GPU since most of a video's
runtime is spent on tracks that have already stabilized.
"""

from collections import defaultdict

import config


class TrackAggregator:
    def __init__(self, samples_required=10):
        self.samples_required = samples_required

        # track_id -> list of raw per-sample profiles (pre-lock)
        self._buffers = defaultdict(list)

        # track_id -> locked, final profile dict (post-lock)
        self._locked_profiles = {}

    def is_locked(self, track_id):
        return track_id in self._locked_profiles

    def get_locked_profile(self, track_id):
        return self._locked_profiles.get(track_id)

    def add_sample(self, track_id, profile):
        """
        Adds one sample for a track. If this sample brings the track to
        samples_required, the profile is computed and locked permanently.
        Returns the locked profile if locking just happened, else None.
        """
        if self.is_locked(track_id):
            # already locked — ignore further samples entirely
            return None

        self._buffers[track_id].append(profile)

        if len(self._buffers[track_id]) >= self.samples_required:
            locked = self._compute_locked_profile(self._buffers[track_id])
            self._locked_profiles[track_id] = locked
            del self._buffers[track_id]  # free the buffer, no longer needed
            return locked

        return None

    def samples_collected(self, track_id):
        if self.is_locked(track_id):
            return self.samples_required
        return len(self._buffers.get(track_id, []))

    # -----------------------------------------------------------------
    # Voting logic
    # -----------------------------------------------------------------

    def _compute_locked_profile(self, samples):
        """
        Confidence-weighted majority vote across all samples for one track.
        Handles both branches (long garment vs standard upper/lower) —
        votes are computed per-branch, then the branch itself is decided
        by majority vote on garment_type first.
        """
        garment_type_votes = defaultdict(float)
        for s in samples:
            garment_type_votes[s["garment_type"]] += s["garment_type_conf"]

        final_garment_type = max(garment_type_votes, key=garment_type_votes.get)
        is_long = final_garment_type is not None and "long" in final_garment_type.lower()

        locked = {
            "garment_type": final_garment_type,
            "garment_type_conf": round(
                garment_type_votes[final_garment_type] / len(samples), 3
            ),
            "num_samples_used": len(samples),
        }

        if is_long:
            locked["long_type"] = self._vote_attribute(samples, "long_type")
            # carry over the color sample box from the most recent sample
            # that has it, purely for drawing — voting doesn't apply to geometry
            for s in reversed(samples):
                if "color_sample_box" in s:
                    locked["color_sample_box"] = s["color_sample_box"]
                    break
        else:
            locked["upper"] = self._vote_attribute(samples, "upper")
            locked["lower"] = self._vote_attribute(samples, "lower")
            # carry over box coords from the most recent sample that has them,
            # purely for drawing purposes — voting doesn't apply to geometry
            for s in reversed(samples):
                if "upper_box" in s:
                    locked["upper_box"] = s["upper_box"]
                    locked["lower_box"] = s["lower_box"]
                    locked["upper_color_box"] = s.get("upper_color_box")
                    locked["lower_color_box"] = s.get("lower_color_box")
                    break

        return locked

    @staticmethod
    def _vote_attribute(samples, key):
        """
        Confidence-weighted vote for one sub-attribute (e.g. "upper" or
        "long_type") across samples that actually contain it. Some samples
        may be missing this key if a track briefly flipped branches due to
        a noisy C1 call before settling — those samples are just skipped
        for this attribute's vote.
        """
        class_votes = defaultdict(float)
        color_votes = defaultdict(float)
        confs = []

        for s in samples:
            attr = s.get(key)
            if attr is None:
                continue

            cls = attr.get("class")
            conf = attr.get("confidence", 0.0)
            if cls is not None:
                class_votes[cls] += conf
                confs.append(conf)

            color = attr.get("color")
            if color is not None and color.get("name") is not None:
                # weight color votes by the same classifier confidence —
                # we don't have a separate "color confidence" signal
                color_votes[color["name"]] += conf

        if not class_votes:
            return {"class": None, "confidence": 0.0, "color": None}

        final_class = max(class_votes, key=class_votes.get)
        final_color_name = max(color_votes, key=color_votes.get) if color_votes else None

        return {
            "class": final_class,
            "confidence": round(sum(confs) / len(confs), 3) if confs else 0.0,
            "color": {"name": final_color_name} if final_color_name else None,
        }