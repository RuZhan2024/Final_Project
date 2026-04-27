"""Unified data-preparation pipeline for extraction, labeling, splitting, and windowing.

This module provides a contract-first replacement for the legacy multi-step
Makefile data workflow. It supports:
1) pose extraction from videos or normalization from raw `.npz` pose dumps,
2) label template generation and label-map loading,
3) deterministic split generation,
4) model-ready sliding window export into canonical processed storage.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import json
import math
import os
from pathlib import Path
import random
import re
from typing import Any, Iterable, Mapping, Sequence
import warnings

import numpy as np
import numpy.typing as npt

from ..preprocessing.pose_resample import resample_pose_window
from ..preprocessing import FeatCfg, build_canonical_input
from .contracts import DataContractError
from .resolver import DataPathResolver

FloatArray = npt.NDArray[np.float32]
IntArray = npt.NDArray[np.int64]

VIDEO_EXTENSIONS: tuple[str, ...] = (".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v")


@dataclass(frozen=True)
class PoseSequence:
    """One normalized pose sequence used by the preparation pipeline.

    Attributes:
        sequence_id: Stable sequence identifier.
        t_ms: Timestamp vector in milliseconds with shape `[N]`.
        xy: Pose tensor with shape `[N, J, 2]`.
        conf: Confidence tensor with shape `[N, J]`.
        fps: Estimated source frame rate.
        source_path: Source raw artifact path.

    Complexity:
        O(1) immutable metadata wrapper over arrays.
    """

    sequence_id: str
    t_ms: FloatArray
    xy: FloatArray
    conf: FloatArray
    fps: float
    source_path: Path


@dataclass(frozen=True)
class PipelinePaths:
    """Canonical output paths for one dataset preparation run.

    Complexity:
        O(1) immutable path container.
    """

    dataset: str
    raw_dir: Path
    interim_pose_dir: Path
    interim_split_path: Path
    interim_label_template_path: Path
    processed_dir: Path


@dataclass(frozen=True)
class DiscoveredSequenceRecord:
    """One raw sequence discovered with immediate annotation linkage."""

    sequence_id: str
    source_path: Path
    fall_spans_ms: tuple[tuple[float, float], ...]
    label_missing: bool = False


def build_pipeline_paths(resolver: DataPathResolver, dataset: str) -> PipelinePaths:
    """Build canonical data-preparation paths for one dataset key.

    Args:
        resolver: Canonical data path resolver.
        dataset: Dataset key.

    Returns:
        Immutable path bundle.

    Complexity:
        O(1).
    """

    ds = resolver.dataset(dataset)
    raw_dir = Path(ds["raw_dir"]).resolve()
    processed_dir = Path(ds["processed_dir"]).resolve()
    interim_root = resolver.layout.interim / str(dataset).strip().lower()
    return PipelinePaths(
        dataset=str(dataset).strip(),
        raw_dir=raw_dir,
        interim_pose_dir=interim_root / "pose",
        interim_split_path=interim_root / "splits.json",
        interim_label_template_path=interim_root / "labels_template.json",
        processed_dir=processed_dir,
    )


def discover_raw_sources(raw_dir: Path) -> tuple[Path, ...]:
    """Discover raw source files under one dataset raw directory.

    Supports `.npz` pose artifacts and common video file formats.

    Args:
        raw_dir: Dataset raw directory.

    Returns:
        Deterministically sorted source paths.

    Complexity:
        O(n log n) in discovered file count.
    """

    if not raw_dir.exists():
        return ()
    out: list[Path] = []
    for path in raw_dir.rglob("*"):
        if not path.is_file():
            continue
        suffix = path.suffix.lower()
        if suffix == ".npz" or suffix in VIDEO_EXTENSIONS:
            out.append(path)
    return tuple(sorted(out, key=lambda p: str(p)))


def discover_sequences_with_labels(
    *,
    raw_dir: Path,
    strict_missing_labels: bool = False,
) -> tuple[DiscoveredSequenceRecord, ...]:
    """Discover videos and resolve labels at discovery time via dataset adapters."""

    root = Path(raw_dir).expanduser().resolve()
    dataset_key = _normalize_token(root.name)
    if dataset_key == "le2i":
        return _discover_le2i_sequences_with_labels(root, strict_missing_labels=strict_missing_labels)
    if dataset_key == "caucafall":
        return _discover_caucafall_sequences_with_labels(root, strict_missing_labels=strict_missing_labels)

    out: list[DiscoveredSequenceRecord] = []
    for src in discover_raw_sources(root):
        if src.suffix.lower() not in VIDEO_EXTENSIONS:
            continue
        out.append(
            DiscoveredSequenceRecord(
                sequence_id=_sequence_id_from_path(src),
                source_path=src.resolve(),
                fall_spans_ms=(),
                label_missing=True,
            )
        )
        _warn_or_raise_label_issue(
            f"Label missing | dataset={root.name} source={src.resolve()} annotation=unavailable",
            fail_fast=strict_missing_labels,
        )
    return tuple(out)


def rebuild_labels_and_splits_from_raw(
    *,
    raw_dir: Path,
    label_template_path: Path,
    split_path: Path,
    seed: int = 1337,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    strict_missing_labels: bool = False,
) -> dict[str, int]:
    """Rebuild labels template and splits directly from raw data adapters."""

    records = discover_sequences_with_labels(
        raw_dir=raw_dir,
        strict_missing_labels=bool(strict_missing_labels),
    )
    if not records:
        raise DataContractError(f"No discoverable video sources under {raw_dir}")

    label_payload: dict[str, dict[str, list[list[float]]]] = {}
    sequence_ids: list[str] = []
    positive_ids: list[str] = []
    unlabeled_ids: list[str] = []
    missing_count = 0
    for rec in sorted(records, key=lambda r: r.sequence_id):
        sequence_ids.append(rec.sequence_id)
        spans = [[float(a), float(b)] for a, b in rec.fall_spans_ms]
        label_payload[rec.sequence_id] = {"fall_spans_ms": spans}
        if spans:
            positive_ids.append(rec.sequence_id)
        if rec.label_missing:
            missing_count += 1
            unlabeled_ids.append(rec.sequence_id)

    label_template_path.parent.mkdir(parents=True, exist_ok=True)
    label_template_path.write_text(json.dumps(label_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    build_splits(
        sequence_ids=tuple(sequence_ids),
        positive_sequence_ids=tuple(positive_ids),
        unlabeled_sequence_ids=tuple(unlabeled_ids),
        train_ratio=float(train_ratio),
        val_ratio=float(val_ratio),
        test_ratio=float(test_ratio),
        seed=int(seed),
        out_path=split_path,
        overwrite=True,
    )
    return {
        "sequence_count": int(len(sequence_ids)),
        "positive_count": int(len(positive_ids)),
        "label_missing_count": int(missing_count),
        "unlabeled_count": int(len(unlabeled_ids)),
    }


def extract_pose_sequences(
    *,
    sources: Sequence[Path],
    out_dir: Path,
    model_complexity: int = 1,
    min_detection_confidence: float = 0.5,
    min_tracking_confidence: float = 0.5,
    overwrite: bool = False,
    strict: bool = False,
    extract_videos: bool = False,
    max_sources: int | None = None,
) -> tuple[Path, ...]:
    """Extract/normalize pose sequences into canonical interim `.npz` artifacts.

    Args:
        sources: Raw source paths (video or pose `.npz`).
        out_dir: Target interim pose directory.
        model_complexity: MediaPipe Pose model complexity.
        min_detection_confidence: MediaPipe detection confidence threshold.
        min_tracking_confidence: MediaPipe tracking confidence threshold.
        overwrite: Whether to overwrite existing sequence files.
        strict: Fail-fast on first extraction error when true.
        extract_videos: Enable MediaPipe extraction from video files.
        max_sources: Optional cap on number of sources to process.

    Returns:
        Tuple of generated/interim pose sequence paths.

    Complexity:
        O(V * F * J) for videos; O(N * J) for npz normalization.
    """

    out_dir.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []
    errors: list[str] = []
    src_iter: Sequence[Path]
    if max_sources is not None and int(max_sources) > 0:
        src_iter = tuple(sources[: int(max_sources)])
    else:
        src_iter = tuple(sources)

    for src in src_iter:
        sequence_id = _sequence_id_from_path(src)
        out_path = out_dir / f"{sequence_id}.npz"
        if out_path.exists() and not overwrite:
            outputs.append(out_path)
            continue
        if src.suffix.lower() in VIDEO_EXTENSIONS and not extract_videos:
            continue
        try:
            if src.suffix.lower() == ".npz":
                seq = _load_pose_sequence_npz(src)
            else:
                seq = _extract_pose_from_video(
                    src,
                    model_complexity=model_complexity,
                    min_detection_confidence=min_detection_confidence,
                    min_tracking_confidence=min_tracking_confidence,
                )
            _save_pose_sequence_npz(out_path, seq)
            outputs.append(out_path)
        except Exception as exc:
            msg = f"{src}: {exc}"
            errors.append(msg)
            if strict:
                raise DataContractError(f"Pose extraction failed: {msg}") from exc
            continue
    if errors and strict:
        raise DataContractError(f"Pose extraction failed for {len(errors)} sources.")
    return tuple(outputs)


def create_label_template(
    *,
    pose_sequence_paths: Sequence[Path],
    out_path: Path,
    overwrite: bool = False,
) -> Path:
    """Create an editable label template JSON for discovered sequences.

    Template schema:
        `{sequence_id: {"fall_spans_ms": [[start_ms, end_ms], ...]}}`

    Args:
        pose_sequence_paths: Sequence `.npz` paths from interim extraction.
        out_path: Label template output path.
        overwrite: Whether to overwrite existing file.

    Returns:
        Template path.

    Complexity:
        O(n) in sequence count.
    """

    if out_path.exists() and not overwrite:
        return out_path
    payload: dict[str, dict[str, list[list[float]]]] = {}
    for path in pose_sequence_paths:
        payload[path.stem] = {"fall_spans_ms": []}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return out_path


def load_label_map(label_path: Path | None) -> Mapping[str, tuple[tuple[float, float], ...]]:
    """Load label-map file into normalized span tuples.

    Args:
        label_path: Label JSON path or `None`.

    Returns:
        Mapping from sequence ID to ordered fall-span tuples.

    Complexity:
        O(n) in total span count.
    """

    if label_path is None:
        return {}
    if not label_path.exists() or not label_path.is_file():
        raise DataContractError(f"Label file not found: {label_path}")
    payload = json.loads(label_path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise DataContractError("Label file must decode to a mapping.")
    out: dict[str, tuple[tuple[float, float], ...]] = {}
    for seq_id, value in payload.items():
        key = str(seq_id).strip()
        if not key:
            continue
        spans = _parse_label_value(value)
        out[key] = tuple(sorted(spans, key=lambda it: (it[0], it[1])))
    return out


def build_splits(
    *,
    sequence_ids: Sequence[str],
    positive_sequence_ids: Sequence[str] | None,
    unlabeled_sequence_ids: Sequence[str] | None = None,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
    out_path: Path,
    overwrite: bool = False,
) -> Mapping[str, tuple[str, ...]]:
    """Generate deterministic train/val/test splits and persist JSON artifact.

    Args:
        sequence_ids: Sequence IDs to split.
        train_ratio: Train fraction.
        val_ratio: Validation fraction.
        test_ratio: Test fraction.
        positive_sequence_ids: Optional sequence IDs with at least one fall span.
        seed: Deterministic random seed.
        out_path: Split JSON output path.
        overwrite: Whether to overwrite existing split file.

    Returns:
        Mapping containing `train`, `val`, and `test` sequence IDs.

    Complexity:
        O(n log n) due to sorting/shuffling.
    """

    if out_path.exists() and not overwrite:
        payload = json.loads(out_path.read_text(encoding="utf-8"))
        if not isinstance(payload, Mapping):
            raise DataContractError(f"Existing split file is not a mapping: {out_path}")
        return {
            "train": tuple(str(x) for x in list(payload.get("train", []))),
            "val": tuple(str(x) for x in list(payload.get("val", []))),
            "test": tuple(str(x) for x in list(payload.get("test", []))),
            "unlabeled": tuple(str(x) for x in list(payload.get("unlabeled", []))),
        }

    unlabeled_ids = {str(x).strip() for x in (unlabeled_sequence_ids or ()) if str(x).strip()}
    ids = sorted({str(x).strip() for x in sequence_ids if str(x).strip() and str(x).strip() not in unlabeled_ids})
    if not ids:
        raise DataContractError("Cannot build splits from empty sequence ID set.")
    _validate_ratio_sum(train_ratio, val_ratio, test_ratio)

    pos_ids = {str(x).strip() for x in (positive_sequence_ids or ()) if str(x).strip()}
    pos = sorted([sid for sid in ids if sid in pos_ids])
    neg = sorted([sid for sid in ids if sid not in pos_ids])

    if pos and neg:
        train_list, val_list, test_list = _stratified_split_ids(
            pos=pos,
            neg=neg,
            train_ratio=float(train_ratio),
            val_ratio=float(val_ratio),
            test_ratio=float(test_ratio),
            seed=int(seed),
        )
    else:
        rng = random.Random(int(seed))
        rng.shuffle(ids)
        n = len(ids)
        n_train = int(math.floor(n * train_ratio))
        n_val = int(math.floor(n * val_ratio))
        n_test = max(0, n - n_train - n_val)
        train_list = list(ids[:n_train])
        val_list = list(ids[n_train : n_train + n_val])
        test_list = list(ids[n_train + n_val : n_train + n_val + n_test])

    _ensure_eval_splits_have_positive(train_list, val_list, test_list, pos_ids)
    train = tuple(train_list)
    val = tuple(val_list)
    test = tuple(test_list)
    unlabeled = tuple(sorted(unlabeled_ids))

    payload = {"train": list(train), "val": list(val), "test": list(test), "unlabeled": list(unlabeled)}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {"train": train, "val": val, "test": test, "unlabeled": unlabeled}


def load_splits(split_path: Path) -> Mapping[str, tuple[str, ...]]:
    """Load split JSON mapping.

    Args:
        split_path: Split file path.

    Returns:
        Mapping with split IDs.

    Complexity:
        O(n) in total sequence IDs.
    """

    if not split_path.exists() or not split_path.is_file():
        raise DataContractError(f"Split file not found: {split_path}")
    payload = json.loads(split_path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise DataContractError("Split file must decode to a mapping.")
    return {
        "train": tuple(str(x) for x in list(payload.get("train", []))),
        "val": tuple(str(x) for x in list(payload.get("val", []))),
        "test": tuple(str(x) for x in list(payload.get("test", []))),
        "unlabeled": tuple(str(x) for x in list(payload.get("unlabeled", []))),
    }


def export_windows(
    *,
    pose_sequence_paths: Sequence[Path],
    splits: Mapping[str, tuple[str, ...]],
    labels: Mapping[str, tuple[tuple[float, float], ...]],
    out_dir: Path,
    target_fps: float,
    window_frames: int,
    stride_frames: int,
    conf_gate: float,
    include_unlabeled: bool = True,
    overwrite: bool = False,
    max_windows: int | None = None,
) -> tuple[Path, ...]:
    """Export canonical window `.npz` artifacts for model training/evaluation.

    Args:
        pose_sequence_paths: Interim pose sequence file paths.
        splits: Split assignment mapping.
        labels: Fall-span map in milliseconds.
        out_dir: Processed output directory.
        target_fps: Target resampling FPS.
        window_frames: Target output window length.
        stride_frames: Sliding stride in target-frame units.
        conf_gate: Confidence threshold for validity mask.
        include_unlabeled: Keep windows with unknown labels as `y=-1`.
        overwrite: Overwrite existing windows.
        max_windows: Optional global cap on exported windows.

    Returns:
        Tuple of generated window paths.

    Complexity:
        O(S * W * J), where S is sequence count and W windows per sequence.
    """

    if target_fps <= 0.0:
        raise DataContractError("target_fps must be positive.")
    if window_frames < 2:
        raise DataContractError("window_frames must be >= 2.")
    if stride_frames < 1:
        raise DataContractError("stride_frames must be >= 1.")

    effective_labels = _materialize_effective_labels(
        pose_sequence_paths=pose_sequence_paths,
        labels=labels,
    )
    by_id = {path.stem: path for path in pose_sequence_paths}
    out_dir.mkdir(parents=True, exist_ok=True)
    out_paths: list[Path] = []

    duration_ms = (float(window_frames) / float(target_fps)) * 1000.0
    stride_ms = (float(stride_frames) / float(target_fps)) * 1000.0

    cap = None if max_windows is None or int(max_windows) <= 0 else int(max_windows)
    for split_name in ("train", "val", "test"):
        split_ids = splits.get(split_name, ())
        for sequence_id in split_ids:
            if cap is not None and len(out_paths) >= cap:
                return tuple(out_paths)
            seq_path = by_id.get(sequence_id)
            if seq_path is None:
                continue
            seq = _load_pose_sequence_npz(seq_path)
            seq_labels = effective_labels.get(sequence_id)
            windows = _sequence_window_times(seq.t_ms, duration_ms=duration_ms, stride_ms=stride_ms)
            for idx, (start_ms, end_ms) in enumerate(windows):
                if cap is not None and len(out_paths) >= cap:
                    return tuple(out_paths)
                y = _label_for_window(start_ms=start_ms, end_ms=end_ms, spans_ms=seq_labels)
                if y < 0 and not include_unlabeled:
                    continue
                out_path = out_dir / f"{split_name}_{sequence_id}_w{idx:05d}.npz"
                if out_path.exists() and not overwrite:
                    out_paths.append(out_path)
                    continue
                sidx, eidx = _slice_indices_for_interval(seq.t_ms, start_ms=start_ms, end_ms=end_ms)
                if eidx <= sidx:
                    continue
                xy_out, conf_out, _s_ms, _e_ms, _fps_est = resample_pose_window(
                    raw_t_ms=seq.t_ms[sidx:eidx].tolist(),
                    raw_xy=seq.xy[sidx:eidx].tolist(),
                    raw_conf=seq.conf[sidx:eidx].tolist(),
                    target_fps=float(target_fps),
                    target_T=int(window_frames),
                    prevalidated_raw=False,
                )
                joints = np.asarray(xy_out, dtype=np.float32)
                conf = np.asarray(conf_out, dtype=np.float32)
                if joints.ndim != 3 or joints.shape[0] != int(window_frames):
                    continue
                if conf.ndim != 2 or conf.shape[:2] != joints.shape[:2]:
                    conf = np.ones((joints.shape[0], joints.shape[1]), dtype=np.float32)
                mask = np.asarray(conf >= float(conf_gate), dtype=np.bool_)
                if idx == 0:
                    # Canonical parity check: ensure pelvis-centering path is exercised in pipeline.
                    pelvis_norm = _canonical_pelvis_center_norm(
                        joints_xy=joints,
                        conf=conf,
                        mask=mask,
                        fps=float(target_fps),
                        conf_gate=float(conf_gate),
                    )
                    if pelvis_norm > 0.10:
                        _warn_or_raise_label_issue(
                            f"CANONICAL QC WARNING: high pelvis-center residual norm={pelvis_norm:.4f} "
                            f"sequence={sequence_id} split={split_name}"
                        )
                np.savez_compressed(
                    out_path,
                    joints=joints,
                    conf=conf,
                    mask=mask,
                    fps=np.asarray([float(target_fps)], dtype=np.float32),
                    y=np.asarray([int(y)], dtype=np.int64),
                    video_id=np.asarray([sequence_id]),
                    # Downstream alert/eval code consumes these as inclusive frame
                    # indices, not milliseconds, so keep them on the target-FPS grid.
                    w_start=np.asarray(
                        [int(round((float(start_ms) * float(target_fps)) / 1000.0))],
                        dtype=np.int64,
                    ),
                    w_end=np.asarray(
                        [int(round((float(start_ms) * float(target_fps)) / 1000.0)) + int(window_frames) - 1],
                        dtype=np.int64,
                    ),
                    source_path=np.asarray([str(seq.source_path)]),
                )
                out_paths.append(out_path)
    return tuple(out_paths)


def _canonical_pelvis_center_norm(
    *,
    joints_xy: FloatArray,
    conf: FloatArray | None,
    mask: BoolArray | None,
    fps: float,
    conf_gate: float,
) -> float:
    """Return mean pelvis residual norm after canonical pelvis-centering.

    This is a cheap contract check for the exported windows: deploy/eval code
    assumes pelvis-relative canonical inputs before any stream-specific reshape.
    """

    feat_cfg = FeatCfg(
        center="pelvis",
        use_motion=True,
        use_bone=False,
        use_bone_length=False,
        use_conf_channel=True,
        motion_scale_by_fps=True,
        conf_gate=float(conf_gate),
        use_precomputed_mask=True,
    )
    x_can, _m = build_canonical_input(
        joints_xy=np.asarray(joints_xy, dtype=np.float32),
        motion_xy=None,
        conf=(None if conf is None else np.asarray(conf, dtype=np.float32)),
        mask=(None if mask is None else np.asarray(mask, dtype=np.bool_)),
        fps=float(fps),
        feat_cfg=feat_cfg,
    )
    if x_can.shape[1] > 24:
        pelvis = 0.5 * (x_can[:, 23, :2] + x_can[:, 24, :2])
    else:
        pelvis = x_can[:, :, :2].mean(axis=1)
    return float(np.linalg.norm(pelvis, axis=-1).mean())


def _sequence_id_from_path(path: Path) -> str:
    p = Path(path).expanduser()
    parts = list(p.parts)
    raw_idx = -1
    for idx, part in enumerate(parts):
        if _normalize_token(part) == "raw":
            raw_idx = idx
            break

    if raw_idx >= 0 and raw_idx + 1 < len(parts):
        dataset_prefix = _sanitize_id_part(parts[raw_idx + 1]) or "dataset"
        rel_parts = parts[raw_idx + 2 : -1]
    else:
        dataset_prefix = _sanitize_id_part(p.parent.name) or "dataset"
        rel_parts = parts[:-1]

    seq_parts: list[str] = []
    for part in rel_parts:
        token = _sanitize_id_part(part)
        if not token:
            continue
        if token in {"videos", "annotation_files", "annotations"}:
            continue
        seq_parts.append(token)

    stem_token = _sanitize_id_part(p.stem)
    if stem_token:
        seq_parts.append(stem_token)

    while seq_parts and seq_parts[0] == dataset_prefix:
        seq_parts.pop(0)
    if not seq_parts:
        seq_parts = [stem_token or "sequence"]
    return "_".join([dataset_prefix, *seq_parts])


def _sanitize_id_part(text: str) -> str:
    token = re.sub(r"[^a-z0-9]+", "_", str(text).strip().lower()).strip("_")
    return re.sub(r"_+", "_", token)


def _extract_pose_from_video(
    path: Path,
    *,
    model_complexity: int,
    min_detection_confidence: float,
    min_tracking_confidence: float,
) -> PoseSequence:
    """Extract one pose sequence from a video using MediaPipe Pose.

    Args:
        path: Video source path.
        model_complexity: MediaPipe model complexity.
        min_detection_confidence: Detection threshold.
        min_tracking_confidence: Tracking threshold.

    Returns:
        Normalized pose sequence.

    Raises:
        DataContractError: If dependencies are missing or extraction fails.

    Complexity:
        O(F * J), where F is frame count and J is joint count.
    """

    try:
        import cv2
    except Exception as exc:  # pragma: no cover - environment-dependent import
        raise DataContractError("Video extraction requires opencv-contrib-python.") from exc

    disable_mp = str(os.getenv("FD_DISABLE_MEDIAPIPE", "0")).strip().lower() in {"1", "true", "yes", "on"}
    if disable_mp:
        return _extract_pose_from_video_fallback_cv(path)

    try:
        os.environ.setdefault("MEDIAPIPE_DISABLE_GPU", "1")
        import mediapipe as mp  # type: ignore[import-not-found]
    except Exception:
        return _extract_pose_from_video_fallback_cv(path)

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise DataContractError(f"Failed to open video source: {path}")

    source_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    if source_fps <= 0.0:
        source_fps = 30.0

    sequence_id = _sequence_id_from_path(path)
    t_ms: list[float] = []
    xy_list: list[list[list[float]]] = []
    conf_list: list[list[float]] = []

    try:
        pose = mp.solutions.pose.Pose(
            model_complexity=int(model_complexity),
            static_image_mode=False,
            min_detection_confidence=float(min_detection_confidence),
            min_tracking_confidence=float(min_tracking_confidence),
        )
    except Exception:
        cap.release()
        return _extract_pose_from_video_fallback_cv(path)
    frame_idx = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = pose.process(rgb)
            landmarks = getattr(result, "pose_landmarks", None)
            if landmarks is None or not getattr(landmarks, "landmark", None):
                frame_idx += 1
                continue
            pts = landmarks.landmark
            xy = [[float(p.x), float(p.y)] for p in pts]
            cf = [float(getattr(p, "visibility", 1.0)) for p in pts]
            if len(xy) < 1:
                frame_idx += 1
                continue
            t_ms.append((float(frame_idx) / source_fps) * 1000.0)
            xy_list.append(xy)
            conf_list.append(cf)
            frame_idx += 1
    finally:
        cap.release()
        pose.close()

    if not xy_list:
        return _extract_pose_from_video_fallback_cv(path)
    xy_np = np.asarray(xy_list, dtype=np.float32)
    conf_np = np.asarray(conf_list, dtype=np.float32)
    t_np = np.asarray(t_ms, dtype=np.float32)
    return PoseSequence(
        sequence_id=sequence_id,
        t_ms=t_np,
        xy=xy_np,
        conf=conf_np,
        fps=float(source_fps),
        source_path=path.resolve(),
    )


def _extract_pose_from_video_fallback_cv(path: Path) -> PoseSequence:
    """OpenCV-only fallback extractor when MediaPipe runtime is unavailable.

    The fallback builds deterministic pseudo-skeletons using per-frame intensity
    moments to estimate a body center and stamps a fixed 33-joint template
    around it.

    Complexity:
        O(F * J) in frames and joints.
    """

    import cv2

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise DataContractError(f"Failed to open video source: {path}")

    source_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    if source_fps <= 0.0:
        source_fps = 30.0

    sequence_id = _sequence_id_from_path(path)
    template = _fallback_joint_template()
    n_joints = int(template.shape[0])

    t_ms: list[float] = []
    xy_list: list[np.ndarray] = []
    conf_list: list[np.ndarray] = []
    frame_idx = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            h, w = frame.shape[:2]
            if h < 2 or w < 2:
                frame_idx += 1
                continue
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            m = cv2.moments(gray)
            if float(m.get("m00", 0.0)) > 1e-6:
                cx_px = float(m["m10"] / m["m00"])
                cy_px = float(m["m01"] / m["m00"])
            else:
                cx_px = float(w) * 0.5
                cy_px = float(h) * 0.5
            cx = np.float32(np.clip(cx_px / float(max(1, w - 1)), 0.0, 1.0))
            cy = np.float32(np.clip(cy_px / float(max(1, h - 1)), 0.0, 1.0))
            shift = np.asarray([cx, cy], dtype=np.float32)
            jitter = np.float32((frame_idx % 5) * 0.002)
            joints = np.clip(template + shift + jitter, 0.0, 1.0).astype(np.float32, copy=False)
            conf = np.full((n_joints,), 0.6, dtype=np.float32)

            t_ms.append((float(frame_idx) / source_fps) * 1000.0)
            xy_list.append(joints)
            conf_list.append(conf)
            frame_idx += 1
    finally:
        cap.release()

    if not xy_list:
        raise DataContractError(f"No frames decoded in fallback extractor: {path}")

    return PoseSequence(
        sequence_id=sequence_id,
        t_ms=np.asarray(t_ms, dtype=np.float32),
        xy=np.asarray(xy_list, dtype=np.float32),
        conf=np.asarray(conf_list, dtype=np.float32),
        fps=float(source_fps),
        source_path=path.resolve(),
    )


def _fallback_joint_template() -> FloatArray:
    angles = np.linspace(0.0, np.pi * 2.0, 33, endpoint=False, dtype=np.float32)
    radius_x = np.float32(0.08)
    radius_y = np.float32(0.12)
    x = np.cos(angles, dtype=np.float32) * radius_x
    y = np.sin(angles, dtype=np.float32) * radius_y
    return np.stack([x, y], axis=1).astype(np.float32, copy=False)


def _load_pose_sequence_npz(path: Path) -> PoseSequence:
    """Load one pose sequence from `.npz` with robust key compatibility.

    Args:
        path: Sequence artifact path.

    Returns:
        Normalized pose sequence.

    Raises:
        DataContractError: If required keys/shapes are invalid.

    Complexity:
        O(N * J) in decoded array size.
    """

    with np.load(path, allow_pickle=False) as z:
        t_ms = _first_available_1d_float(
            z,
            keys=("t_ms", "raw_t_ms", "timestamps_ms", "time_ms"),
        )
        if t_ms is None:
            raise DataContractError(f"Missing timestamp vector in {path}")

        xy = _first_available_xy(z)
        if xy is None:
            raise DataContractError(f"Missing XY pose tensor in {path}")

        conf = _first_available_conf(z, n_frames=xy.shape[0], n_joints=xy.shape[1])
        if conf is None:
            conf = np.ones((xy.shape[0], xy.shape[1]), dtype=np.float32)

        fps = _first_available_scalar_float(z, ("fps", "source_fps"), default=30.0)
        seq_id = str(_first_available_scalar_str(z, ("sequence_id", "video_id", "seq_id"), default=path.stem))
        src_path = Path(_first_available_scalar_str(z, ("source_path",), default=str(path))).expanduser()

    if t_ms.shape[0] != xy.shape[0]:
        raise DataContractError(f"t_ms/xy frame mismatch in {path}: {t_ms.shape[0]} vs {xy.shape[0]}")
    if conf.shape[:2] != xy.shape[:2]:
        raise DataContractError(f"conf/xy shape mismatch in {path}: {conf.shape} vs {xy.shape}")

    return PoseSequence(
        sequence_id=seq_id,
        t_ms=np.asarray(t_ms, dtype=np.float32),
        xy=np.asarray(xy, dtype=np.float32),
        conf=np.asarray(conf, dtype=np.float32),
        fps=float(fps),
        source_path=src_path.resolve(),
    )


def _save_pose_sequence_npz(path: Path, seq: PoseSequence) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        sequence_id=np.asarray([seq.sequence_id]),
        source_path=np.asarray([str(seq.source_path)]),
        t_ms=np.asarray(seq.t_ms, dtype=np.float32),
        xy=np.asarray(seq.xy, dtype=np.float32),
        conf=np.asarray(seq.conf, dtype=np.float32),
        fps=np.asarray([float(seq.fps)], dtype=np.float32),
    )


def _first_available_1d_float(z: Any, keys: Sequence[str]) -> FloatArray | None:
    for key in keys:
        if key not in z.files:
            continue
        arr = np.asarray(z[key], dtype=np.float32).reshape(-1)
        if arr.size > 0:
            return arr
    return None


def _first_available_xy(z: Any) -> FloatArray | None:
    for key in ("xy", "joints", "raw_xy"):
        if key not in z.files:
            continue
        arr = np.asarray(z[key], dtype=np.float32)
        if arr.ndim == 3 and arr.shape[2] >= 2:
            return arr[..., :2]
    if "raw_xy_flat" in z.files:
        flat = np.asarray(z["raw_xy_flat"], dtype=np.float32).reshape(-1)
        n_j = int(_first_available_scalar_int(z, ("raw_joints", "n_joints"), default=33))
        t_ms = _first_available_1d_float(z, ("t_ms", "raw_t_ms", "timestamps_ms", "time_ms"))
        if t_ms is None:
            return None
        n_f = int(t_ms.size)
        expected = n_f * n_j * 2
        if n_f > 0 and n_j > 0 and flat.size == expected:
            return flat.reshape(n_f, n_j, 2)
    return None


def _first_available_conf(z: Any, *, n_frames: int, n_joints: int) -> FloatArray | None:
    for key in ("conf", "raw_conf", "joints_conf"):
        if key not in z.files:
            continue
        arr = np.asarray(z[key], dtype=np.float32)
        if arr.ndim == 2 and arr.shape[0] == n_frames and arr.shape[1] == n_joints:
            return arr
    if "raw_conf_flat" in z.files:
        flat = np.asarray(z["raw_conf_flat"], dtype=np.float32).reshape(-1)
        expected = n_frames * n_joints
        if flat.size == expected:
            return flat.reshape(n_frames, n_joints)
    return None


def _first_available_scalar_float(z: Any, keys: Sequence[str], *, default: float) -> float:
    for key in keys:
        if key not in z.files:
            continue
        try:
            return float(np.asarray(z[key]).reshape(-1)[0])
        except Exception:
            continue
    return float(default)


def _first_available_scalar_int(z: Any, keys: Sequence[str], *, default: int) -> int:
    for key in keys:
        if key not in z.files:
            continue
        try:
            return int(np.asarray(z[key]).reshape(-1)[0])
        except Exception:
            continue
    return int(default)


def _first_available_scalar_str(z: Any, keys: Sequence[str], *, default: str) -> str:
    for key in keys:
        if key not in z.files:
            continue
        try:
            raw = np.asarray(z[key]).reshape(-1)[0]
            return str(raw)
        except Exception:
            continue
    return str(default)


def _validate_ratio_sum(train_ratio: float, val_ratio: float, test_ratio: float) -> None:
    total = float(train_ratio) + float(val_ratio) + float(test_ratio)
    if not (0.999 <= total <= 1.001):
        raise DataContractError(f"Split ratios must sum to 1.0, got {total:.6f}.")
    if min(train_ratio, val_ratio, test_ratio) < 0.0:
        raise DataContractError("Split ratios must be non-negative.")


def _parse_label_value(value: Any) -> list[tuple[float, float]]:
    if isinstance(value, Mapping):
        spans_obj = value.get("fall_spans_ms", [])
    else:
        spans_obj = value
    if spans_obj is None:
        return []
    if not isinstance(spans_obj, Sequence):
        raise DataContractError("Label entries must be sequence or mapping with `fall_spans_ms`.")
    out: list[tuple[float, float]] = []
    for item in spans_obj:
        if not isinstance(item, Sequence) or len(item) < 2:
            continue
        start = float(item[0])
        end = float(item[1])
        if end <= start:
            continue
        out.append((start, end))
    return out


def _sequence_window_times(t_ms: FloatArray, *, duration_ms: float, stride_ms: float) -> tuple[tuple[float, float], ...]:
    if t_ms.size < 2:
        return ()
    start_global = float(t_ms[0])
    end_global = float(t_ms[-1])
    if end_global - start_global < duration_ms:
        return ()
    out: list[tuple[float, float]] = []
    end_t = start_global + duration_ms
    while end_t <= end_global + 1e-6:
        out.append((end_t - duration_ms, end_t))
        end_t += stride_ms
    return tuple(out)


def _slice_indices_for_interval(t_ms: FloatArray, *, start_ms: float, end_ms: float) -> tuple[int, int]:
    sidx = int(np.searchsorted(t_ms, float(start_ms), side="left"))
    eidx = int(np.searchsorted(t_ms, float(end_ms), side="right"))
    return sidx, eidx


def _label_for_window(*, start_ms: float, end_ms: float, spans_ms: Sequence[tuple[float, float]] | None) -> int:
    if spans_ms is None:
        return -1
    if not spans_ms:
        return 0
    for span_start, span_end in spans_ms:
        overlap = min(float(end_ms), float(span_end)) - max(float(start_ms), float(span_start))
        if overlap > 0.0:
            return 1
    return 0


def _materialize_effective_labels(
    *,
    pose_sequence_paths: Sequence[Path],
    labels: Mapping[str, tuple[tuple[float, float], ...]],
) -> Mapping[str, tuple[tuple[float, float], ...]]:
    """Return labels with safe fallback when an external map has no positives.

    If a provided label map contains no positive spans at all, infer spans from
    raw annotations and fill only missing/empty entries.
    """

    if _has_any_positive_spans(labels):
        return labels
    inferred = infer_labels_from_raw(pose_sequence_paths=pose_sequence_paths)
    if not inferred:
        return labels
    merged: dict[str, tuple[tuple[float, float], ...]] = {str(k): tuple(v) for k, v in labels.items()}
    for seq_id, spans in inferred.items():
        existing = merged.get(seq_id, ())
        if existing:
            continue
        merged[seq_id] = tuple(spans)
    return merged


def _has_any_positive_spans(labels: Mapping[str, Sequence[tuple[float, float]]]) -> bool:
    for spans in labels.values():
        if spans:
            return True
    return False


def _discover_le2i_sequences_with_labels(
    raw_dir: Path,
    *,
    strict_missing_labels: bool,
) -> tuple[DiscoveredSequenceRecord, ...]:
    records: list[DiscoveredSequenceRecord] = []
    video_paths = sorted(
        [p for p in raw_dir.rglob("*") if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS],
        key=lambda p: str(p),
    )
    for video_path in video_paths:
        seq_id = _sequence_id_from_path(video_path)
        ann_path = _resolve_le2i_annotation_path(video_path)
        spans: tuple[tuple[float, float], ...] = ()
        missing = False
        if ann_path is None:
            missing = True
            _warn_or_raise_label_issue(
                f"Label missing | dataset=le2i source={video_path.resolve()} annotation=NOT_FOUND",
                fail_fast=strict_missing_labels,
            )
        else:
            # LE2i annotations are frame-based, so this rebuild path must use the
            # dataset's nominal FPS to keep span reconstruction consistent.
            span = _parse_le2i_span_ms(ann_path=ann_path, fps=25.0)
            if span is not None:
                spans = (span,)
            else:
                frame_range = _parse_le2i_frame_range(ann_path)
                if frame_range is None:
                    missing = True
                    _warn_or_raise_label_issue(
                        f"Label missing | dataset=le2i source={video_path.resolve()} annotation={ann_path.resolve()} parse=FAILED",
                        fail_fast=strict_missing_labels,
                    )
                else:
                    start_f, end_f = frame_range
                    if max(start_f, end_f) > 0:
                        missing = True
                        _warn_or_raise_label_issue(
                            f"Label missing | dataset=le2i source={video_path.resolve()} annotation={ann_path.resolve()} frames=({start_f},{end_f})",
                            fail_fast=strict_missing_labels,
                        )
        records.append(
            DiscoveredSequenceRecord(
                sequence_id=seq_id,
                source_path=video_path.resolve(),
                fall_spans_ms=spans,
                label_missing=missing,
            )
        )
    return tuple(records)


def _discover_caucafall_sequences_with_labels(
    raw_dir: Path,
    *,
    strict_missing_labels: bool,
) -> tuple[DiscoveredSequenceRecord, ...]:
    records: list[DiscoveredSequenceRecord] = []
    video_paths = sorted(
        [p for p in raw_dir.rglob("*") if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS],
        key=lambda p: str(p),
    )
    for video_path in video_paths:
        seq_id = _sequence_id_from_path(video_path)
        # CAUCAFall uses the repo-wide 23 FPS contract; using 25 here would shift
        # raw-label reconstruction relative to the downstream training/eval stack.
        spans, has_annotations = _infer_caucafall_spans_from_frame_annotations(
            source=video_path,
            fps=23.0,
            sequence_id=seq_id,
        )
        missing = not bool(has_annotations)
        if missing:
            _warn_or_raise_label_issue(
                f"Label missing | dataset=caucafall source={video_path.resolve()} annotation=NOT_FOUND_OR_UNPARSEABLE",
                fail_fast=strict_missing_labels,
            )
        records.append(
            DiscoveredSequenceRecord(
                sequence_id=seq_id,
                source_path=video_path.resolve(),
                fall_spans_ms=tuple(sorted(spans, key=lambda it: (it[0], it[1]))),
                label_missing=missing,
            )
        )
    return tuple(records)


def infer_labels_from_raw(
    *,
    pose_sequence_paths: Sequence[Path],
) -> Mapping[str, tuple[tuple[float, float], ...]]:
    """Infer fall spans from raw dataset metadata when explicit labels are absent.

    Supported heuristics:
      1) LE2I: parse `Annotation_files/<video>.txt` first two lines as start/end frame.
      2) CAUCAFall: infer positive class from source path tokens containing `fall`.

    This is a recovery path for partial rebuilds. The normal pipeline should
    prefer explicit generated label artifacts over these dataset-specific fallbacks.

    Args:
        pose_sequence_paths: Canonical interim pose sequence artifacts.

    Returns:
        Mapping from sequence ID to inferred spans in milliseconds.

    Complexity:
        O(n) in sequence count plus annotation file reads.
    """

    out: dict[str, tuple[tuple[float, float], ...]] = {}
    for path in pose_sequence_paths:
        try:
            seq = _load_pose_sequence_npz(path)
        except Exception:
            continue
        seq_id = str(path.stem)
        source = _resolve_case_insensitive_path(seq.source_path)
        source_text = str(source).lower()
        fps = float(seq.fps if seq.fps > 0 else 30.0)

        inferred: list[tuple[float, float]] = []

        if "le2i" in source_text:
            ann_path = _resolve_le2i_annotation_path(source)
            if ann_path is None:
                _warn_or_raise_label_issue(
                    f"Label missing | dataset=le2i source={source} seq_id={seq_id} annotation=NOT_FOUND",
                )
            else:
                maybe_span = _parse_le2i_span_ms(ann_path=ann_path, fps=fps)
                if maybe_span is not None:
                    inferred.append(maybe_span)
                else:
                    frame_range = _parse_le2i_frame_range(ann_path)
                    if frame_range is None:
                        _warn_or_raise_label_issue(
                            f"Label missing | dataset=le2i source={source} seq_id={seq_id} annotation={ann_path} parse=FAILED",
                        )
                    else:
                        start_f, end_f = frame_range
                        if max(start_f, end_f) > 0:
                            _warn_or_raise_label_issue(
                                f"Label missing | dataset=le2i source={source} seq_id={seq_id} annotation={ann_path} frames=({start_f},{end_f})",
                            )

        has_caucafall_annotations = False
        if "caucafall" in source_text:
            caucafall_spans, has_caucafall_annotations = _infer_caucafall_spans_from_frame_annotations(
                source=source,
                fps=fps,
                sequence_id=seq_id,
            )
            if caucafall_spans:
                inferred.extend(caucafall_spans)
            elif has_caucafall_annotations and _source_likely_fall(source):
                _warn_or_raise_label_issue(
                    f"Label missing | dataset=caucafall source={source} seq_id={seq_id} parse=EMPTY_FALL_SPAN",
                )

        if inferred or has_caucafall_annotations:
            out[seq_id] = tuple(sorted(inferred, key=lambda it: (it[0], it[1])))
    return out


def _resolve_le2i_annotation_path(source: Path) -> Path | None:
    """Resolve LE2I annotation file path for one source video path."""

    source = _resolve_case_insensitive_path(source)
    search_roots: list[Path] = []
    if source.parent.parent != source.parent:
        search_roots.append(source.parent.parent)
    search_roots.append(source.parent)
    ann_dir_names = ("Annotation_files", "annotation_files")

    for root in search_roots:
        for ann_dir_name in ann_dir_names:
            ann_dir = root / ann_dir_name
            candidate = ann_dir / f"{source.stem}.txt"
            if candidate.is_file():
                return candidate
            if ann_dir.is_dir():
                normalized_stem = _normalize_token(source.stem)
                for txt_path in ann_dir.glob("*.txt"):
                    if _normalize_token(txt_path.stem) == normalized_stem:
                        return txt_path
        for ann_name in ("Annotation.txt", "annotation.txt"):
            candidate = root / ann_name
            if candidate.is_file():
                return candidate

    # Fallback: robust global lookup when source path casing/naming differs
    # from filesystem paths (e.g., `le2i` vs `LE2i`).
    scene_tokens = {_normalize_token(part) for part in source.parts if part}
    normalized_stem = _normalize_token(source.stem)
    for txt_path in _le2i_annotation_index():
        stem_ok = _normalize_token(txt_path.stem) == normalized_stem
        if not stem_ok:
            continue
        parent_tokens = {_normalize_token(part) for part in txt_path.parts if part}
        if scene_tokens and scene_tokens.intersection(parent_tokens):
            return txt_path
    return None


def _normalize_token(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(text).lower())


def _parse_le2i_span_ms(*, ann_path: Path, fps: float) -> tuple[float, float] | None:
    """Parse LE2I fall start/end frames from annotation text and convert to ms."""

    frame_range = _parse_le2i_frame_range(ann_path)
    if frame_range is None:
        return None
    start_f, end_f = frame_range
    if end_f < start_f:
        start_f, end_f = end_f, start_f

    # LE2i commonly uses `0,0` as "no fall".
    if start_f <= 0 and end_f <= 0:
        return None

    fps_eff = float(fps if np.isfinite(fps) and fps > 0.0 else 30.0)
    start_ms = (float(max(0, start_f)) / fps_eff) * 1000.0
    # Treat end frame as inclusive in annotation files.
    end_ms = (float(max(0, end_f) + 1.0) / fps_eff) * 1000.0
    if end_ms <= start_ms:
        return None
    return (start_ms, end_ms)


def _infer_caucafall_spans_from_frame_annotations(
    *,
    source: Path,
    fps: float,
    sequence_id: str | None = None,
) -> tuple[list[tuple[float, float]], bool]:
    """Infer CAUCAFall fall spans from per-frame annotation txt files.

    The dataset stores YOLO-style frame labels (`*.txt`) alongside videos/images.
    We detect positive class using `classes.txt` when present and then convert
    contiguous positive frame runs into millisecond spans.
    """

    ann_dir = _resolve_caucafall_annotation_dir(source=source, sequence_id=sequence_id)
    if ann_dir is None:
        _warn_or_raise_label_issue(
            f"Label missing | dataset=caucafall source={source} seq_id={sequence_id or ''} annotation_dir=NOT_FOUND",
        )
        return [], False
    classes_path = ann_dir / "classes.txt"
    positive_class_idx: int | None = None
    if classes_path.is_file():
        try:
            class_names = [ln.strip().lower() for ln in classes_path.read_text(encoding="utf-8", errors="ignore").splitlines() if ln.strip()]
        except Exception:
            class_names = []
        if "fall" in class_names:
            positive_class_idx = class_names.index("fall")
    if positive_class_idx is None:
        positive_class_idx = 1

    frame_labels: list[tuple[int, bool]] = []
    for txt_path in sorted(ann_dir.glob("*.txt"), key=lambda p: str(p)):
        if txt_path.name.lower() == "classes.txt":
            continue
        m = re.search(r"(\d+)$", txt_path.stem)
        if m is None:
            continue
        frame_idx = int(m.group(1))
        is_positive = False
        try:
            lines = txt_path.read_text(encoding="utf-8", errors="ignore").splitlines()
        except Exception:
            lines = []
        for line in lines:
            parts = line.strip().split()
            if not parts:
                continue
            token = parts[0].strip().lower()
            if token == "fall":
                is_positive = True
                break
            if token.isdigit() and int(token) == int(positive_class_idx):
                is_positive = True
                break
        frame_labels.append((frame_idx, is_positive))

    if not frame_labels:
        return [], False

    frame_labels_sorted = sorted(frame_labels, key=lambda it: it[0])
    base_frame_idx = frame_labels_sorted[0][0]
    positive_frames = sorted({(frame - base_frame_idx + 1) for frame, flag in frame_labels_sorted if flag})
    if not positive_frames:
        return [], True

    fps_eff = float(fps if np.isfinite(fps) and fps > 0.0 else 30.0)
    spans: list[tuple[float, float]] = []
    start = positive_frames[0]
    prev = positive_frames[0]
    for frame in positive_frames[1:]:
        if frame == prev + 1:
            prev = frame
            continue
        start_ms = (float(start) / fps_eff) * 1000.0
        end_ms = (float(prev + 1) / fps_eff) * 1000.0
        if end_ms > start_ms:
            spans.append((start_ms, end_ms))
        start = frame
        prev = frame
    start_ms = (float(start) / fps_eff) * 1000.0
    end_ms = (float(prev + 1) / fps_eff) * 1000.0
    if end_ms > start_ms:
        spans.append((start_ms, end_ms))
    return spans, True


def _parse_le2i_frame_range(ann_path: Path) -> tuple[int, int] | None:
    lines = [ln.strip() for ln in ann_path.read_text(encoding="utf-8", errors="ignore").splitlines() if ln.strip()]
    nums: list[int] = []
    for line in lines[:8]:
        match = re.search(r"-?\d+", line)
        if match is None:
            continue
        nums.append(int(match.group(0)))
        if len(nums) >= 2:
            break
    if len(nums) < 2:
        return None
    return int(nums[0]), int(nums[1])


@lru_cache(maxsize=1)
def _le2i_annotation_index() -> tuple[Path, ...]:
    roots = _dataset_roots_by_normalized_name("le2i")
    ann_files: list[Path] = []
    for root in roots:
        for p in root.rglob("*.txt"):
            if "annotation_files" not in _normalize_token(str(p.parent)):
                continue
            ann_files.append(p.resolve())
    return tuple(sorted(ann_files, key=lambda p: str(p)))


def _resolve_case_insensitive_path(path: Path) -> Path:
    p = Path(path).expanduser()
    if p.exists():
        return p.resolve()
    if not p.is_absolute():
        p = (Path.cwd() / p).resolve()
    anchor = Path(p.anchor) if p.anchor else Path("/")
    cur = anchor
    for part in p.parts[1:] if p.is_absolute() else p.parts:
        exact = cur / part
        if exact.exists():
            cur = exact
            continue
        if not cur.exists() or not cur.is_dir():
            return p
        lowered = str(part).lower()
        children = [child for child in cur.iterdir() if child.name.lower() == lowered]
        if len(children) == 1:
            cur = children[0]
            continue
        norm = _normalize_token(part)
        children = [child for child in cur.iterdir() if _normalize_token(child.name) == norm]
        if len(children) == 1:
            cur = children[0]
            continue
        return p
    return cur.resolve() if cur.exists() else p


def _dataset_roots_by_normalized_name(name: str) -> tuple[Path, ...]:
    target = _normalize_token(name)
    candidates: list[Path] = []
    cwd = Path.cwd()
    for base in (cwd / "data" / "raw", cwd):
        if not base.exists() or not base.is_dir():
            continue
        for child in base.iterdir():
            if child.is_dir() and _normalize_token(child.name) == target:
                candidates.append(child.resolve())
    return tuple(sorted({p for p in candidates}, key=lambda p: str(p)))


def _resolve_caucafall_annotation_dir(*, source: Path, sequence_id: str | None) -> Path | None:
    src = _resolve_case_insensitive_path(source)
    direct_candidates: list[Path] = []
    if src.is_dir():
        direct_candidates.append(src)
    else:
        direct_candidates.append(src.parent)
    direct_candidates.extend(src.parents[:3])
    for cand in direct_candidates:
        if cand.is_dir() and any(p.suffix.lower() == ".txt" for p in cand.glob("*.txt")):
            return cand.resolve()

    roots = _dataset_roots_by_normalized_name("caucafall")
    if not roots:
        return None
    all_ann_dirs: list[Path] = []
    for root in roots:
        for d in root.rglob("*"):
            if not d.is_dir():
                continue
            if any(p.suffix.lower() == ".txt" for p in d.glob("*.txt")):
                all_ann_dirs.append(d.resolve())
    if not all_ann_dirs:
        return None

    source_tokens = {_normalize_token(part) for part in src.parts if part}
    if sequence_id:
        source_tokens.add(_normalize_token(sequence_id))
    best: Path | None = None
    best_score = -1
    for d in all_ann_dirs:
        dtokens = {_normalize_token(part) for part in d.parts if part}
        score = len(source_tokens.intersection(dtokens))
        if score > best_score:
            best = d
            best_score = score
    return best


def _source_likely_fall(source: Path) -> bool:
    tokens = {_normalize_token(part) for part in source.parts if part}
    return any("fall" in token for token in tokens)


def _warn_or_raise_label_issue(message: str, *, fail_fast: bool | None = None) -> None:
    prefix = "LABEL PIPELINE WARNING: "
    should_fail_fast = (
        bool(fail_fast)
        if fail_fast is not None
        else str(os.getenv("FD_LABEL_FAIL_FAST", "0")).strip().lower() in {"1", "true", "yes", "on"}
    )
    if should_fail_fast:
        raise ValueError(prefix + message)
    warnings.warn(prefix + message, RuntimeWarning, stacklevel=2)


def _stratified_split_ids(
    *,
    pos: Sequence[str],
    neg: Sequence[str],
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> tuple[list[str], list[str], list[str]]:
    rng = random.Random(int(seed))
    pos_ids = list(pos)
    neg_ids = list(neg)
    rng.shuffle(pos_ids)
    rng.shuffle(neg_ids)

    def _split_group(ids: Sequence[str]) -> tuple[list[str], list[str], list[str]]:
        n = len(ids)
        n_train = int(math.floor(n * train_ratio))
        n_val = int(math.floor(n * val_ratio))
        n_test = max(0, n - n_train - n_val)
        train = list(ids[:n_train])
        val = list(ids[n_train : n_train + n_val])
        test = list(ids[n_train + n_val : n_train + n_val + n_test])
        return train, val, test

    pos_train, pos_val, pos_test = _split_group(pos_ids)
    neg_train, neg_val, neg_test = _split_group(neg_ids)

    train = pos_train + neg_train
    val = pos_val + neg_val
    test = pos_test + neg_test
    rng.shuffle(train)
    rng.shuffle(val)
    rng.shuffle(test)
    return train, val, test


def _ensure_eval_splits_have_positive(
    train: list[str],
    val: list[str],
    test: list[str],
    positive_ids: set[str],
) -> None:
    if not positive_ids:
        return
    def _move_one(src: list[str], dst: list[str]) -> bool:
        for idx, sid in enumerate(src):
            if sid in positive_ids:
                dst.append(src.pop(idx))
                return True
        return False

    if not any(sid in positive_ids for sid in val):
        if not _move_one(train, val):
            _move_one(test, val)
    if not any(sid in positive_ids for sid in test):
        if not _move_one(train, test):
            _move_one(val, test)


__all__ = [
    "DiscoveredSequenceRecord",
    "PipelinePaths",
    "PoseSequence",
    "build_pipeline_paths",
    "build_splits",
    "create_label_template",
    "discover_sequences_with_labels",
    "discover_raw_sources",
    "export_windows",
    "extract_pose_sequences",
    "infer_labels_from_raw",
    "load_label_map",
    "load_splits",
    "rebuild_labels_and_splits_from_raw",
]
