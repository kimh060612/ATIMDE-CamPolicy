#!/usr/bin/env python3
"""Evaluate captured Orbbec RGB/depth pairs with Depth Anything V2.

The relative Depth Anything V2 output is affine-aligned to the ground-truth
inverse depth (scale + shift), converted back to metric depth, and evaluated
with AbsRel and delta-1 (A1).
"""

from __future__ import annotations

import argparse
import csv
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np


DEFAULT_DATASET_ROOT = Path(
    "/home/wego/ati_workspace/ati_dataset/ae_orbbec_mde_scene/"
    "pair_000_auto_exposure"
)
DEFAULT_MODEL_ID = "depth-anything/Depth-Anything-V2-Small-hf"
DEFAULT_OUTPUT = Path("evaluate_ae_captured_results.csv")

EVALUATION_FIELDS = (
    "record_type",
    "lap_name",
    "evaluation_status",
    "evaluation_model_id",
    "evaluation_alignment",
    "evaluation_device",
    "evaluation_depth_scale_m_per_raw_unit",
    "evaluation_inference_ms",
    "alignment_scale",
    "alignment_shift",
    "abs_rel",
    "a1",
    "valid_depth_pixels",
    "evaluated_frames",
    "failed_frames",
    "evaluation_error",
)


@dataclass(frozen=True)
class Sample:
    lap_name: str
    metadata_path: Path
    metadata: dict[str, str]


class DepthAnythingV2Predictor:
    """Small inference-only wrapper around the Hugging Face DA-V2 model."""

    def __init__(
        self,
        model_id: str,
        device: str,
        precision: str,
        local_files_only: bool,
    ) -> None:
        try:
            import torch
            import torch.nn.functional as torch_f
            from transformers import AutoImageProcessor, AutoModelForDepthEstimation
        except ImportError as exc:
            raise RuntimeError(
                "Depth Anything V2 requires torch, transformers, and pillow. "
                "Install the dependencies in requirements_orbbec_absrel.txt."
            ) from exc

        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        if device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError(f"CUDA device {device!r} was requested but is unavailable.")

        self.torch = torch
        self.torch_f = torch_f
        self.device = torch.device(device)
        self.use_fp16 = precision == "fp16" or (
            precision == "auto" and self.device.type == "cuda"
        )
        if self.use_fp16 and self.device.type != "cuda":
            raise ValueError("fp16 inference is only supported on CUDA by this script.")
        self.dtype = torch.float16 if self.use_fp16 else torch.float32

        print(f"[Model] Loading processor: {model_id}", flush=True)
        self.processor = AutoImageProcessor.from_pretrained(
            model_id, local_files_only=local_files_only
        )
        model_kwargs: dict[str, Any] = {"local_files_only": local_files_only}
        if self.use_fp16:
            model_kwargs["torch_dtype"] = torch.float16
        print(
            f"[Model] Loading model: {model_id} "
            f"(device={self.device}, precision={'fp16' if self.use_fp16 else 'fp32'})",
            flush=True,
        )
        self.model = AutoModelForDepthEstimation.from_pretrained(
            model_id, **model_kwargs
        )
        self.model.eval().to(self.device)

    def predict(
        self, image_bgr: np.ndarray, target_shape: tuple[int, int]
    ) -> tuple[np.ndarray, float]:
        torch = self.torch
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        pixel_values = self.processor(
            images=image_rgb, return_tensors="pt"
        )["pixel_values"].to(
            device=self.device, dtype=self.dtype, non_blocking=True
        )

        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        started = time.perf_counter()
        with torch.inference_mode():
            with torch.autocast(
                device_type=self.device.type,
                dtype=torch.float16,
                enabled=self.use_fp16,
            ):
                prediction = self.model(pixel_values=pixel_values).predicted_depth
            if prediction.ndim == 3:
                prediction = prediction.unsqueeze(1)
            prediction = self.torch_f.interpolate(
                prediction.float(),
                size=target_shape,
                mode="bicubic",
                align_corners=False,
            )[0, 0]
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        inference_ms = (time.perf_counter() - started) * 1000.0
        return (
            np.ascontiguousarray(prediction.cpu().numpy(), dtype=np.float32),
            inference_ms,
        )


def _fit_scale_shift(
    prediction: np.ndarray, target: np.ndarray
) -> tuple[float, float]:
    """Fit ``scale * prediction + shift`` to target by least squares."""

    predicted = prediction.astype(np.float64, copy=False)
    expected = target.astype(np.float64, copy=False)
    predicted_centered = predicted - predicted.mean()
    denominator = float(np.dot(predicted_centered, predicted_centered))
    if not math.isfinite(denominator) or denominator < 1e-12:
        raise ValueError("Affine alignment is singular for this frame.")
    scale = float(
        np.dot(predicted_centered, expected - expected.mean()) / denominator
    )
    shift = float(expected.mean() - scale * predicted.mean())
    if not math.isfinite(scale) or not math.isfinite(shift) or scale <= 0.0:
        raise ValueError(
            f"Affine alignment returned invalid scale/shift: {scale}, {shift}."
        )
    return scale, shift


def evaluate_prediction(
    prediction: np.ndarray,
    target_m: np.ndarray,
    *,
    alignment: str,
    min_depth_m: float,
    max_depth_m: float,
    min_valid_pixels: int,
) -> dict[str, float | int]:
    """Affine-align one prediction and calculate frame-level AbsRel and A1."""

    if prediction.ndim != 2 or target_m.ndim != 2:
        raise ValueError(
            f"Prediction and GT must be 2-D; got {prediction.shape}, {target_m.shape}."
        )
    if prediction.shape != target_m.shape:
        prediction = cv2.resize(
            prediction,
            (target_m.shape[1], target_m.shape[0]),
            interpolation=cv2.INTER_CUBIC,
        )

    valid = (
        np.isfinite(prediction)
        & np.isfinite(target_m)
        & (target_m >= min_depth_m)
        & (target_m <= max_depth_m)
    )
    if int(valid.sum()) < min_valid_pixels:
        raise ValueError(
            f"Only {int(valid.sum())} valid GT pixels; require {min_valid_pixels}."
        )

    if alignment == "scale_shift_inverse":
        fit_target = 1.0 / np.maximum(target_m[valid], 1e-6)
        scale, shift = _fit_scale_shift(prediction[valid], fit_target)
        aligned_inverse = scale * prediction + shift
        valid &= np.isfinite(aligned_inverse) & (aligned_inverse > 1e-6)
        aligned_depth = 1.0 / np.maximum(aligned_inverse, 1e-6)
    elif alignment == "scale_shift_depth":
        scale, shift = _fit_scale_shift(prediction[valid], target_m[valid])
        aligned_depth = scale * prediction + shift
    else:
        raise ValueError(f"Unsupported alignment mode: {alignment}")

    valid &= np.isfinite(aligned_depth) & (aligned_depth > 1e-6)
    valid_count = int(valid.sum())
    if valid_count < min_valid_pixels:
        raise ValueError(
            f"Only {valid_count} valid pixels remain after affine alignment."
        )

    predicted = aligned_depth[valid].astype(np.float64, copy=False)
    expected = target_m[valid].astype(np.float64, copy=False)
    ratio = np.maximum(predicted / expected, expected / predicted)
    return {
        "alignment_scale": scale,
        "alignment_shift": shift,
        "abs_rel": float(np.mean(np.abs(predicted - expected) / expected)),
        "a1": float(np.mean(ratio < 1.25)),
        "valid_depth_pixels": valid_count,
    }


def load_samples(dataset_root: Path) -> tuple[list[Sample], list[str]]:
    """Read every ``lap_*/metadata.csv`` in deterministic order."""

    metadata_paths = sorted(dataset_root.glob("lap_*/metadata.csv"))
    if not metadata_paths:
        raise FileNotFoundError(f"No lap_*/metadata.csv found under {dataset_root}")

    samples: list[Sample] = []
    metadata_fields: list[str] = []
    seen_fields: set[str] = set()
    for metadata_path in metadata_paths:
        with metadata_path.open("r", newline="", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle)
            if not reader.fieldnames:
                raise ValueError(f"CSV has no header: {metadata_path}")
            for field in reader.fieldnames:
                if field not in seen_fields:
                    metadata_fields.append(field)
                    seen_fields.add(field)
            for row in reader:
                samples.append(
                    Sample(metadata_path.parent.name, metadata_path, dict(row))
                )

    if not samples:
        raise ValueError(f"No metadata rows found under {dataset_root}")
    for required in ("rgb_path", "depth_path"):
        if required not in seen_fields:
            raise ValueError(f"Required metadata column {required!r} is missing.")
    return samples, metadata_fields


def resolve_data_path(
    raw_path: str, sample: Sample, dataset_root: Path, subdirectory: str
) -> Path:
    """Resolve absolute, scene-relative, pair-relative, and lap-relative paths."""

    path = Path(raw_path).expanduser()
    candidates: Iterable[Path]
    if path.is_absolute():
        candidates = (path,)
    else:
        candidates = (
            dataset_root.parent / path,
            dataset_root / path,
            sample.metadata_path.parent / path,
            sample.metadata_path.parent / subdirectory / path.name,
        )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"Could not resolve {subdirectory} path: {raw_path}")


def _result_row(
    sample: Sample,
    *,
    model_id: str,
    alignment: str,
    device: str,
    depth_scale: float,
) -> dict[str, Any]:
    row: dict[str, Any] = dict(sample.metadata)
    row.update(
        record_type="frame",
        lap_name=sample.lap_name,
        evaluation_status="",
        evaluation_model_id=model_id,
        evaluation_alignment=alignment,
        evaluation_device=device,
        evaluation_depth_scale_m_per_raw_unit=depth_scale,
        evaluation_inference_ms="",
        alignment_scale="",
        alignment_shift="",
        abs_rel="",
        a1="",
        valid_depth_pixels="",
        evaluated_frames="",
        failed_frames="",
        evaluation_error="",
    )
    return row


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate all lap_*/metadata.csv RGB/depth pairs with affine-aligned "
            "Depth Anything V2 relative depth."
        )
    )
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, cuda:0, ...")
    parser.add_argument(
        "--precision", choices=("auto", "fp16", "fp32"), default="auto"
    )
    parser.add_argument(
        "--alignment",
        choices=("scale_shift_inverse", "scale_shift_depth"),
        default="scale_shift_inverse",
        help="Affine alignment space (default: inverse depth for relative DA-V2).",
    )
    parser.add_argument(
        "--depth-scale",
        type=float,
        default=0.001,
        help="GT meters per stored depth unit (this dataset stores millimeters).",
    )
    parser.add_argument("--min-depth-m", type=float, default=0.1)
    parser.add_argument("--max-depth-m", type=float, default=10.0)
    parser.add_argument("--min-valid-pixels", type=int, default=1000)
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="Do not download the model; use the local Hugging Face cache only.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Evaluate only the first N frames (useful for a smoke test).",
    )
    parser.add_argument(
        "--progress-every", type=int, default=10, help="Progress print interval."
    )
    args = parser.parse_args()
    if args.depth_scale <= 0:
        parser.error("--depth-scale must be positive")
    if not 0 < args.min_depth_m < args.max_depth_m:
        parser.error("require 0 < --min-depth-m < --max-depth-m")
    if args.min_valid_pixels < 1:
        parser.error("--min-valid-pixels must be positive")
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    if args.progress_every < 1:
        parser.error("--progress-every must be positive")
    return args


def main() -> int:
    args = parse_args()
    dataset_root = args.dataset_root.expanduser().resolve()
    output_path = args.output.expanduser().resolve()
    samples, metadata_fields = load_samples(dataset_root)
    if args.limit is not None:
        samples = samples[: args.limit]

    predictor = DepthAnythingV2Predictor(
        args.model_id, args.device, args.precision, args.local_files_only
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(EVALUATION_FIELDS) + [
        field for field in metadata_fields if field not in EVALUATION_FIELDS
    ]

    abs_rel_values: list[float] = []
    a1_values: list[float] = []
    failed = 0
    print(
        f"[Dataset] {len(samples)} frames from {dataset_root}; output={output_path}",
        flush=True,
    )
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for index, sample in enumerate(samples, start=1):
            result = _result_row(
                sample,
                model_id=args.model_id,
                alignment=args.alignment,
                device=str(predictor.device),
                depth_scale=args.depth_scale,
            )
            try:
                rgb_path = resolve_data_path(
                    sample.metadata["rgb_path"], sample, dataset_root, "rgb"
                )
                depth_path = resolve_data_path(
                    sample.metadata["depth_path"], sample, dataset_root, "depth"
                )
                image = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
                if image is None:
                    raise OSError(f"Failed to read RGB image: {rgb_path}")
                target_m = np.load(depth_path, allow_pickle=False).astype(
                    np.float32, copy=False
                )
                if target_m.ndim != 2:
                    raise ValueError(
                        f"GT depth must be 2-D, got {target_m.shape}: {depth_path}"
                    )
                target_m = np.ascontiguousarray(target_m * args.depth_scale)
                prediction, inference_ms = predictor.predict(image, target_m.shape)
                metrics = evaluate_prediction(
                    prediction,
                    target_m,
                    alignment=args.alignment,
                    min_depth_m=args.min_depth_m,
                    max_depth_m=args.max_depth_m,
                    min_valid_pixels=args.min_valid_pixels,
                )
                result.update(
                    evaluation_status="ok",
                    evaluation_inference_ms=inference_ms,
                    **metrics,
                )
                abs_rel_values.append(float(metrics["abs_rel"]))
                a1_values.append(float(metrics["a1"]))
            except (OSError, ValueError, RuntimeError) as exc:
                failed += 1
                result.update(evaluation_status="error", evaluation_error=str(exc))

            writer.writerow(result)
            handle.flush()
            if (
                index == 1
                or index == len(samples)
                or index % args.progress_every == 0
            ):
                metric_text = (
                    f"AbsRel={result['abs_rel']:.6f} A1={result['a1']:.6f}"
                    if result["evaluation_status"] == "ok"
                    else f"ERROR={result['evaluation_error']}"
                )
                print(
                    f"[Evaluate] {index}/{len(samples)} {sample.lap_name} "
                    f"frame={sample.metadata.get('frame_index', '?')} {metric_text}",
                    flush=True,
                )

        if not abs_rel_values:
            print(
                f"[Summary] No frame was evaluated successfully; failures={failed}",
                flush=True,
            )
            return 1

        mean_abs_rel = float(np.mean(abs_rel_values))
        mean_a1 = float(np.mean(a1_values))
        summary = {field: "" for field in fieldnames}
        summary.update(
            record_type="summary",
            evaluation_status="ok" if failed == 0 else "partial",
            evaluation_model_id=args.model_id,
            evaluation_alignment=args.alignment,
            evaluation_device=str(predictor.device),
            evaluation_depth_scale_m_per_raw_unit=args.depth_scale,
            abs_rel=mean_abs_rel,
            a1=mean_a1,
            evaluated_frames=len(abs_rel_values),
            failed_frames=failed,
        )
        writer.writerow(summary)
        handle.flush()

    print(
        f"[Summary] evaluated={len(abs_rel_values)} failed={failed} "
        f"Mean AbsRel={mean_abs_rel:.6f} Mean A1={mean_a1:.6f}",
        flush=True,
    )
    print(f"[Summary] CSV saved to {output_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
