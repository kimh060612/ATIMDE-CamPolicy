# Orbbec deterministic probing with Depth Anything V2 AbsRel

## What changed

- Captures synchronized RGB + depth frames.
- Uses Orbbec `AlignFilter` to align depth to the color view.
- Converts Orbbec depth to metres using `raw_depth * depth_scale / 1000`.
- Loads Depth Anything V2 once and keeps it on the selected Torch device.
- Uses CUDA FP16 by default on Jetson.
- Computes `q = AbsRel(predicted_depth, aligned_depth_gt)`.
- Captures every cell in the current four-cell probing cycle **before** running DNN inference, so model latency does not increase the interval between the four parameterized captures.
- Saves aligned GT as lossless 16-bit PNG in millimetres unless `--no-save-depth` is used.

## Recommended Jetson setup

Use the PyTorch build compatible with the JetPack version installed on the Jetson. Do not replace it blindly with a generic CPU-only PyPI wheel.

Install the remaining Python dependencies in the same environment:

```bash
pip install "transformers>=4.45" pillow safetensors numpy opencv-python
pip install --upgrade pyorbbecsdk2
```

Verify CUDA before running:

```bash
python - <<'PY'
import torch
print("torch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("CUDA version:", torch.version.cuda)
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
PY
```

## Default: metric indoor model

The default checkpoint is:

```text
depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf
```

It is the most appropriate default for computing direct metric AbsRel in an indoor Orbbec experiment.

```bash
python orbbec_deterministic_probing_absrel.py \
  --motion-state 0 \
  --light-state 0 \
  --output-dir runs/absrel_metric \
  --depth-device cuda \
  --depth-precision fp16 \
  --depth-score-mode metric \
  --min-depth-m 0.2 \
  --max-depth-m 10.0 \
  --reset-state
```

The probing controller minimizes `q`, so a lower AbsRel is better.

## Relative Depth Anything V2 checkpoint

A relative checkpoint cannot be compared directly with metre-valued GT. Choose an alignment mode explicitly.

For disparity/inverse-depth-like output:

```bash
python orbbec_deterministic_probing_absrel.py \
  --depth-model-id depth-anything/Depth-Anything-V2-Small-hf \
  --depth-score-mode scale_shift_inverse \
  --depth-device cuda \
  --depth-precision fp16 \
  --output-dir runs/absrel_relative \
  --reset-state
```

For a model whose output should be aligned in the depth domain:

```bash
--depth-score-mode scale_shift_depth
```

## Cached/offline model

Download/cache the checkpoint once, then run without network access:

```bash
--depth-model-local-files-only
```

## Output

```text
runs/absrel_metric/
├── controller_state.json
├── probing_log.csv
├── images/
└── depth_gt/
```

`probing_log.csv` stores the q score. The `extra_json` column includes:

- `abs_rel`
- `valid_depth_pixels`
- `valid_depth_ratio`
- `depth_alignment_scale`
- `depth_alignment_shift`
- `depth_inference_ms`

## Important interpretation

This q is an **oracle experiment score**, because it uses aligned RGB-D ground truth during control. It is suitable for testing whether the deterministic probing algorithm selects good exposure/gain cells. It is not the final deployable structural-risk model, which must operate without GT.

The aligned depth GT may still contain RGB-D occlusion-boundary errors, invalid zeros, flying pixels, and material-dependent failures. The implementation masks invalid/out-of-range GT, but results should still be checked visually and with a minimum valid-pixel threshold.
