# Delay-Aware Risk-Sensitive Time-Varying Contextual GP Bandit

`orbbec_ati_risk_bandit.py` is an independent RGB camera sensor-control path.
It minimizes the trained camera-error predictor's existing `QScore.q` risk over
the safe subset of `hardware.utils.ALL_CELLS`. It does not physically probe the
candidate grid: each round captures one RGB-D frame, while the NumPy exact-GP
surrogate evaluates at most 16 safe actions in memory.

This entrypoint loads `config/safety_envelop.json` by default. For every latest
motion/light context, candidate exposure is capped by
`max_exposure_ms_by_motion`, gain is restricted by `allowed_gains_by_light`,
and context-specific entries in `disabled_cells_by_context` are removed. Pass
`--safety-config /path/to/another.json` to override the default envelope.

The GP uses the product of action-grid, categorical motion/light, and temporal
kernels. Its rolling history is bounded (48 observations by default), and its
LCB acquisition separates GP epistemic uncertainty from the aleatoric
uncertainty already included in `QScore.q`.

Each round selects the next safe cell from observations available before that
round's inference, then calls the camera-error model's full-depth
`predict_batch()` synchronously. The score and raw depth map come from that one
model call; the depth is saved under `depth_pred_raw/` before the next camera
cell is applied and the single captured frame is logged. No background
inference thread or robot motion command is used by this path.

After control ends, the executable directly uses the shared
`FairDepthEvaluator` and then writes the unchanged `CaptureLogger` CSV schema.

Example:

```bash
python orbbec_ati_risk_bandit.py \
  --camera-error-checkpoint /path/to/checkpoint.pt \
  --output-dir /path/to/risk_bandit_output \
  --safety-config config/safety_envelop.json \
  --motion-source ros \
  --max-rounds 200 \
  --q-uncertainty-weight 1.645 \
  --bandit-window-size 48 \
  --bandit-exploration-beta 1.0 \
  --bandit-switch-penalty 0.005 \
  --bandit-temporal-scale-sec 5.0 \
  --evaluation-precision fp32
```

The value `1.645` is only an example corresponding to a Gaussian upper 95%
VaR; the common CLI default remains unchanged.
