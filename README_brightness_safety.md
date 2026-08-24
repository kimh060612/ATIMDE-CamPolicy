# Image-driven brightness safety envelope

`orbbec_ati_geometric_search_update.py`, `orbbec_ati_risk_bandit.py`, and
`orbbec_ati_neldermead.py` separate camera safety into three layers:

1. `SafetyPolicy.safe_cells()` applies the camera grid, the motion-dependent exposure ceiling, disabled context cells, and the configured gain set.
2. `BrightnessGuard` computes luminance/clipping statistics from the current valid RGB frame and derives a dynamic EV envelope inside that hard-safe set.
3. The learned controller proposes only inside the resulting admissible set.

Only `severe_over` in `enforce` mode bypasses the learned switch decision. It deterministically steps to a darker hard-safe cell, preferring an exact -1 EV exposure reduction. Ordinary `over` merely blocks brighter proposals; the learned score still decides whether to switch.

## Modes

- `off`: no image statistics and exactly the legacy control path. This is also the default when `brightness_guard` is absent.
- `log`: record statistics and decisions without changing proposals, simplex state, or commits.
- `filter`: restrict proposals to the brightness envelope; learned scores retain final authority.
- `enforce`: apply filtering and force deterministic recovery for `severe_over`.

Run the calibration configuration in logging mode first:

```bash
python orbbec_ati_geometric_search_update.py \
  --camera-error-checkpoint CHECKPOINT.pt \
  --safety-config config/safety_envelope_brightness.json
```

The same `--safety-config` option enables the guard for the risk-bandit and
Nelder–Mead entrypoints. Risk-bandit applies the current frame's decision to
the next pre-applied cell immediately. Nelder–Mead has a capture-then-observe
state machine, so the decision constrains its next round; a filtered or forced
cell restarts the simplex from the cell that was actually captured.

The `brightness_guard` section controls EMA/debounce, luminance clipping and shadow thresholds, tile-level clipping evidence, the target median brightness, and EV envelope radii. The CSV records the smoothed percentiles and ratios used for the decision, state transitions, admissible-cell count, preferred direction, and recovery availability/action.

These defaults are initial values only. Calibrate them on Orbbec images from the actual lens, exposure/gain grid, lighting, and scene distribution before changing `mode` from `log` to `filter` or `enforce`.

Evaluate these ablations separately:

- learned policy only (`off`)
- brightness guard only (use logged/enforced guard decisions without learned proposal claims)
- learned policy + brightness guard (`filter` or `enforce`)

## Motion-state warning

The current mapping is `stop=0`, `slow=1`, `fast=2`, `rotate=3`, `spin=4`. The example array `[16, 16, 32, 16, 8]` therefore permits 32 ms in `fast`. Confirm that this is intentional for the deployment; this implementation does not reinterpret or change that value.
