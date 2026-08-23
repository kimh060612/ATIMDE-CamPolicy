for rep in {1..5}; do
  python3 orbbec_ati_geometric_search_update.py \
    --model-size small \
    --device cuda \
    --precision fp32 \
    --motion-source ros \
    --max-rounds 200 \
    --settle-frames 4 \
    --max-pair-capture-gap-ms 250 \
    --invalid-edge-cooldown-rounds 2 \
    --max-consecutive-invalid-pairs 2 \
    --recovery-current-only-rounds 1 \
    --periodic-reprobe-interval 30 \
    --pending-std-floor 1e-4 \
    --camera-error-checkpoint ../ati_mde_ckpt/scalar_camera_induced_error_degrade_lr5_softCE_vanilla/ckpt_model_epoch190.pt \
    --settle-frames 4 \
    --max-pair-capture-gap-ms 250 \
    --safety-config ./config/safety_envelop.json \
    --probe-trigger-threshold 0.03 \
    --switch-margin 0.005 \
    --pair-uncertainty-weight 0.25 \
    --lighting-state dark \
    --initial-exposure-ms 8 \
    --initial-gain 64 \
    --simplex-memory-ttl-rounds 10 \
    --output-dir "runs/dark_geosearch_v11_tuned13_scene2_rep${rep}"
done
