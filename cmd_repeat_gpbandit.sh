for rep in {1..5}; do
  python3 orbbec_ati_risk_bandit.py \
	  --camera-error-checkpoint ../ati_mde_ckpt/scalar_camera_induced_error_degrade_lr5_softCE_vanilla/ckpt_model_epoch190.pt \
	  --safety-config ./config/safety_envelop.json \
	  --motion-source ros \
	  --precision fp32 \
	  --max-rounds 200 \
	  --settle-frames 4 \
	  --initial-exposure-ms 8 \
	  --initial-gain 64 \
	  --q-uncertainty-weight 1.645 \
	  --bandit-window-size 48 \
	  --bandit-exploration-beta 1.0 \
	  --bandit-switch-penalty 0.005 \
	  --bandit-temporal-scale-sec 5.0 \
	  --evaluation-precision fp32 \
	  --lighting-state normal \
	  --output-dir "runs/normal_gpbandit_v1_scene3_rep${rep}"
done
