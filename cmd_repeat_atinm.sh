for rep in {1..5}; do
  python3 orbbec_ati_neldermead.py \
	  --camera-error-checkpoint ../ati_mde_ckpt/scalar_camera_induced_error_degrade_lr5_softCE_vanilla/ckpt_model_epoch190.pt \
	  --safety-config config/safety_envelop.json \
	  --motion-source ros \
	  --max-rounds 200 \
	  --q-uncertainty-weight 1.645 \
	  --simplex-restart-frames 60 \
	  --simplex-tolerance 0.02 \
	  --evaluation-precision fp32 \
	  --initial-exposure-ms 8 \
	  --initial-gain 64 \
	  --settle-frames 4 \
	  --precision fp32 \
	  --lighting-state dark \
	  --output-dir "runs/dark_nm_v1_scene3_rep${rep}"
done
