for rep in {1..10}; do
  python3 orbbec_drl_control.py \
	  --max-frames 200 \
	  --settle-frames 4 \
	  --disable-awb \
	  --gain-min-db 1.0 \
	  --gain-max-db 24 \
	  --output-dir "runs/drl_control_scene1_dim_rep${rep}"
done
