for rep in {1..5}; do
  python3 orbbec_iqa_control.py \
    --mode 2 \
    --num-frames 200 \
    --gain-step 16 \
    --exposure-max-ms 32 \
    --settle-frames 4 \
    --output-dir "runs/iqa_grid_control_mode2_scene2_normal_rep${rep}"
done
