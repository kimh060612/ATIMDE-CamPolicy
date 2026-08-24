for rep in {1..5}; do
  python3 orbbec_iqa_grid_control.py \
    --mode 2 \
    --num-frames 200 \
    --settle-frames 4 \
    --output-dir "runs/iqa_grid_control_mode2_scene2_normal_rep${rep}"
done
