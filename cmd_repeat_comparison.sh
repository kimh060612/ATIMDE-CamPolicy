for rep in {1..5}; do
  python3 orbbec_iqa_control.py \
    --mode 2 \
    --num-frames 200 \
    --settle-frames 4 \
    --initial-exposure-ms 16 \
    --initial-gain 64 \
    --disable-awb \
    --output-dir "runs/iqa_controlfixed_scene4_dark_rep${rep}"
done
