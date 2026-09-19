for rep in {1..5}; do
  python3 measure_latency_iqa_controller.py \
    --mode 2 \
    --num-frames 200 \
    --settle-frames 4 \
    --initial-exposure-ms 16 \
    --initial-gain 64 \
    --disable-awb \
    --output-dir "runs/iqa_measure_latency_rep${rep}"
done
