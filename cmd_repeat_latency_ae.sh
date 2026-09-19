for rep in {1..5}; do
	python3 measure_latency_ae.py --num-frames 200 --warmup-frames 8 --output-dir "runs/ae_measure_latency_rep${rep}"
done
