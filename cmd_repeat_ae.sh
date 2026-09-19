for rep in {1..5}; do
	python3 orbbec_ae_control.py --num-frames 200 --warmup-frames 8 --output-dir "runs/scene2_ae_dark_rep${rep}"
done
