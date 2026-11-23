PYTHONPATH="$(pwd)" CUDA_VISIBLE_DEVICES=0 python tools/train.py --config-file configs/EVUAV/evuav_polarity_accum.py --num-gpus 1 --options save_path=exp/EVUAV/polarity_state_accum

PYTHONPATH="$(pwd)" CUDA_VISIBLE_DEVICES=0 python tools/train.py --config-file configs/EVUAV/esuav_polarity_accum.py --num-gpus 1 --options save_path=exp/ESUAV/polarity_state_accum
