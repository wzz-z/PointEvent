PYTHONPATH="$(pwd)" CUDA_VISIBLE_DEVICES=0 python tools/test.py --config-file configs/EVUAV/evuav_polarity_accum.py --num-gpus 1 --options save_path=exp/EVUAV/test weight=exp/EVUAV/polarity_state_accum/model/model_last.pth

PYTHONPATH="$(pwd)" CUDA_VISIBLE_DEVICES=0 python tools/test.py --config-file configs/EVUAV/esuav_polarity_accum.py --num-gpus 1 --options save_path=exp/ESUAV/test weight=exp/ESUAV/polarity_state_accum/model/model_last.pth
