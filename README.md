# Supplemental Code

This package contains the code for "PointEvent: Polarity-Conditioned Motion Evidence Accumulation for Event-Based Tiny Object Detection".

## Data Layout

Place test `.npz` files under:

```text
data/EV-UAV/test/*.npz
data/ES-UAV/test/*.npz
```

Each `.npz` file should contain event data in `ev` or `evs_norm` format. The expected columns are `x, y, t, polarity, label`.

## Run Training

```bash
python tools/train.py \
    --config-file configs/EVUAV/evuav_polarity_accum.py \
    --num-gpus 1 \
    --options save_path=exp/EVUAV/polarity_state_accum
```

or simply `bash scripts/train.sh`.

## Run Inference

```bash
python tools/test.py \
    --config-file configs/EVUAV/evuav_polarity_accum.py \
    --num-gpus 1 \
    --options save_path=exp/EVUAV/test \
              weight=exp/EVUAV/polarity_state_accum/model/model_last.pth
```


Outputs are saved under `save_path`, including prediction `.npz`, optional `.txt`, visualizations, and `test.log`.
