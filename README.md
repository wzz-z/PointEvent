# Supplemental Inference Code

This package contains the minimal code needed to run test-time inference with the provided checkpoints.

## Data Layout

Place test `.npz` files under:

```text
data/EV-UAV/test/*.npz
data/ES-UAV/test/*.npz
```

Each `.npz` file should contain event data in `ev` or `evs_norm` format. The expected columns are `x, y, t, polarity, label`.

## Run Inference

EV-UAV:

```bash
python tools/test.py --config-file configs/EVUAV/evuav.py
```

ES-UAV:

```bash
python tools/test.py --config-file configs/EVUAV/esuav.py
```

Override paths if the data or checkpoint is stored elsewhere:

```bash
python tools/test.py --config-file configs/EVUAV/evuav.py --options data_root=/path/to/data weight=/path/to/model_best.pth save_path=exp/test
```

Outputs are saved under `save_path`, including prediction `.npz`, optional `.txt`, visualizations, and `test.log`.
