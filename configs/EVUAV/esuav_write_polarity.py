_base_ = ["./esuav.py"]

model = dict(backbone=dict(
    polarity_write_mode="polarity",
    polarity_encoding="zero_one",
    relation_scale=(0.03, 0.03, 0.03),
))
save_path = "exp/ESUAV/write_v1_polarity"
