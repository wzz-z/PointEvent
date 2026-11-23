_base_ = ["./evuav.py"]

# One SSM per scan, total state dimension 16 = positive 8 + negative 8.
# Feature widths/depths and training settings follow evuav.py (including any
# server-side change from 8/16 channels to 12/24 channels).
model = dict(backbone=dict(
    polarity_write_mode="polarity_state",
    polarity_encoding="zero_one",
))
save_path = "exp/EVUAV/polarity_state_accum"
