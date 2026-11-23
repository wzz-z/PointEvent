_base_ = ["../_base_/default_runtime.py"]

# EV-UAV point-wise binary detection
batch_size = 24
batch_size_val = 1
batch_size_test = 1
lr_per_sample = 0.00008
grid_size = (0.02, 0.02, 0.02)
point_max = 40000

gather_num = 1
epoch = 100
eval_epoch = 10

seed = 1024
num_worker = 8
empty_cache = False
enable_amp = False

num_classes = 2
ignore_index = -1
names = ["background", "target"]

data_root = ".."
dataset_type = "denoise_evuav"

size = (720, 1280)
shuffle_orders = False

test_npz_key = "evs_norm"
test_vis_dpi = 200
test_save_txt = True
test_save_pic = True

out_channel = 12
hidden_channel = 24

model = dict(
    type="DefaultSegmentor",
    num_classes=num_classes,
    backbone_out_channels=out_channel,
    backbone=dict(
        type="PointEvent",
        in_channels=4,
        gather_num=gather_num,
        order=["z", "z-trans", "hilbert", "hilbert-trans"],
        stride=(2),
        enc_depths=(2, 4),
        enc_channels=(out_channel, hidden_channel),
        dec_depths=(2),
        dec_channels=(out_channel,),
        mlp_ratio=4,
        proj_drop=0.0,
        drop_path=0.3,
        pre_norm=True,
        shuffle_orders=shuffle_orders,
        cls_mode=False,
        pdnorm_bn=False,
        pdnorm_ln=False,
        pdnorm_decouple=True,
        pdnorm_adaptive=False,
        pdnorm_affine=True,
        pdnorm_conditions=("ScanNet", "S3DIS", "Structured3D"),
    ),
    criteria=[
        dict(
    type="CrossEntropyLoss",
    weight=[1.0, 15.0],
    loss_weight=1.0,
    ignore_index=ignore_index,
),
    ],
)

optimizer = dict(type="AdamW", lr=lr_per_sample * batch_size, weight_decay=0.005)
scheduler = dict(
    type="OneCycleLR",
    max_lr=[lr_per_sample * batch_size, lr_per_sample * batch_size * 0.1],
    pct_start=0.05,
    anneal_strategy="cos",
    div_factor=10.0,
    final_div_factor=1000.0,
)
param_dicts = [dict(keyword="block", lr=lr_per_sample * batch_size * 0.1)]

data = dict(
    num_classes=num_classes,
    ignore_index=ignore_index,
    names=names,
    train=dict(
        type=dataset_type,
        split="train",
        data_root=data_root,
        gather_num=gather_num,
        transform=[
            dict(
                type="GridSampleEventNorm",
                grid_size=grid_size,
                hash_type="fnv",
                mode="train",
                keys=("coord", "strength", "segment"),
                return_grid_coord=True,
            ),
            dict(type="SphereCrop", point_max=point_max, mode="random"),
            dict(type="ToTensor"),
            dict(
                type="Collect",
                keys=("coord", "grid_coord", "segment"),
                feat_keys=("coord", "strength"),
            ),
        ],
        test_mode=False,
        ignore_index=ignore_index,
    ),
    val=dict(
        type=dataset_type,
        split="val",
        data_root=data_root,
        gather_num=gather_num,
        transform=[
            dict(
                type="GridSampleEventNorm",
                grid_size=grid_size,
                hash_type="fnv",
                mode="train",
                keys=("coord", "strength", "segment"),
                return_grid_coord=True,
            ),
            dict(type="ToTensor"),
            dict(
                type="Collect",
                keys=("coord", "grid_coord", "segment"),
                feat_keys=("coord", "strength"),
            ),
        ],
        test_mode=False,
        ignore_index=ignore_index,
    ),
    test=dict(
        type=dataset_type,
        split="test",
        data_root=data_root,
        gather_num=gather_num,
        transform=[
            dict(type="Copy", keys_dict={"segment": "origin_segment"}),
            dict(
                type="GridSampleEventNorm",
                grid_size=grid_size,
                hash_type="fnv",
                mode="train",
                keys=("coord", "strength", "segment"),
                return_inverse=True,
            ),
        ],
        test_mode=True,
        test_cfg=dict(
            scale2kitti=None,
            voxelize=dict(
                type="GridSampleEventNorm",
                grid_size=grid_size,
                hash_type="fnv",
                mode="test",
                return_grid_coord=True,
                keys=("coord", "strength"),
            ),
            crop=None,
            post_transform=[
                dict(type="ToTensor"),
                dict(
                    type="Collect",
                    keys=("coord", "grid_coord", "index"),
                    feat_keys=("coord", "strength"),
                ),
            ],
            aug_transform=[[dict(type="Add")]],
        ),
        ignore_index=ignore_index,
    ),
)
