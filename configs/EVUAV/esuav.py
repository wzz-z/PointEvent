_base_ = ["../_base_/default_runtime.py"]

data_root = "data/ES-UAV"
weight = "exp/ES-UAV/model/model_best.pth"
save_path = "exp/ES-UAV/test"

num_classes = 2
ignore_index = -1
names = ["background", "target"]

grid_size = (0.02, 0.02, 0.02)
gather_num = 1
size = (720, 1280)
shuffle_orders = False

test_npz_key = "evs_norm"
test_vis_dpi = 200
test_save_txt = True
test_save_pic = True
test_save_npz = True
target_class = 1
target_thresh = 0.9

out_channel = 8
hidden_channel = 16

model = dict(
    type="DefaultSegmentor",
    num_classes=num_classes,
    backbone_out_channels=out_channel,
    criteria=None,
    backbone=dict(
        type="PointEvent",
        in_channels=4,
        gather_num=gather_num,
        order=["z", "z-trans", "hilbert", "hilbert-trans"],
        stride=2,
        enc_depths=(2, 4),
        enc_channels=(out_channel, hidden_channel),
        dec_depths=(2,),
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
)

data = dict(
    num_classes=num_classes,
    ignore_index=ignore_index,
    names=names,
    test=dict(
        type="denoise_evuav",
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
