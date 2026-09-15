import torch
import torch.nn as nn

from models.utils.structure import Point_V2
from .builder import MODELS, build_model

@MODELS.register_module()
class DefaultSegmentor(nn.Module):
    def __init__(
        self,
        num_classes,
        backbone_out_channels,
        backbone=None,
        criteria=None,
    ):
        super().__init__()
        self.seg_head = nn.Sequential(
            nn.Linear(backbone_out_channels, num_classes),
            ) if num_classes > 0 else nn.Identity()
        self.backbone = build_model(backbone)
        self.criteria = None
        if criteria is not None:
            from models.losses import build_criteria
            self.criteria = build_criteria(criteria)

    def forward(self, input_dict):
        point_0 = Point_V2(input_dict)
        point_1 = self.backbone(point_0)
        seg_logits = self.seg_head(point_1.feat)
        # seg_logits = point_0.feat[:,:4]-seg_logits_0
        # print(seg_logits.shape)
        # print(input_dict["segment"].shape)

        if self.training:
            if self.criteria is None:
                raise RuntimeError("Training criteria are not configured.")
            loss = self.criteria(seg_logits, input_dict["segment"])
            return dict(loss=loss)
        # evals
        elif "segment" in input_dict.keys():
            if self.criteria is None:
                return dict(seg_logits=seg_logits)
            return dict(
                loss=self.criteria(seg_logits, input_dict["segment"]),
                seg_logits=seg_logits,
            )
        # test
        else:
            return dict(seg_logits=seg_logits)

@MODELS.register_module()
class DefaultClassifier(nn.Module):
    def __init__(
        self,
        backbone=None,
        criteria=None,
        num_classes=40,
        backbone_embed_dim=256,
    ):
        super().__init__()
        self.backbone = build_model(backbone)
        self.criteria = None
        if criteria is not None:
            from models.losses import build_criteria
            self.criteria = build_criteria(criteria)
        self.num_classes = num_classes
        self.backbone_embed_dim = backbone_embed_dim
        self.cls_head = nn.Sequential(
            nn.Linear(backbone_embed_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(128, num_classes),
        )

    def forward(self, input_dict):
        feat = self.backbone(input_dict)
        cls_logits = self.cls_head(feat)
        if self.training:
            if self.criteria is None:
                raise RuntimeError("Training criteria are not configured.")
            loss = self.criteria(cls_logits, input_dict["category"])
            return dict(loss=loss)
        elif "category" in input_dict.keys():
            if self.criteria is None:
                return dict(cls_logits=cls_logits)
            return dict(
                loss=self.criteria(cls_logits, input_dict["category"]),
                cls_logits=cls_logits,
            )
        else:
            return dict(cls_logits=cls_logits)
