"""
Checkpoint Utils for Models

Author: Zongze Wu (zongze0907@gmail.com)
Please cite our work if the code is helpful to you.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from .builder import LOSSES
from pytorch3d.ops import knn_points, knn_gather
import faiss
import numpy as np

@LOSSES.register_module()
class CrossEntropyLoss(nn.Module):
    def __init__(
        self,
        weight=None,
        size_average=None,
        reduce=None,
        reduction="mean",
        label_smoothing=0.0,
        loss_weight=1.0,
        ignore_index=-1,
    ):
        super(CrossEntropyLoss, self).__init__()
        weight = torch.tensor(weight).cuda() if weight is not None else None
        self.loss_weight = loss_weight
        self.loss = nn.CrossEntropyLoss(
            weight=weight,
            size_average=size_average,
            ignore_index=ignore_index,
            reduce=reduce,
            reduction=reduction,
            label_smoothing=label_smoothing,
        )

    def forward(self, pred, target):
        return self.loss(pred, target) * self.loss_weight

@LOSSES.register_module()
class FrequencyLoss(nn.Module):
    def __init__(self, reduction="mean", loss_weight=1.0, ignore_index=-1):
        super(FrequencyLoss, self).__init__()
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.ignore_index = ignore_index

    def forward(self, pred, target):
        pred_labels = torch.argmax(pred, dim=1) # (N, C) -> (N,)

        if self.ignore_index >= 0:
            mask = target != self.ignore_index
            pred_labels = pred_labels[mask].contiguous()
            target = target[mask].contiguous()

        pred_labels = pred_labels.float().view(-1, 1)
        target = target.float().view(-1, 1)

        with torch.no_grad():
            pred_fft = torch.fft.fft(pred_labels).abs()
            target_fft = torch.fft.fft(target).abs()

        diff = torch.abs(pred_fft - target_fft)
        diff = diff / (torch.max(diff, torch.tensor(1e-6, device=diff.device)) + 1e-8)
        loss = torch.mean(diff**2)

        # reduction
        if self.reduction == "mean":
            loss = loss.mean()
        elif self.reduction == "sum":
            loss = loss.sum()

        return loss * self.loss_weight 

@LOSSES.register_module()
class WeightedCrossEntropyLoss(nn.Module):
    def __init__(
        self,
        size_average=None,
        reduce=None,
        reduction="mean",
        label_smoothing=0.0,
        loss_weight=1.0,
        ignore_index=-1,
    ):
        super(WeightedCrossEntropyLoss, self).__init__()
        self.loss_weight = loss_weight
        self.size_average = size_average
        self.reduce = reduce
        self.reduction = reduction
        self.label_smoothing = label_smoothing
        self.ignore_index = ignore_index

        # 创建 CrossEntropyLoss，但权重动态生成
        self.loss = nn.CrossEntropyLoss(
            weight=None,  # 初始不设置权重，动态计算
            size_average=size_average,
            ignore_index=ignore_index,
            reduce=reduce,
            reduction=reduction,
            label_smoothing=label_smoothing,
        )

    def forward(self, pred, target):
        # Step 1: 动态计算权重
        class_counts = torch.bincount(target.flatten(), minlength=pred.size(1))
        # 避免除零
        class_weights = 1.0 / (class_counts + 1e-6)
        class_weights /= class_weights.sum()  # 归一化权重
        class_weights = class_weights.to(pred.device)

        # Step 2: 更新权重到 loss
        self.loss.weight = class_weights
        loss = self.loss(pred, target) * self.loss_weight
        # print(f"Class weights: {class_weights.cpu().numpy()}")
        # print(f"Loss: {loss.item()}")
        # Step 3: 计算损失并加权
        return loss

@LOSSES.register_module()
class BCEWithLogitsLoss(nn.Module):
    def __init__(
        self,
        weight=None,
        size_average=None,
        reduce=None,
        reduction="mean",
        loss_weight=1.0,
        pos_weight=None,  # 用于不平衡数据集
    ):
        super(BCEWithLogitsLoss, self).__init__()
        weight = torch.tensor(weight).cuda() if weight is not None else None
        pos_weight = torch.tensor(pos_weight).cuda() if pos_weight is not None else None
        self.loss_weight = loss_weight
        self.loss = nn.BCEWithLogitsLoss(
            weight=weight,  # 权重对每个样本应用
            reduction=reduction,  # 如何对 batch 的 loss 进行归约
            pos_weight=pos_weight,  # 正样本的权重（针对类别不平衡）
        )

    def forward(self, pred, target):
        target = target.float()  # BCEWithLogitsLoss 要求 target 是 float 类型
        return self.loss(pred, target) * self.loss_weight

@LOSSES.register_module()
class SmoothCELoss(nn.Module):
    def __init__(self, smoothing_ratio=0.1):
        super(SmoothCELoss, self).__init__()
        self.smoothing_ratio = smoothing_ratio

    def forward(self, pred, target):
        eps = self.smoothing_ratio
        n_class = pred.size(1)
        one_hot = torch.zeros_like(pred).scatter(1, target.view(-1, 1), 1)
        one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (n_class - 1)
        log_prb = F.log_softmax(pred, dim=1)
        loss = -(one_hot * log_prb).total(dim=1)
        loss = loss[torch.isfinite(loss)].mean()
        return loss


@LOSSES.register_module()
class BinaryFocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=0.5, logits=True, reduce=True, loss_weight=1.0):
        """Binary Focal Loss
        <https://arxiv.org/abs/1708.02002>`
        """
        super(BinaryFocalLoss, self).__init__()
        assert 0 < alpha < 1
        self.gamma = gamma
        self.alpha = alpha
        self.logits = logits
        self.reduce = reduce
        self.loss_weight = loss_weight

    def forward(self, pred, target, **kwargs):
        """Forward function.
        Args:
            pred (torch.Tensor): The prediction with shape (N)
            target (torch.Tensor): The ground truth. If containing class
                indices, shape (N) where each value is 0≤targets[i]≤1, If containing class probabilities,
                same shape as the input.
        Returns:
            torch.Tensor: The calculated loss
        """
        target = target[:, None].float()
        if self.logits:
            bce = F.binary_cross_entropy_with_logits(pred, target, reduction="none")
        else:
            bce = F.binary_cross_entropy(pred, target, reduction="none")
        pt = torch.exp(-bce)
        alpha = self.alpha * target + (1 - self.alpha) * (1 - target)
        focal_loss = alpha * (1 - pt) ** self.gamma * bce

        if self.reduce:
            focal_loss = torch.mean(focal_loss)
        return focal_loss * self.loss_weight


@LOSSES.register_module()
class FocalLoss(nn.Module):
    def __init__(
        self, gamma=2.0, alpha=0.5, reduction="mean", loss_weight=1.0, ignore_index=-1
    ):
        """Focal Loss
        <https://arxiv.org/abs/1708.02002>`
        """
        super(FocalLoss, self).__init__()
        assert reduction in (
            "mean",
            "sum",
        ), "AssertionError: reduction should be 'mean' or 'sum'"
        assert isinstance(
            alpha, (float, list)
        ), "AssertionError: alpha should be of type float"
        assert isinstance(gamma, float), "AssertionError: gamma should be of type float"
        assert isinstance(
            loss_weight, float
        ), "AssertionError: loss_weight should be of type float"
        assert isinstance(ignore_index, int), "ignore_index must be of type int"
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.ignore_index = ignore_index

    def forward(self, pred, target, **kwargs):
        """Forward function.
        Args:
            pred (torch.Tensor): The prediction with shape (N, C) where C = number of classes.
            target (torch.Tensor): The ground truth. If containing class
                indices, shape (N) where each value is 0≤targets[i]≤C−1, If containing class probabilities,
                same shape as the input.
        Returns:
            torch.Tensor: The calculated loss
        """
        # [B, C, d_1, d_2, ..., d_k] -> [C, B, d_1, d_2, ..., d_k]
        # print("Prediction Tensor: ", pred)
        # print("Target Tensor: ", target)
        pred = pred.transpose(0, 1)
        # [C, B, d_1, d_2, ..., d_k] -> [C, N]
        pred = pred.reshape(pred.size(0), -1)
        # [C, N] -> [N, C]
        pred = pred.transpose(0, 1).contiguous()
        # (B, d_1, d_2, ..., d_k) --> (B * d_1 * d_2 * ... * d_k,)
        target = target.view(-1).contiguous()
        assert pred.size(0) == target.size(
            0
        ), "The shape of pred doesn't match the shape of target"
        valid_mask = target != self.ignore_index
        target = target[valid_mask]
        pred = pred[valid_mask]

        if len(target) == 0:
            return 0.0

        num_classes = pred.size(1)
        target = F.one_hot(target, num_classes=num_classes)

        alpha = self.alpha
        if isinstance(alpha, list):
            alpha = pred.new_tensor(alpha)
        pred_sigmoid = pred.sigmoid()
        target = target.type_as(pred)
        one_minus_pt = (1 - pred_sigmoid) * target + pred_sigmoid * (1 - target)
        focal_weight = (alpha * target + (1 - alpha) * (1 - target)) * one_minus_pt.pow(
            self.gamma
        )

        loss = (
            F.binary_cross_entropy_with_logits(pred, target, reduction="none")
            * focal_weight
        )
        if self.reduction == "mean":
            loss = loss.mean()
        elif self.reduction == "sum":
            loss = loss.total()
        return self.loss_weight * loss


@LOSSES.register_module()
class DiceLoss(nn.Module):
    def __init__(self, smooth=1, exponent=2, loss_weight=1.0, ignore_index=-1):
        """DiceLoss.
        This loss is proposed in `V-Net: Fully Convolutional Neural Networks for
        Volumetric Medical Image Segmentation <https://arxiv.org/abs/1606.04797>`_.
        """
        super(DiceLoss, self).__init__()
        self.smooth = smooth
        self.exponent = exponent
        self.loss_weight = loss_weight
        self.ignore_index = ignore_index

    def forward(self, pred, target, **kwargs):
        # [B, C, d_1, d_2, ..., d_k] -> [C, B, d_1, d_2, ..., d_k]
        pred = pred.transpose(0, 1)
        # [C, B, d_1, d_2, ..., d_k] -> [C, N]
        pred = pred.reshape(pred.size(0), -1)
        # [C, N] -> [N, C]
        pred = pred.transpose(0, 1).contiguous()
        # (B, d_1, d_2, ..., d_k) --> (B * d_1 * d_2 * ... * d_k,)
        target = target.view(-1).contiguous()
        assert pred.size(0) == target.size(
            0
        ), "The shape of pred doesn't match the shape of target"
        valid_mask = target != self.ignore_index
        target = target[valid_mask]
        pred = pred[valid_mask]

        pred = F.softmax(pred, dim=1)
        num_classes = pred.shape[1]
        target = F.one_hot(
            torch.clamp(target.long(), 0, num_classes - 1), num_classes=num_classes
        )

        total_loss = 0
        for i in range(num_classes):
            if i != self.ignore_index:
                num = torch.sum(torch.mul(pred[:, i], target[:, i])) * 2 + self.smooth
                den = (
                    torch.sum(
                        pred[:, i].pow(self.exponent) + target[:, i].pow(self.exponent)
                    )
                    + self.smooth
                )
                dice_loss = 1 - num / den
                total_loss += dice_loss
        loss = total_loss / num_classes
        return self.loss_weight * loss

@LOSSES.register_module()
class CombinedPointCloudLoss(nn.Module):
    def __init__(self, chamfer_weight=1.0, intensity_weight=0.5, use_gpu=True):
        """
        合并的点云损失计算类，支持 Chamfer Distance 和 Intensity Loss。

        Args:
            chamfer_weight (float): Chamfer Distance 的损失权重。
            intensity_weight (float): Intensity Loss 的损失权重。
            use_gpu (bool): 是否使用 GPU 加速。
        """
        super(CombinedPointCloudLoss, self).__init__()
        self.chamfer_weight = chamfer_weight
        self.intensity_weight = intensity_weight
        self.use_gpu = use_gpu

    def knn_search_batch(self, pred_coords, target_coords, batch_size=10000, use_gpu=True):
        """
        分批次使用 Faiss 进行最近邻搜索，减少显存压力。
        
        Args:
            pred_coords: (N, 3) 预测点的坐标 (NumPy array)。
            target_coords: (M, 3) 目标点的坐标 (NumPy array)。
            batch_size: 每次处理的点数量。
            use_gpu: 是否使用 GPU 加速。
        
        Returns:
            indices_pred_to_target: (N,) 每个预测点最近邻目标点的索引。
            indices_target_to_pred: (M,) 每个目标点最近邻预测点的索引。
        """
        if use_gpu:
            res = faiss.StandardGpuResources()  # 创建 GPU 资源
            index = faiss.IndexFlatL2(3)
            gpu_index = faiss.index_cpu_to_gpu(res, 0, index)  # 将索引迁移到 GPU
        else:
            index = faiss.IndexFlatL2(3)

        # 构建索引
        if use_gpu:
            gpu_index.add(target_coords)
        else:
            index.add(target_coords)

        # 分批处理预测点
        num_pred = pred_coords.shape[0]
        indices_pred_to_target = []

        for i in range(0, num_pred, batch_size):
            batch_pred = pred_coords[i:i + batch_size]
            if use_gpu:
                _, indices = gpu_index.search(batch_pred, 1)
            else:
                _, indices = index.search(batch_pred, 1)
            indices_pred_to_target.append(indices)

        indices_pred_to_target = np.concatenate(indices_pred_to_target, axis=0).squeeze(1)

        # 构建反向索引 (目标 -> 预测)
        if use_gpu:
            gpu_index.reset()
            gpu_index.add(pred_coords)
            _, indices_target_to_pred = gpu_index.search(target_coords, 1)
        else:
            index.reset()
            index.add(pred_coords)
            _, indices_target_to_pred = index.search(target_coords, 1)

        return indices_pred_to_target, indices_target_to_pred.squeeze(1)


    def forward(self, pred, target):
        """
        前向计算两个点云的合并损失。

        Args:
            pred (torch.Tensor): 预测的点云 (N, 4)，包含 (x, y, z, intensity)。
            target (torch.Tensor): GT 点云 (M, 4)，包含 (x, y, z, intensity)。

        Returns:
            total_loss (torch.Tensor): 合并后的总损失。
        """
        # 提取坐标和强度
        pred_coords = pred[:, :3].detach().cpu().numpy()  # (N, 3)
        target_coords = target[:, :3].detach().cpu().numpy()  # (M, 3)
        pred_intensity = pred[:, 3]  # (N,)
        target_intensity = target[:, 3]  # (M,)

        # 最近邻查找
        indices_pred_to_target, indices_target_to_pred = self.knn_search_batch(pred_coords, target_coords)  # Pred -> Target

        # **Chamfer Distance 计算**
        chamfer_loss = 0
        if self.chamfer_weight > 0:
            dist_pred_to_target = torch.norm(pred[:, :3] - target[indices_pred_to_target, :3], dim=1)
            dist_target_to_pred = torch.norm(target[:, :3] - pred[indices_target_to_pred, :3], dim=1)
            chamfer_loss = torch.mean(dist_pred_to_target) + torch.mean(dist_target_to_pred)

        # **Intensity Loss 计算**
        intensity_loss = 0
        if self.intensity_weight > 0:
            matched_target_intensity = target_intensity[indices_pred_to_target]
            intensity_loss = F.mse_loss(pred_intensity, matched_target_intensity)

        # 合并损失
        total_loss = self.chamfer_weight * chamfer_loss + self.intensity_weight * intensity_loss
        return total_loss

@LOSSES.register_module()
class ChamferDistance(nn.Module):
    def __init__(self, loss_weight=1.0, use_gpu=True):
        """
        Chamfer Distance损失，用于度量点云之间的距离。
        
        Args:
            loss_weight (float): 损失的权重。
            use_gpu (bool): 是否使用GPU加速。
        """
        super(ChamferDistance, self).__init__()
        self.loss_weight = loss_weight
        self.use_gpu = use_gpu

    def forward(self, adv_pc, ori_pc):
        """
        adv_pc: (N, 4) 预测的点云，包含(x, y, z, p)的信息
        ori_pc: (M, 4) GT 点云，包含(x, y, z, p)的信息
        """
        adv_xyz = adv_pc[:, :3].detach().cpu().numpy()   # (N, 3)
        ori_xyz = ori_pc[:, :3].detach().cpu().numpy()  # (M, 3)

        # 使用 Faiss 进行最近邻查找
        if self.use_gpu:
            res = faiss.StandardGpuResources()  # 创建 GPU 资源
            index = faiss.IndexFlatL2(3)  # L2 距离索引
            gpu_index = faiss.index_cpu_to_gpu(res, 0, index)  # 将索引迁移到 GPU
            gpu_index.add(ori_xyz)  # 将目标点云添加到索引中

            # 查询预测点的最近邻
            _, indices_adv_to_ori = gpu_index.search(adv_xyz, 1)  # 查找 adv 到 ori 的最近邻
            _, indices_ori_to_adv = gpu_index.search(ori_xyz, 1)  # 查找 ori 到 adv 的最近邻
        else:
            index = faiss.IndexFlatL2(3)  # L2 距离索引 (CPU)
            index.add(ori_xyz)  # 将目标点云添加到索引中

            # 查询预测点的最近邻
            _, indices_adv_to_ori = index.search(adv_xyz, 1)  # 查找 adv 到 ori 的最近邻
            _, indices_ori_to_adv = index.search(ori_xyz, 1)  # 查找 ori 到 adv 的最近邻

        # 计算距离
        min_dist_adv_to_ori = torch.mean(torch.from_numpy(indices_adv_to_ori).float())  # 计算 adv 到 ori 的最小距离
        min_dist_ori_to_adv = torch.mean(torch.from_numpy(indices_ori_to_adv).float())  # 计算 ori 到 adv 的最小距离

        # 计算 Chamfer Loss
        chamfer_loss = min_dist_adv_to_ori + min_dist_ori_to_adv
        return chamfer_loss * self.loss_weight

# @LOSSES.register_module()
# class ChamferDistance(nn.Module):
#     def __init__(self, loss_weight=1.0):
#         """
#         Chamfer Distance损失，用于度量点云之间的距离。
        
#         Args:
#             loss_weight (float): 损失的权重。
#         """
#         super(ChamferDistance, self).__init__()
#         self.loss_weight = loss_weight

#     def forward(self, adv_pc, ori_pc):
#             """
#             pred: (N, 4) 去雨后点云，包含(x, y, z, p)的信息
#             target: (M, 4) 去雨前的点云 GT，包含(x, y, z, p)的信息
#             """
#             adv_xyz = adv_pc[:, :3]
#             ori_xyz = ori_pc[:, :3]

#             # 使用 KNN 查找最近邻点
#             adv_KNN = knn_points(adv_xyz.unsqueeze(0), ori_xyz.unsqueeze(0), K=1)
#             ori_KNN = knn_points(ori_xyz.unsqueeze(0), adv_xyz.unsqueeze(0), K=1)

#             # 计算几何距离的 Chamfer Distance
#             min_dist_adv_to_ori = adv_KNN.dists.squeeze(0).min(dim=1)[0]
#             min_dist_ori_to_adv = ori_KNN.dists.squeeze(0).min(dim=1)[0]
#             chamfer_loss = torch.mean(min_dist_adv_to_ori) + torch.mean(min_dist_ori_to_adv)
#             return chamfer_loss*self.loss_weight

# @LOSSES.register_module()
# class IntensityLoss(nn.Module):
#     def __init__(self, loss_weight=1.0):
#         super(IntensityLoss, self).__init__()
#         self.loss_weight = loss_weight

#     def forward(self, pred, target):
#         """
#         pred: (N, 4) 预测的点云，包含(x, y, z, p)的信息
#         target: (M, 4) GT 点云，包含(x, y, z, p)的信息
#         """
#         pred_intensity = pred[:, 3]  # 预测的强度
#         target_intensity = target[:, 3]  # GT强度

#         # 如果 pred 和 target 的大小不同，我们进行最近邻匹配
#         if pred.size(0) != target.size(0):
#             # 计算空间坐标的欧氏距离 (N, M)
#             dist = torch.cdist(pred[:, :3], target[:, :3], p=2)  # 计算预测点和目标点之间的欧氏距离
#             min_dist_pred_to_gt, idx = dist.min(dim=1)  # 获取每个预测点到最近目标点的索引
#             target_intensity = target_intensity[idx]  # 使用这些索引来选择目标点的强度值

#         # 计算强度的L2损失
#         intensity_loss = F.mse_loss(pred_intensity, target_intensity)
#         return intensity_loss*self.loss_weight
@LOSSES.register_module()
class IntensityLoss(nn.Module):
    def __init__(self, loss_weight=1.0, use_gpu=True):
        super(IntensityLoss, self).__init__()
        self.loss_weight = loss_weight
        self.use_gpu = use_gpu

    def forward(self, pred, target):
        """
        pred: (N, 4) 预测的点云，包含(x, y, z, p)的信息
        target: (M, 4) GT 点云，包含(x, y, z, p)的信息
        """
        pred_intensity = pred[:, 3]  # 预测的强度
        target_intensity = target[:, 3]  # GT 强度

        # 提取点的空间坐标
        pred_coords = pred[:, :3].detach().cpu().numpy()  # (N, 3)
        target_coords = target[:, :3].detach().cpu().numpy()  # (M, 3)

        # 构建 Faiss 索引
        if self.use_gpu:
            res = faiss.StandardGpuResources()  # 创建 GPU 资源
            index = faiss.IndexFlatL2(3)  # L2 距离索引
            gpu_index = faiss.index_cpu_to_gpu(res, 0, index)  # 将索引迁移到 GPU
            gpu_index.add(target_coords)  # 将目标点云添加到索引中
            _, indices = gpu_index.search(pred_coords, 1)  # 查询最近邻
        else:
            index = faiss.IndexFlatL2(3)  # L2 距离索引 (CPU)
            index.add(target_coords)  # 将目标点云添加到索引中
            _, indices = index.search(pred_coords, 1)  # 查询最近邻

        # 根据最近邻匹配获取目标点的强度
        indices = indices.squeeze(1)  # (N,)
        matched_target_intensity = target_intensity[indices]

        # 计算强度的 L2 损失
        intensity_loss = F.mse_loss(pred_intensity, matched_target_intensity)
        return intensity_loss * self.loss_weight

@LOSSES.register_module()
class ChamferDistanceWithPolarity(nn.Module):
    def __init__(self, loss_weight=1.0, alpha=1.0, beta=0.5):
        """
        Chamfer Distance损失，结合极性信息来度量点云之间的距离。

        Args:
            loss_weight (float): 总体损失的权重。
            alpha (float): 几何距离的权重。
            beta (float): 极性差异的权重。
        """
        super(ChamferDistanceWithPolarity, self).__init__()
        self.loss_weight = loss_weight
        self.alpha = alpha  # 几何距离的权重
        self.beta = beta    # 极性差异的权重

    def forward(self, pred, target):
        """
        pred: (N, 4) 预测的点云，包含 (x, y, z, p) 的信息
        target: (M, 4) GT点云，包含 (x, y, z, p) 的信息
        """
        # 分离出空间坐标和极性信息
        pred_xyz = pred[:, :3]  # (N, 3)，空间坐标
        target_xyz = target[:, :3]  # (M, 3)，GT空间坐标
        pred_p = pred[:, 3].unsqueeze(1)  # (N, 1)，预测点的极性
        target_p = target[:, 3].unsqueeze(0)  # (1, M)，GT点的极性

        # 计算几何距离 (x, y, z)
        dist_xyz = torch.cdist(pred_xyz, target_xyz, p=2) ** 2  # (N, M)，平方欧氏距离

        # 计算极性差异 (p)
        dist_p = (pred_p - target_p) ** 2  # (N, M)，极性差异平方

        # 综合距离: 几何距离 + 极性差异 (加权)
        combined_dist = self.alpha * dist_xyz + self.beta * dist_p  # (N, M)

        # 计算 Chamfer 距离
        min_dist_pred_to_gt = combined_dist.min(dim=1)[0]  # 每个预测点到最近GT点的距离
        min_dist_gt_to_pred = combined_dist.min(dim=0)[0]  # 每个GT点到最近预测点的距离

        # Chamfer损失: 对于每个点，计算最小距离的平方，并进行求和
        chamfer_loss = torch.mean(min_dist_pred_to_gt) + torch.mean(min_dist_gt_to_pred)

        return chamfer_loss * self.loss_weight

@LOSSES.register_module()
class chamfer_loss_with_intensity(nn.Module):
    def __init__(self, loss_weight=1.0, intensity_weight=0.5):
        super(chamfer_loss_with_intensity, self).__init__()
        self.loss_weight = loss_weight
        self.intensity_weight = intensity_weight

    def forward(self, adv_pc, ori_pc):
        device = adv_pc.device

        # 只考虑空间坐标 (x, y, z)
        adv_xyz = adv_pc[:, :3]
        ori_xyz = ori_pc[:, :3]

        # 使用 KNN 查找最近邻点
        adv_KNN = knn_points(adv_xyz.unsqueeze(0), ori_xyz.unsqueeze(0), K=1)
        ori_KNN = knn_points(ori_xyz.unsqueeze(0), adv_xyz.unsqueeze(0), K=1)

        # 计算几何距离的 Chamfer Distance
        min_dist_adv_to_ori = adv_KNN.dists.squeeze(0).min(dim=1)[0]
        min_dist_ori_to_adv = ori_KNN.dists.squeeze(0).min(dim=1)[0]
        chamfer_loss = torch.mean(min_dist_adv_to_ori) + torch.mean(min_dist_ori_to_adv)

        # 计算强度差异的损失
        adv_intensity = adv_pc[:, 3]
        ori_intensity = ori_pc[:, 3]

        # 使用最近邻点对齐强度
        adv_indices = adv_KNN.idx.squeeze(0).squeeze(-1).to(device).long()
        ori_indices = ori_KNN.idx.squeeze(0).squeeze(-1).to(device).long()

        # 方向 1：从 ori_pc 中取出 adv_pc 对应的强度
        ori_intensity_aligned = ori_intensity[adv_indices]
        adv_intensity_diff_1 = torch.mean((adv_intensity - ori_intensity_aligned) ** 2)

        # 方向 2：从 adv_pc 中取出 ori_pc 对应的强度
        adv_intensity_aligned = adv_intensity[ori_indices]
        adv_intensity_diff_2 = torch.mean((ori_intensity - adv_intensity_aligned) ** 2)

        # 强度损失取平均值
        intensity_loss = (adv_intensity_diff_1 + adv_intensity_diff_2) / 2
        # 结合几何距离和强度差异的损失
        total_loss = chamfer_loss * self.loss_weight + intensity_loss * self.intensity_weight
        return total_loss

@LOSSES.register_module()
class AsymmetricHausdorffDistance(nn.Module):
    def __init__(self, loss_weight=1.0):
        """
        非对称 Hausdorff 距离损失，用于度量 P′ 到 P 的最大最近距离。
        
        Args:
            loss_weight (float): 损失的权重。
        """
        super(AsymmetricHausdorffDistance, self).__init__()
        self.loss_weight = loss_weight

    def forward(self, pred, target):
        """
        pred: (N, 4) 去雨后点云，包含(x, y, z, p)的信息
        target: (M, 4) 去雨前点云 GT，包含(x, y, z, p)的信息
        """
        # 只考虑空间坐标 (x, y, z)
        pred = pred[:, :3]
        target = target[:, :3]

        # 计算两组点云之间的欧式距离
        dist = torch.cdist(pred, target, p=2)  # (N, M)

        # 计算 pred 中每个点到 target 的最近距离
        min_dist_pred_to_target = dist.min(dim=1)[0]  # 每个预测点到最近 GT 点的距离

        # 非对称 Hausdorff 距离
        hausdorff_loss = min_dist_pred_to_target.max()  # 最大的最近距离
        return hausdorff_loss * self.loss_weight

@LOSSES.register_module()
class CurvatureLoss(nn.Module):
    def __init__(self, loss_weight=1.0):
        super(CurvatureLoss, self).__init__()
        self.loss_weight = loss_weight

    def forward(self, adv_pc, ori_pc):
        """
        adv_pc: (N, 4) 对抗点云，包含 (x, y, z, t) 信息
        ori_pc: (M, 4) 原始点云，包含 (x, y, z, t) 信息
        """
        # 只取前三个维度 (x, y, z)，忽略时间维度 (t)
        adv_pc = adv_pc[:, :3]  # (N, 3)
        ori_pc = ori_pc[:, :3]  # (M, 3)

        # 计算对抗点云的曲率
        kappa_adv, _ = self._get_kappa_adv(adv_pc, ori_pc, k=2)

        # 计算原始点云的曲率（参考自身）
        kappa_ori, _ = self._get_kappa_adv(ori_pc, ori_pc, k=2)

        # 找到每个对抗点云点在原始点云中的最近邻点
        knn_result = knn_points(adv_pc.unsqueeze(0), ori_pc.unsqueeze(0), K=1)
        _, indices, _ = knn_result.dists, knn_result.idx, knn_result.knn
        kappa_ori_matched = kappa_ori[indices.squeeze(0).squeeze(1)]  # (N,)

        # 计算曲率损失
        curv_loss = ((kappa_adv - kappa_ori_matched) ** 2).mean()

        return curv_loss * self.loss_weight

    def _get_kappa_adv(self, adv_pc, ori_pc, k=10):
        """
        使用 PCA 计算点云曲率。
        adv_pc: (N, 3)
        ori_pc: (M, 3)
        """
        # 增加批次维度以适配 knn_points
        adv_pc = adv_pc.unsqueeze(0)  # (1, N, 3)
        ori_pc = ori_pc.unsqueeze(0)  # (1, M, 3)

        # 计算 adv_pc 和自身的最近邻点
        inter_KNN = knn_points(adv_pc, adv_pc, K=k)
        nn_pts = knn_gather(adv_pc, inter_KNN.idx).squeeze(0)[:, 1:, :]  # (N, k, 3)

        # 对邻域点进行 PCA
        vectors = nn_pts - adv_pc.squeeze(0).unsqueeze(1)  # (N, k, 3)
        cov_matrices = torch.matmul(vectors.transpose(1, 2), vectors) / k  # (N, 3, 3)
        eigvals, _ = torch.linalg.eigh(cov_matrices)  # (N, 3)

        # 曲率为最小特征值与特征值总和的比值
        kappa_adv = eigvals[:, 0] / (eigvals.sum(dim=1) + 1e-6)  # (N,)

        return kappa_adv, None  # 不返回法向量


    # @staticmethod
    # def _normalize(vectors):
    #     """
    #     对向量进行归一化。
    #     vectors: (..., 3) 的张量
    #     返回归一化后的张量，形状与输入相同
    #     """
    #     norm = torch.norm(vectors, dim=-1, keepdim=True)  # (..., 1)
    #     norm = torch.clamp(norm, min=1e-6)  # 防止除以零
    #     return vectors / norm