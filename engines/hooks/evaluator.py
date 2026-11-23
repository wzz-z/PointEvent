import numpy as np
import torch
import torch.distributed as dist
import pointops

import utils.comm as comm
from utils.misc import intersection_and_union_gpu

from .builder import HOOKS
from .default import HookBase


@HOOKS.register_module()
class SemSegEvaluator(HookBase):
    """Validation hook for EV-UAV binary point-wise detection."""

    def after_epoch(self):
        self.ignore_index = self.trainer.cfg.data.ignore_index
        self.target_class = getattr(self.trainer.cfg, "target_class", 1)
        self.target_thresh = getattr(self.trainer.cfg, "target_thresh", 0.9)
        if self.trainer.cfg.evaluate:
            self.eval()

    def eval(self):
        logger = self.trainer.logger
        logger.info(">>>>>>>>>>>>>>>> Start Evaluation >>>>>>>>>>>>>>>>")
        self.trainer.model.eval()

        num_classes = self.trainer.cfg.data.num_classes
        intersection_sum = np.zeros(num_classes, dtype=np.float64)
        union_sum = np.zeros(num_classes, dtype=np.float64)
        target_sum = np.zeros(num_classes, dtype=np.float64)

        # 新增：按样本统计，跳过无目标且无预测的空样本
        target_iou_valid_sum = 0.0
        target_iou_valid_count = 0

        target_acc_valid_sum = 0.0
        target_acc_valid_count = 0

        for i, input_dict in enumerate(self.trainer.val_loader):
            for key, value in input_dict.items():
                if isinstance(value, torch.Tensor):
                    input_dict[key] = value.cuda(non_blocking=True)

            with torch.no_grad():
                output_dict = self.trainer.model(input_dict)

            logits = output_dict["seg_logits"]
            loss = output_dict["loss"]

            target_prob = torch.softmax(logits, dim=1)[:, self.target_class]
            pred = (target_prob >= self.target_thresh).long()
            segment = input_dict["segment"]

            if "origin_coord" in input_dict:
                idx, _ = pointops.knn_query(
                    1,
                    input_dict["coord"].float(),
                    input_dict["offset"].int(),
                    input_dict["origin_coord"].float(),
                    input_dict["origin_offset"].int(),
                )
                pred = pred[idx.flatten().long()]
                segment = input_dict["origin_segment"]

            intersection, union, target = intersection_and_union_gpu(
                pred,
                segment,
                num_classes,
                self.ignore_index,
            )

            if comm.get_world_size() > 1:
                dist.all_reduce(intersection)
                dist.all_reduce(union)
                dist.all_reduce(target)

            intersection_np = intersection.cpu().numpy()
            union_np = union.cpu().numpy()
            target_np = target.cpu().numpy()

            intersection_sum += intersection_np
            union_sum += union_np
            target_sum += target_np

            inter_fg = intersection_np[self.target_class]
            union_fg = union_np[self.target_class]
            target_fg = target_np[self.target_class]

            # -------------------------------
            # 1. target IoU
            # -------------------------------
            # union_fg > 0 说明：
            #   有 GT 目标，或者有预测目标
            # union_fg == 0 说明：
            #   没有 GT，也没有预测，是纯空样本，不参与样本平均 IoU
            if union_fg > 0:
                target_iou_iter = inter_fg / union_fg
                target_iou_valid_sum += target_iou_iter
                target_iou_valid_count += 1
                target_iou_log = "{:.4f}".format(target_iou_iter)
            else:
                target_iou_iter = None
                target_iou_log = "N/A"

            # -------------------------------
            # 2. target Acc / Recall
            # -------------------------------
            # Acc = TP / GT，只有 GT 目标存在时才有意义
            if target_fg > 0:
                target_acc_iter = inter_fg / target_fg
                target_acc_valid_sum += target_acc_iter
                target_acc_valid_count += 1
                target_acc_log = "{:.4f}".format(target_acc_iter)
            else:
                target_acc_iter = None
                target_acc_log = "N/A"

            self.trainer.storage.put_scalar("val_loss", loss.item())

            logger.info(
                "Test: [{iter}/{max_iter}] Loss {loss:.4f} || "
                "target_Acc {target_acc} || target_IoU {target_iou} || "
                "target_num {target_num:.0f} || union_num {union_num:.0f}".format(
                    iter=i + 1,
                    max_iter=len(self.trainer.val_loader),
                    loss=loss.item(),
                    target_acc=target_acc_log,
                    target_iou=target_iou_log,
                    target_num=target_fg,
                    union_num=union_fg,
                )
            )

        loss_avg = self.trainer.storage.history("val_loss").avg

        # --------------------------------------------------
        # 全局累计指标：推荐作为最终主指标
        # IoU = sum(TP) / sum(TP + FP + FN)
        # --------------------------------------------------
        iou_class = intersection_sum / (union_sum + 1e-10)
        acc_class = intersection_sum / (target_sum + 1e-10)

        target_iou_global = iou_class[self.target_class]
        target_acc_global = acc_class[self.target_class]

        # --------------------------------------------------
        # 样本平均指标：跳过无目标且无预测的空样本
        # --------------------------------------------------
        target_iou_valid_mean = (
            target_iou_valid_sum / target_iou_valid_count
            if target_iou_valid_count > 0
            else 0.0
        )

        target_acc_valid_mean = (
            target_acc_valid_sum / target_acc_valid_count
            if target_acc_valid_count > 0
            else 0.0
        )

        target_name = self.trainer.cfg.data.names[self.target_class]

        logger.info(
            "Val result: global_target_Acc/global_target_IoU {:.4f}/{:.4f}.".format(
                target_acc_global, target_iou_global
            )
        )

        logger.info(
            "Val result: valid_mean_target_Acc/valid_mean_target_IoU {:.4f}/{:.4f}, "
            "valid_acc_count {}, valid_iou_count {}.".format(
                target_acc_valid_mean,
                target_iou_valid_mean,
                target_acc_valid_count,
                target_iou_valid_count,
            )
        )

        logger.info(
            "Class_{idx}-{name} Result: "
            "global accuracy/iou {global_acc:.4f}/{global_iou:.4f}, "
            "valid mean accuracy/iou {mean_acc:.4f}/{mean_iou:.4f}".format(
                idx=self.target_class,
                name=target_name,
                global_acc=target_acc_global,
                global_iou=target_iou_global,
                mean_acc=target_acc_valid_mean,
                mean_iou=target_iou_valid_mean,
            )
        )

        current_epoch = self.trainer.epoch + 1
        if self.trainer.writer is not None:
            self.trainer.writer.add_scalar("val/loss", loss_avg, current_epoch)

            # 全局累计指标
            self.trainer.writer.add_scalar(
                "val/global_target_Acc", target_acc_global, current_epoch
            )
            self.trainer.writer.add_scalar(
                "val/global_target_IoU", target_iou_global, current_epoch
            )

            # 跳过空样本后的样本平均指标
            self.trainer.writer.add_scalar(
                "val/valid_mean_target_Acc", target_acc_valid_mean, current_epoch
            )
            self.trainer.writer.add_scalar(
                "val/valid_mean_target_IoU", target_iou_valid_mean, current_epoch
            )

        logger.info("<<<<<<<<<<<<<<<<< End Evaluation <<<<<<<<<<<<<<<<<")

        # 建议仍然用 global_target_IoU 作为保存 best model 的指标
        # 因为它会惩罚空样本中的误检 FP，更符合检测任务
        self.trainer.comm_info["current_metric_value"] = target_iou_global
        self.trainer.comm_info["current_metric_name"] = "global_target_IoU"