"""
Checkpoint Utils for Models

Author: Zongze Wu (zongze0907@gmail.com)
Please cite our work if the code is helpful to you.
"""
import os
import sys
import time
import types
from collections import OrderedDict
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.data
from sklearn.metrics import auc, roc_curve

from .defaults import create_ddp_model
from datasets import build_dataset, collate_fn
from models import build_model
from utils.logger import get_root_logger
from utils.misc import AverageMeter, intersection_and_union, make_dirs
from utils.registry import Registry
import utils.comm as comm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from utils.view import * 
TESTERS = Registry("testers")


def install_legacy_pointcept_aliases():
    """Allow checkpoints saved before the flat-layout refactor to be loaded."""
    if "pointcept" in sys.modules:
        return
    import datasets
    import engines
    import models
    import utils

    pointcept = types.ModuleType("pointcept")
    pointcept.datasets = datasets
    pointcept.engines = engines
    pointcept.models = models
    pointcept.utils = utils

    sys.modules["pointcept"] = pointcept
    sys.modules["pointcept.datasets"] = datasets
    sys.modules["pointcept.engines"] = engines
    sys.modules["pointcept.models"] = models
    sys.modules["pointcept.utils"] = utils


class TesterBase:
    def __init__(self, cfg, model=None, test_loader=None, verbose=False) -> None:
        torch.multiprocessing.set_sharing_strategy("file_system")
        self.logger = get_root_logger(
            log_file=os.path.join(cfg.save_path, "test.log"),
            file_mode="a" if cfg.resume else "w",
        )
        self.logger.info("=> Loading config ...")
        self.cfg = cfg
        self.verbose = verbose
        if self.verbose:
            self.logger.info(f"Save path: {cfg.save_path}")
            self.logger.info(f"Config:\n{cfg.pretty_text}")

        if model is None:
            self.logger.info("=> Building model ...")
            self.model = self.build_model()
        else:
            self.model = model

        if test_loader is None:
            self.logger.info("=> Building test dataset & dataloader ...")
            self.test_loader = self.build_test_loader()
        else:
            self.test_loader = test_loader

    def build_model(self):
        model = build_model(self.cfg.model)
        total_parameters = sum(p.numel() for p in model.parameters())
        trainable_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
        self.logger.info(
            "Model params: total={:,} ({:.3f}M), trainable={:,} ({:.3f}M)".format(
                total_parameters,
                total_parameters / 1e6,
                trainable_parameters,
                trainable_parameters / 1e6,
            )
        )
        model = create_ddp_model(
            model.cuda(),
            broadcast_buffers=False,
            find_unused_parameters=self.cfg.find_unused_parameters,
        )
        if not os.path.isfile(self.cfg.weight):
            raise RuntimeError("=> No checkpoint found at '{}'".format(self.cfg.weight))

        self.logger.info(f"Loading weight at: {self.cfg.weight}")
        install_legacy_pointcept_aliases()
        checkpoint = torch.load(self.cfg.weight)
        weight = OrderedDict()
        for key, value in checkpoint["state_dict"].items():
            if key.startswith("module."):
                if comm.get_world_size() == 1:
                    key = key[7:]
            elif comm.get_world_size() > 1:
                key = "module." + key
            weight[key] = value
        model.load_state_dict(weight, strict=True)
        self.logger.info(
            "=> Loaded weight '{}' (epoch {})".format(
                self.cfg.weight, checkpoint["epoch"]
            )
        )
        return model

    def build_test_loader(self):
        test_dataset = build_dataset(self.cfg.data.test)
        if comm.get_world_size() > 1:
            test_sampler = torch.utils.data.distributed.DistributedSampler(test_dataset)
        else:
            test_sampler = None

        return torch.utils.data.DataLoader(
            test_dataset,
            batch_size=self.cfg.batch_size_test_per_gpu,
            shuffle=False,
            num_workers=self.cfg.num_worker_per_gpu,
            pin_memory=True,
            sampler=test_sampler,
            collate_fn=self.__class__.collate_fn,
        )

    def test(self):
        raise NotImplementedError

    @staticmethod
    def collate_fn(batch):
        return collate_fn(batch)


@TESTERS.register_module()
class SemSegTester(TesterBase):
    """Test-time evaluator for EV-UAV binary point-wise detection."""

    @staticmethod
    def collate_fn(batch):
        return batch

    @staticmethod
    def _count_model_params(model):
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        return total_params, trainable_params

    @staticmethod
    def _forward_with_timing(model, input_dict):
        """Return model output and pure forward inference time in seconds."""
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        start_time = time.perf_counter()
        output = model(input_dict)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elapsed_time = time.perf_counter() - start_time
        return output, elapsed_time

    @staticmethod
    def _save_prediction_npz(
        npz_path,
        save_path,
        pred,
        gt,
        pred_scores,
        target_scores,
        npz_key,
        target_class,
        target_thresh,
    ):
        """
        Save original npz content together with model prediction.

        Saved fields:
            - all original fields from npz_path, e.g. evs_norm
            - pred: hard prediction label of each event/point
            - gt: ground-truth label
            - pred_scores: softmax probability of all classes
            - target_scores: softmax probability of target_class
            - target_class / target_thresh: threshold metadata
            - source_npz: original npz path
        """
        with np.load(npz_path) as data:
            save_dict = {key: data[key] for key in data.files}

        save_dict["pred"] = pred.astype(np.int64, copy=False)
        save_dict["gt"] = gt.astype(np.int64, copy=False)
        save_dict["pred_scores"] = pred_scores.astype(np.float32, copy=False)
        save_dict["target_scores"] = target_scores.astype(np.float32, copy=False)
        save_dict["target_class"] = np.array(target_class, dtype=np.int64)
        save_dict["target_thresh"] = np.array(target_thresh, dtype=np.float32)
        save_dict["npz_key"] = np.array(str(npz_key))
        save_dict["source_npz"] = np.array(str(npz_path))

        np.savez_compressed(save_path, **save_dict)

    def test(self):
        assert self.test_loader.batch_size == 1
        logger = get_root_logger()
        logger.info(">>>>>>>>>>>>>>>> Start Evaluation >>>>>>>>>>>>>>>>")

        self.model.eval()

        total_params, trainable_params = self._count_model_params(self.model)
        if comm.is_main_process():
            logger.info(
                "Model params: total={:,} ({:.3f}M), trainable={:,} ({:.3f}M)".format(
                    total_params,
                    total_params / 1e6,
                    trainable_params,
                    trainable_params / 1e6,
                )
            )

        txt_save_path = os.path.join(self.cfg.save_path, "txt")
        pic_save_path = os.path.join(self.cfg.save_path, "pic")
        pred_npz_save_path = os.path.join(self.cfg.save_path, "npz")

        make_dirs(txt_save_path)
        make_dirs(pic_save_path)
        make_dirs(pred_npz_save_path)

        test_data_cfg = self.cfg.data.test

        npz_root = Path(test_data_cfg.data_root) / test_data_cfg.split
        npz_key = getattr(self.cfg, "test_npz_key", "evs_norm")

        vis_height, vis_width = self.cfg.size
        vis_dpi = getattr(self.cfg, "test_vis_dpi", 200)

        test_save_txt = getattr(self.cfg, "test_save_txt", True)
        test_save_pic = getattr(self.cfg, "test_save_pic", True)
        test_save_npz = getattr(self.cfg, "test_save_npz", True)

        target_class = getattr(self.cfg, "target_class", 1)
        target_thresh = getattr(self.cfg, "target_thresh", 0.9)
        num_classes = self.cfg.data.num_classes
        ignore_index = self.cfg.data.ignore_index

        batch_time = AverageMeter()
        inference_time = AverageMeter()
        inference_time_sum = 0.0
        inference_time_count = 0
        infer_forward_count = 0
        infer_warmup = int(getattr(self.cfg, "test_infer_warmup", 1))
        intersection_meter = AverageMeter()
        union_meter = AverageMeter()
        target_meter = AverageMeter()
        record = {}

        for idx, data_dict in enumerate(self.test_loader):
            end = time.time()
            data_dict = data_dict[0]
            fragment_list = data_dict.pop("fragment_list")
            segment = data_dict.pop("segment")
            data_name = data_dict.pop("name")
            if isinstance(segment, torch.Tensor):
                segment = segment.cpu().numpy()

            num_points = segment.shape[0]
            pred_scores = torch.zeros((num_points, num_classes), device="cuda")
            count = torch.zeros((num_points, 1), device="cuda")

            for batch_idx, fragment in enumerate(fragment_list):
                input_dict = collate_fn([fragment])
                for key, value in input_dict.items():
                    if isinstance(value, torch.Tensor):
                        input_dict[key] = value.cuda(non_blocking=True)

                with torch.no_grad():
                    output, infer_elapsed = self._forward_with_timing(self.model, input_dict)
                    logits = output["seg_logits"]
                    scores = F.softmax(logits, dim=-1)

                infer_forward_count += 1
                if infer_forward_count > infer_warmup:
                    inference_time.update(infer_elapsed)
                    inference_time_sum += infer_elapsed
                    inference_time_count += 1
                    logger.info("Infer {:.3f} ms".format(inference_time.val * 1000.0))

                idx_part = input_dict["index"]
                start = 0
                for end_offset in input_dict["offset"]:
                    end_offset = int(end_offset.item())
                    pred_scores[idx_part[start:end_offset], :] += scores[start:end_offset]
                    count[idx_part[start:end_offset], :] += 1
                    start = end_offset


            pred_scores = (pred_scores / count.clamp_min(1)).cpu().numpy()
            if "origin_segment" in data_dict:
                inverse = data_dict["inverse"]
                if isinstance(inverse, torch.Tensor):
                    inverse = inverse.cpu().numpy()
                pred_scores = pred_scores[inverse]
                segment = data_dict["origin_segment"]
                if isinstance(segment, torch.Tensor):
                    segment = segment.cpu().numpy()

            target_scores = pred_scores[:, target_class]
            pred = np.zeros(segment.shape, dtype=np.int64)
            pred[target_scores >= target_thresh] = target_class
            gt = segment.astype(np.int64, copy=False)

            data_name_raw = str(data_name)
            safe_name = data_name_raw.replace("+", "_")

            # 兼容这些情况：
            # 1. 004_label_opint_000000
            # 2. test_004_label_opint_000000
            # 3. test+004_label_opint_000000
            candidate_names = []

            candidate_names.append(data_name_raw)
            candidate_names.append(safe_name)

            if data_name_raw.startswith("test+"):
                candidate_names.append(data_name_raw[len("test+"):])

            if data_name_raw.startswith("test_"):
                candidate_names.append(data_name_raw[len("test_"):])

            if safe_name.startswith("test_"):
                candidate_names.append(safe_name[len("test_"):])

            # 去重，保持顺序
            candidate_names = list(dict.fromkeys(candidate_names))

            npz_path = None
            for name in candidate_names:
                candidate_path = npz_root / f"{name}.npz"
                if candidate_path.exists():
                    npz_path = candidate_path
                    break

            if npz_path is None:
                checked_paths = [str(npz_root / f"{name}.npz") for name in candidate_names]
                raise FileNotFoundError(
                    "找不到原始 npz 文件，已尝试:\n" + "\n".join(checked_paths)
                )

            # 输出文件名也建议去掉 test_ 前缀，和原 npz 对齐
            out_name = npz_path.stem

            if test_save_npz:
                pred_npz_path = os.path.join(
                    pred_npz_save_path,
                    "{}.npz".format(out_name)
                )
                self._save_prediction_npz(
                    npz_path=npz_path,
                    save_path=pred_npz_path,
                    pred=pred,
                    gt=gt,
                    pred_scores=pred_scores,
                    target_scores=target_scores,
                    npz_key=npz_key,
                    target_class=target_class,
                    target_thresh=target_thresh,
                )

            event_array = None

            if test_save_txt:
                txt_path = os.path.join(
                    txt_save_path,
                    "{}.txt".format(out_name)
                )

                event_array = save_event_txt_from_npz(
                    npz_path=npz_path,
                    pred=pred,
                    txt_path=txt_path,
                    npz_key=npz_key
                )

            if test_save_pic:
                if event_array is None:
                    event_array = build_event_array_from_npz(
                        npz_path=npz_path,
                        pred=pred,
                        npz_key=npz_key
                    )

                pic_path = os.path.join(
                    pic_save_path,
                    "{}_vis.png".format(out_name)
                )

                save_event_visualization(
                    event_array=event_array,
                    pic_path=pic_path,
                    width=vis_width,
                    height=vis_height,
                    dpi=vis_dpi
                )

            intersection, union, target = intersection_and_union(
                pred, gt, num_classes, ignore_index
            )
            intersection_meter.update(intersection)
            union_meter.update(union)
            target_meter.update(target)

            iou_class = intersection / (union + 1e-10)
            acc_class = intersection / (target + 1e-10)
            target_iou = iou_class[target_class]
            target_acc = acc_class[target_class]
            sample_auc = self._safe_auc(gt, target_scores, target_class)

            record[data_name] = dict(
                intersection=intersection,
                union=union,
                target=target,
                auc=sample_auc,
            )

            batch_time.update(time.time() - end)
            logger.info(
                "Test: [{idx}/{total}]-{name} "
                "Batch {batch_time.val:.3f} ({batch_time.avg:.3f}) "
                "target_Acc {target_acc:.4f} || target_IoU {target_iou:.4f} || "
                "AUC {auc:.4f}".format(
                    idx=idx + 1,
                    total=len(self.test_loader),
                    name=data_name,
                    batch_time=batch_time,
                    target_acc=target_acc,
                    target_iou=target_iou,
                    auc=sample_auc,
                )
            )

        logger.info("Syncing ...")
        comm.synchronize()
        record_sync = comm.gather(record, dst=0)
        time_sync = comm.gather(
            dict(
                inference_time_sum=inference_time_sum,
                inference_time_count=inference_time_count,
            ),
            dst=0,
        )

        if not comm.is_main_process():
            return

        merged_record = {}
        for worker_record in record_sync:
            merged_record.update(worker_record)

        intersection = np.sum(
            [meters["intersection"] for meters in merged_record.values()], axis=0
        )
        union = np.sum([meters["union"] for meters in merged_record.values()], axis=0)
        target = np.sum([meters["target"] for meters in merged_record.values()], axis=0)
        auc_values = np.array([meters["auc"] for meters in merged_record.values()])
        mean_auc = np.nanmean(auc_values) if not np.all(np.isnan(auc_values)) else np.nan

        iou_class = intersection / (union + 1e-10)
        acc_class = intersection / (target + 1e-10)
        target_iou = iou_class[target_class]
        target_acc = acc_class[target_class]
        target_name = self.cfg.data.names[target_class]

        logger.info(
            "Test result: target_Acc/target_IoU {:.4f}/{:.4f}".format(
                target_acc, target_iou
            )
        )
        logger.info(
            "Class_{idx} - {name} Result: accuracy/iou {accuracy:.4f}/{iou:.4f}".format(
                idx=target_class,
                name=target_name,
                accuracy=target_acc,
                iou=target_iou,
            )
        )
        logger.info("AUC:{:.4f}".format(mean_auc))

        total_infer_time = sum(item["inference_time_sum"] for item in time_sync)
        total_infer_count = sum(item["inference_time_count"] for item in time_sync)
        avg_infer_time = total_infer_time / max(total_infer_count, 1)
        logger.info("Infer {:.3f} ms".format(avg_infer_time * 1000.0))
        logger.info("<<<<<<<<<<<<<<<<< End Evaluation <<<<<<<<<<<<<<<<<")

    @staticmethod
    def _safe_auc(gt, target_scores, target_class):
        binary_gt = (gt == target_class).astype(np.int64)
        if np.unique(binary_gt).size < 2:
            return np.nan
        fpr, tpr, _ = roc_curve(binary_gt, target_scores)
        return auc(fpr, tpr)
