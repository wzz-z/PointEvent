import os
import sys
import time
import types
from pathlib import Path
from collections import OrderedDict

import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.data

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def normalize_polarity(p):
    """
    兼容两种极性格式：
    1. p in {-1, 1}
    2. p in {0, 1}
    """
    unique_p = np.unique(p)

    if np.all(np.isin(unique_p, [0, 1])):
        pos_mask = p == 1
        neg_mask = p == 0
    else:
        pos_mask = p > 0
        neg_mask = p < 0

    return pos_mask, neg_mask


def convert_normalized_xy_to_pixel(events, width=1280, height=720):
    """
    将归一化 x, y 恢复为像素坐标。
    """
    events = events.copy()

    x_norm = events[:, 0]
    y_norm = events[:, 1]

    x_pixel = np.round(x_norm * (width - 1)).astype(np.int64)
    y_pixel = np.round(y_norm * (height - 1)).astype(np.int64)

    x_pixel = np.clip(x_pixel, 0, width - 1)
    y_pixel = np.clip(y_pixel, 0, height - 1)

    events[:, 0] = x_pixel
    events[:, 1] = y_pixel

    return events


def events_to_image(events, width=1280, height=720):
    """
    将事件沿时间维度累积成图像。
    正极性：红色
    负极性：蓝色
    背景：白色
    """
    img = np.ones((height, width, 3), dtype=np.float32)

    if events.shape[0] == 0:
        return img

    x = events[:, 0].astype(np.int64)
    y = events[:, 1].astype(np.int64)
    p = events[:, 3]

    valid = (x >= 0) & (x < width) & (y >= 0) & (y < height)

    x = x[valid]
    y = y[valid]
    p = p[valid]

    if x.shape[0] == 0:
        return img

    pos_mask, neg_mask = normalize_polarity(p)

    pos_count = np.zeros((height, width), dtype=np.float32)
    neg_count = np.zeros((height, width), dtype=np.float32)

    np.add.at(pos_count, (y[pos_mask], x[pos_mask]), 1)
    np.add.at(neg_count, (y[neg_mask], x[neg_mask]), 1)

    pos_pixel = pos_count > neg_count
    neg_pixel = neg_count > pos_count
    both_pixel = (pos_count > 0) & (neg_count > 0) & (pos_count == neg_count)

    img[pos_pixel] = [1.0, 0.0, 0.0]
    img[neg_pixel] = [0.0, 0.0, 1.0]
    img[both_pixel] = [0.6, 0.0, 0.6]

    return img


def _find_npz_key(npz_data, candidates):
    """
    在 npz 中查找候选 key。
    """
    keys = list(npz_data.files)
    lower_map = {k.lower(): k for k in keys}

    for cand in candidates:
        if cand in keys:
            return cand
        if cand.lower() in lower_map:
            return lower_map[cand.lower()]

    return None


def _get_struct_field(arr, candidates):
    """
    从 structured array 中读取字段。
    """
    if arr.dtype.names is None:
        return None

    names = list(arr.dtype.names)
    lower_map = {n.lower(): n for n in names}

    for cand in candidates:
        if cand in names:
            return arr[cand]
        if cand.lower() in lower_map:
            return arr[lower_map[cand.lower()]]

    return None


def _get_array_from_npz(npz_data, candidates):
    """
    从 npz 的独立数组 key 中读取字段。
    """
    key = _find_npz_key(npz_data, candidates)
    if key is None:
        return None
    return npz_data[key]


def _guess_xy_is_normalized(x, y, width=1280, height=720):
    """
    判断 x,y 是否已经是归一化坐标。
    如果 x,y 基本都在 [0,1]，认为是归一化坐标。
    """
    if len(x) == 0:
        return True

    x_max = np.nanmax(x)
    y_max = np.nanmax(y)
    x_min = np.nanmin(x)
    y_min = np.nanmin(y)

    return x_min >= -1e-6 and y_min >= -1e-6 and x_max <= 1.0 + 1e-6 and y_max <= 1.0 + 1e-6


def build_event_array_from_npz(
    npz_path,
    pred,
    npz_key="evs_norm",
):
    """
    兼容不同 npz 事件格式，最终统一输出:

        x, y, t, p, label, predict

    支持格式：
    1. events 是二维数组:
        shape = (N, >=5)
        columns = x,y,t,p,label,...

    2. events 是 structured array:
        shape = (N,)
        dtype.names 包含 x,y,t,p,label 等字段

    3. npz 中 x/y/t/p/label 是独立 key:
        x.npy, y.npy, t.npy, p.npy, label.npy
    """
    npz_path = Path(npz_path)
    npz_data = np.load(npz_path, allow_pickle=True)

    # 兼容配置里的 npz_key 写错或不同数据集 key 不同的情况
    real_key = None
    if npz_key in npz_data.files:
        real_key = npz_key
    else:
        real_key = _find_npz_key(
            npz_data,
            [
                npz_key,
                "evs_norm",
                "ev",
                "events",
                "event",
                "data",
            ],
        )

    if real_key is not None:
        events = npz_data[real_key]
    else:
        events = None

    pred = np.asarray(pred).reshape(-1).astype(np.int32)

    # ==================================================
    # 情况 1：events 是二维普通数组，例如 evs_norm: (N, 5)
    # ==================================================
    if events is not None and events.ndim == 2:
        if events.shape[1] < 4:
            raise ValueError(
                f"{npz_path.name} 的 {real_key} 列数不足，"
                f"至少需要 x,y,t,p，当前 shape={events.shape}"
            )

        x = events[:, 0]
        y = events[:, 1]
        t = events[:, 2]
        p = events[:, 3]

        if events.shape[1] >= 5:
            label = events[:, 4]
        else:
            label = _get_array_from_npz(
                npz_data,
                ["label", "labels", "segment", "seg", "gt", "target"],
            )
            if label is None:
                label = np.zeros(events.shape[0], dtype=np.int32)

    # ==================================================
    # 情况 2：events 是 structured array，例如 ev: (N,)
    # ==================================================
    elif events is not None and events.ndim == 1 and events.dtype.names is not None:
        x = _get_struct_field(events, ["x", "xs", "coord_x", "col"])
        y = _get_struct_field(events, ["y", "ys", "coord_y", "row"])
        t = _get_struct_field(events, ["t", "ts", "time", "timestamp"])
        p = _get_struct_field(events, ["p", "pol", "polarity"])

        label = _get_struct_field(
            events,
            ["label", "labels", "segment", "seg", "gt", "target"],
        )

        if x is None or y is None or t is None or p is None:
            raise ValueError(
                f"{npz_path.name} 的 {real_key} 是 structured array，"
                f"但字段不完整，当前字段={events.dtype.names}，"
                f"至少需要 x,y,t,p"
            )

        if label is None:
            label = _get_array_from_npz(
                npz_data,
                ["label", "labels", "segment", "seg", "gt", "target"],
            )

        if label is None:
            # 如果测试集没有标签，这里先填 0，保证 txt/pic 可以生成
            label = np.zeros(events.shape[0], dtype=np.int32)

    # ==================================================
    # 情况 3：events 是一维 object，每个元素可能是一个事件 tuple/list
    # ==================================================
    elif events is not None and events.ndim == 1 and events.dtype == object:
        try:
            events_2d = np.stack(events, axis=0)
        except Exception as e:
            raise ValueError(
                f"{npz_path.name} 的 {real_key} 是 object 一维数组，"
                f"但无法 stack 成二维数组，shape={events.shape}, error={e}"
            )

        if events_2d.ndim != 2 or events_2d.shape[1] < 4:
            raise ValueError(
                f"{npz_path.name} 的 {real_key} object 转换后格式错误，"
                f"shape={events_2d.shape}"
            )

        x = events_2d[:, 0]
        y = events_2d[:, 1]
        t = events_2d[:, 2]
        p = events_2d[:, 3]

        if events_2d.shape[1] >= 5:
            label = events_2d[:, 4]
        else:
            label = np.zeros(events_2d.shape[0], dtype=np.int32)

    # ==================================================
    # 情况 4：npz 里没有统一 events，而是 x/y/t/p 独立保存
    # ==================================================
    else:
        x = _get_array_from_npz(npz_data, ["x", "xs", "coord_x", "col"])
        y = _get_array_from_npz(npz_data, ["y", "ys", "coord_y", "row"])
        t = _get_array_from_npz(npz_data, ["t", "ts", "time", "timestamp"])
        p = _get_array_from_npz(npz_data, ["p", "pol", "polarity"])

        label = _get_array_from_npz(
            npz_data,
            ["label", "labels", "segment", "seg", "gt", "target"],
        )

        if x is None or y is None or t is None or p is None:
            raise ValueError(
                f"{npz_path.name} 无法解析事件格式。"
                f"当前 npz keys={npz_data.files}，"
                f"配置 npz_key={npz_key}"
            )

        if label is None:
            label = np.zeros(len(x), dtype=np.int32)

    x = np.asarray(x).reshape(-1)
    y = np.asarray(y).reshape(-1)
    t = np.asarray(t).reshape(-1)
    p = np.asarray(p).reshape(-1)
    label = np.asarray(label).reshape(-1)

    n = len(x)

    if not (len(y) == len(t) == len(p) == len(label) == n):
        raise ValueError(
            f"{npz_path.name} 中 x/y/t/p/label 长度不一致: "
            f"x={len(x)}, y={len(y)}, t={len(t)}, p={len(p)}, label={len(label)}"
        )

    if len(pred) != n:
        raise ValueError(
            f"{npz_path.name} 长度不匹配: events={n}, pred={len(pred)}"
        )

    event_array = np.column_stack([
        x.astype(np.float32),
        y.astype(np.float32),
        t.astype(np.float32),
        p.astype(np.int32),
        label.astype(np.int32),
        pred.astype(np.int32),
    ])

    return event_array


def save_event_txt_from_npz(npz_path, pred, txt_path, npz_key="evs_norm"):
    """
    保存 x,y,t,p,label,predict 的 txt。
    """
    event_array = build_event_array_from_npz(
        npz_path=npz_path,
        pred=pred,
        npz_key=npz_key
    )

    np.savetxt(
        txt_path,
        event_array,
        fmt=["%.8f", "%.8f", "%.8f", "%d", "%d", "%d"],
        delimiter=",",
        header="x,y,t,p,label,predict",
        comments=""
    )

    return event_array


def save_event_visualization(
    event_array,
    pic_path,
    width=1280,
    height=720,
    dpi=200
):
    """
    根据 x,y,t,p,label,predict 直接生成三联图。
    """
    x = event_array[:, 0]
    y = event_array[:, 1]

    if _guess_xy_is_normalized(x, y, width=width, height=height):
        data = convert_normalized_xy_to_pixel(
            event_array,
            width=width,
            height=height
        )
    else:
        data = event_array.copy()
        data[:, 0] = np.clip(np.round(data[:, 0]), 0, width - 1)
        data[:, 1] = np.clip(np.round(data[:, 1]), 0, height - 1)

    all_events = data
    label_events = data[data[:, 4].astype(np.int32) == 1]
    pred_events = data[data[:, 5].astype(np.int32) == 1]

    img_all = events_to_image(all_events, width, height)
    img_label = events_to_image(label_events, width, height)
    img_pred = events_to_image(pred_events, width, height)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6), dpi=dpi)

    titles = [
        "All Events",
        "Label = 1",
        "Predict = 1"
    ]

    images = [
        img_all,
        img_label,
        img_pred
    ]

    for ax, img, title in zip(axes, images, titles):
        ax.imshow(img)
        ax.set_title(title, fontsize=14)
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(pic_path, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)