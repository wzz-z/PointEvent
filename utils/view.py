from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def normalize_polarity(p):
    unique_p = np.unique(p)
    if np.all(np.isin(unique_p, [0, 1])):
        return p == 1, p == 0
    return p > 0, p < 0


def convert_normalized_xy_to_pixel(events, width=1280, height=720):
    events = events.copy()
    x_pixel = np.round(events[:, 0] * (width - 1)).astype(np.int64)
    y_pixel = np.round(events[:, 1] * (height - 1)).astype(np.int64)
    events[:, 0] = np.clip(x_pixel, 0, width - 1)
    events[:, 1] = np.clip(y_pixel, 0, height - 1)
    return events


def events_to_image(events, width=1280, height=720):
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
    keys = list(npz_data.files)
    lower_map = {key.lower(): key for key in keys}
    for candidate in candidates:
        if candidate in keys:
            return candidate
        if candidate.lower() in lower_map:
            return lower_map[candidate.lower()]
    return None


def _get_struct_field(arr, candidates):
    if arr.dtype.names is None:
        return None
    lower_map = {name.lower(): name for name in arr.dtype.names}
    for candidate in candidates:
        if candidate in arr.dtype.names:
            return arr[candidate]
        if candidate.lower() in lower_map:
            return arr[lower_map[candidate.lower()]]
    return None


def _get_array_from_npz(npz_data, candidates):
    key = _find_npz_key(npz_data, candidates)
    return None if key is None else npz_data[key]


def build_event_array_from_npz(npz_path, pred, npz_key="evs_norm"):
    npz_path = Path(npz_path)
    with np.load(npz_path, allow_pickle=True) as npz_data:
        real_key = _find_npz_key(
            npz_data,
            [npz_key, "evs_norm", "ev", "events", "event", "data"],
        )
        events = npz_data[real_key] if real_key is not None else None

        if events is not None and events.ndim == 2:
            if events.shape[1] < 4:
                raise ValueError(f"{npz_path.name}: event array needs at least 4 columns.")
            x, y, t, p = events[:, 0], events[:, 1], events[:, 2], events[:, 3]
            label = events[:, 4] if events.shape[1] >= 5 else None
        elif events is not None and events.ndim == 1 and events.dtype.names is not None:
            x = _get_struct_field(events, ["x", "xs", "coord_x", "col"])
            y = _get_struct_field(events, ["y", "ys", "coord_y", "row"])
            t = _get_struct_field(events, ["t", "ts", "time", "timestamp"])
            p = _get_struct_field(events, ["p", "pol", "polarity"])
            label = _get_struct_field(events, ["label", "labels", "segment", "seg", "gt", "target"])
            if x is None or y is None or t is None or p is None:
                raise ValueError(f"{npz_path.name}: structured event array is missing x/y/t/p fields.")
        elif events is not None and events.ndim == 1 and events.dtype == object:
            events_2d = np.stack(events, axis=0)
            if events_2d.ndim != 2 or events_2d.shape[1] < 4:
                raise ValueError(f"{npz_path.name}: object event array has an invalid shape.")
            x, y, t, p = events_2d[:, 0], events_2d[:, 1], events_2d[:, 2], events_2d[:, 3]
            label = events_2d[:, 4] if events_2d.shape[1] >= 5 else None
        else:
            x = _get_array_from_npz(npz_data, ["x", "xs", "coord_x", "col"])
            y = _get_array_from_npz(npz_data, ["y", "ys", "coord_y", "row"])
            t = _get_array_from_npz(npz_data, ["t", "ts", "time", "timestamp"])
            p = _get_array_from_npz(npz_data, ["p", "pol", "polarity"])
            label = _get_array_from_npz(npz_data, ["label", "labels", "segment", "seg", "gt", "target"])
            if x is None or y is None or t is None or p is None:
                raise ValueError(f"{npz_path.name}: could not parse event fields from npz.")

        if label is None:
            label = np.zeros(len(x), dtype=np.int32)

    x = np.asarray(x).reshape(-1)
    y = np.asarray(y).reshape(-1)
    t = np.asarray(t).reshape(-1)
    p = np.asarray(p).reshape(-1)
    label = np.asarray(label).reshape(-1)
    pred = np.asarray(pred).reshape(-1).astype(np.int32)

    if not (len(y) == len(t) == len(p) == len(label) == len(x)):
        raise ValueError(f"{npz_path.name}: event fields have inconsistent lengths.")
    if len(pred) != len(x):
        raise ValueError(f"{npz_path.name}: prediction length does not match events.")

    return np.column_stack(
        [
            x.astype(np.float32),
            y.astype(np.float32),
            t.astype(np.float32),
            p.astype(np.int32),
            label.astype(np.int32),
            pred,
        ]
    )


def save_event_txt_from_npz(npz_path, pred, txt_path, npz_key="evs_norm"):
    event_array = build_event_array_from_npz(npz_path=npz_path, pred=pred, npz_key=npz_key)
    np.savetxt(
        txt_path,
        event_array,
        fmt=["%.8f", "%.8f", "%.8f", "%d", "%d", "%d"],
        delimiter=",",
        header="x,y,t,p,label,predict",
        comments="",
    )
    return event_array


def _guess_xy_is_normalized(x, y):
    if len(x) == 0:
        return True
    return (
        np.nanmin(x) >= -1e-6
        and np.nanmin(y) >= -1e-6
        and np.nanmax(x) <= 1.0 + 1e-6
        and np.nanmax(y) <= 1.0 + 1e-6
    )


def save_event_visualization(event_array, pic_path, width=1280, height=720, dpi=200):
    x = event_array[:, 0]
    y = event_array[:, 1]
    if _guess_xy_is_normalized(x, y):
        data = convert_normalized_xy_to_pixel(event_array, width=width, height=height)
    else:
        data = event_array.copy()
        data[:, 0] = np.clip(np.round(data[:, 0]), 0, width - 1)
        data[:, 1] = np.clip(np.round(data[:, 1]), 0, height - 1)

    images = [
        events_to_image(data, width, height),
        events_to_image(data[data[:, 4].astype(np.int32) == 1], width, height),
        events_to_image(data[data[:, 5].astype(np.int32) == 1], width, height),
    ]
    titles = ["All Events", "Label = 1", "Predict = 1"]

    fig, axes = plt.subplots(1, 3, figsize=(18, 6), dpi=dpi)
    for ax, image, title in zip(axes, images, titles):
        ax.imshow(image)
        ax.set_title(title, fontsize=14)
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(pic_path, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
