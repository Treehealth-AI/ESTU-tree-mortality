"""Evaluate estu on the shared Testsdata image and label pairs."""
import argparse
import csv
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from PIL import Image

PROJECT = Path(__file__).resolve().parent
DATA = PROJECT.parent / "Testsdata"
INDEX = PROJECT.parent / "test_index"

NAMES = ("background", "live", "partially dead", "dead")
PIXELS = ("background_px", "live_px", "partially_dead_px", "dead_px")


def read_manifest(check_files=False):
    manifest = INDEX / "manifest.csv"
    with manifest.open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError("Empty test manifest")
    for number, row in enumerate(rows, 1):
        if row["id"] != str(number) or row["image"] != f"images/{number}.jpg":
            raise ValueError(f"Unexpected test ID or image at row {number}")
        row["mask"] = f"labels/{number}.png"
        if (row["width"], row["height"]) != ("256", "256") or sum(int(row[k]) for k in PIXELS) != 256**2:
            raise ValueError(f"Invalid recorded label size at row {number}")
        if check_files:
            for field in ("image", "mask"):
                if not (DATA / row[field]).is_file():
                    raise FileNotFoundError(DATA / row[field])
    return rows


def load_model(weight, device):
    state = torch.load(weight, map_location="cpu", weights_only=True)
    # The archived factories contain both a Swin and a convolutional branch.
    # Use the branch recorded in the checkpoint; require every state key to match.
    is_conv = any(k.startswith("backbone.stage1.") for k in state)
    is_swin = any(k.startswith("backbone.patch_embed.") for k in state)
    if is_conv == is_swin:
        raise ValueError(f"Unrecognized checkpoint backbone in {weight}")
    backbone = "simple" if is_conv else "swin_transformer_v2_large"
    from nets.modified_unet import create_sobel_unet as factory
    net = factory(num_classes=4, pretrained=False, backbone=backbone, input_shape=[256, 256])
    if bool(net.use_swin) != is_swin:
        raise RuntimeError("Constructed backbone disagrees with checkpoint")
    architecture = "SwinTransformerV2 Large" if is_swin else "archived convolutional backbone"
    net.load_state_dict(state, strict=True)
    del state
    from utils.callbacks import EvalCallback
    net.to(device).eval()
    callback = EvalCallback(net, [256, 256], 4, [], "", "", device == "cuda", eval_flag=False)
    return callback, architecture


def confusion(callback, row, data):
    with Image.open(data / row["image"]) as im, Image.open(data / row["mask"]) as gt:
        if im.mode != "RGB" or im.size != (256, 256) or gt.mode not in ("L", "P") or gt.size != im.size:
            raise ValueError(f"Invalid saved pair: {row['image']}")
        truth = np.array(gt, dtype=np.int64)
        pred = np.asarray(callback.get_miou_png(im), dtype=np.int64)
    if pred.shape != truth.shape or pred.max() > 3 or truth.max() > 3:
        raise ValueError(f"Invalid class map: {row['image']}")
    hist = np.bincount((4 * truth + pred).ravel(), minlength=16).reshape(4, 4)
    if tuple(hist.sum(axis=1)) != tuple(int(row[k]) for k in PIXELS):
        raise ValueError(f"Mask differs from manifest: {row['mask']}")
    return hist


def measures(hist):
    tp = hist.diagonal().astype(float)
    gt, pred = hist.sum(axis=1), hist.sum(axis=0)
    # Same zero-denominator convention as the archived metrics.
    return np.column_stack((100 * tp / np.maximum(gt + pred - tp, 1),
                            200 * tp / np.maximum(gt + pred, 1),
                            100 * tp / np.maximum(pred, 1),
                            100 * tp / np.maximum(gt, 1)))


def write_results(folder, hist, model_name, total):
    if hist.sum() != total * 256**2:
        raise ValueError("Incomplete evaluation")
    values = measures(hist)
    labels = ("IoU_pct", "F1_pct", "precision_pct", "recall_pct")
    with (folder / "class_metrics.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(("class", *labels, "true_pixels", "predicted_pixels"))
        for c, name in enumerate(NAMES):
            writer.writerow((name, *values[c], int(hist[c].sum()), int(hist[:,c].sum())))
    with (folder / "confusion_matrix.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(("true/predicted", *NAMES))
        for name, row in zip(NAMES, hist):
            writer.writerow((name, *row))
    overall = {
        "Model": model_name,
        "Pairs": total,
        "Overall Accuracy (%)": 100 * hist.trace() / hist.sum(),
        "mPA (%)": values[:, 3].mean(),
        "mIoU (%)": values[:, 0].mean(),
        "Precision (%)": values[:, 2].mean(),
        "Recall (%)": values[:, 3].mean(),
        "F1-score (%)": values[:, 1].mean(),
    }
    with (folder / "overall_metrics.csv").open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(overall))
        writer.writeheader()
        writer.writerow(overall)
    print(f"mIoU = {overall['mIoU (%)']:.3f}%  |  Results: {folder}", flush=True)


def main():
    model_name = 'estu'
    parser = argparse.ArgumentParser()
    parser.add_argument("--check-only", action="store_true", help="Check one real tile per plot on CPU")
    parser.add_argument("--device", choices=("cpu", "cuda"), default=None)
    parser.add_argument("--result-dir", type=Path, help="New model directory for a batch run")
    parser.add_argument("--weights", type=Path, default=PROJECT / "weights" / 'best_epoch_weights.pth')
    args = parser.parse_args()
    device = args.device or ("cpu" if args.check_only or not torch.cuda.is_available() else "cuda")
    torch.set_num_threads(2)
    rows = read_manifest()
    data = DATA
    total = len(rows)
    weight = args.weights
    callback, architecture = load_model(weight, device)
    print(f"{model_name}: {architecture}; {device}; {weight.name}", flush=True)
    if args.check_only:
        plots = sorted({int(r["plot"]) for r in rows})
        rows = [next(r for r in rows if int(r["plot"]) == p) for p in plots]
    hist = np.zeros((4, 4), dtype=np.int64)
    with torch.inference_mode():
        for i, row in enumerate(rows, 1):
            hist += confusion(callback, row, data)
            if i % 500 == 0:
                print(f"Evaluated {i}/{len(rows)}", flush=True)
    if args.check_only:
        print(f"PASS: strict checkpoint load and one real pair from each of {len(plots)} plots.")
        return
    folder = args.result_dir or (PROJECT / "results" /
                                 datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
    folder.mkdir(parents=True, exist_ok=False)
    write_results(folder, hist, model_name, total)


if __name__ == "__main__":
    main()
