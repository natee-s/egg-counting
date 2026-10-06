"""Export custom-trained YOLOv8 nano checkpoints to fixed-shape OpenVINO.

This tool NEVER downloads stock COCO/ImageNet weights: neither is an egg
segmenter or crack/dirt classifier. Run once, then deploy XML/BIN/metadata offline.
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def export(weights: Path, task: str, size: int, destination: Path,
           int8: bool, calibration: str | None) -> Path:
    from ultralytics import YOLO
    if not weights.is_file() or weights.suffix != ".pt":
        raise FileNotFoundError(f"Need a local trained .pt checkpoint: {weights}")
    if size < 32 or size % 32:
        raise ValueError("Image size must be a positive multiple of 32")
    model = YOLO(str(weights), task=task)
    if model.task != task:
        raise ValueError(f"Expected {task}, found {model.task}")
    labels = {str(v).lower() for v in model.names.values()}
    if task == "segment" and "egg" not in labels:
        raise ValueError("Segmentation weights must have class 'egg'")
    if task == "classify" and (not labels & {"good", "normal"} or not labels & {"defect", "cracked", "crack", "dirty", "dirt"}):
        raise ValueError("Classifier must be trained for good and defective eggs")
    if int8 and not calibration:
        raise ValueError("INT8 requires representative egg calibration data; do not use default datasets")
    args = dict(format="openvino", imgsz=size, batch=1, dynamic=False,
                half=False, int8=int8, device="cpu", nms=False)
    if calibration:
        args["data"] = calibration
    output = Path(model.export(**args))
    if destination.exists():
        raise FileExistsError(f"Destination already exists: {destination}; choose a new output directory")
    shutil.copytree(output, destination)
    return destination


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seg", type=Path, required=True, help="Trained YOLOv8n-seg checkpoint")
    parser.add_argument("--cls", type=Path, required=True, help="Trained YOLOv8n-cls checkpoint")
    parser.add_argument("--seg-size", type=int, default=416)
    parser.add_argument("--cls-size", type=int, default=224)
    parser.add_argument("--seg-out", type=Path, default=ROOT / "models/egg_seg_openvino_model")
    parser.add_argument("--cls-out", type=Path, default=ROOT / "models/egg_cls_openvino_model")
    parser.add_argument("--int8", action="store_true", help="Requires nncf and both calibration datasets")
    parser.add_argument("--seg-data", help="Representative segmentation dataset YAML for INT8")
    parser.add_argument("--cls-data", help="Representative egg classification dataset directory for INT8")
    args = parser.parse_args()
    # Fail before exporting either model if an output could be overwritten.
    for output in (args.seg_out, args.cls_out):
        if output.exists():
            parser.error(f"Output already exists: {output}")
    print(export(args.seg, "segment", args.seg_size, args.seg_out, args.int8, args.seg_data))
    print(export(args.cls, "classify", args.cls_size, args.cls_out, args.int8, args.cls_data))


if __name__ == "__main__":
    main()
