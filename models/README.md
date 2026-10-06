# Trained models

The configured real mode uses the user-selected industoai nano segmenter from
`candidates/egg_segmentor_416_openvino_model`, on CPU, with White-Egg and
Brown-Egg class mapping. `EGG_DEFECT_ENABLED=0` disables quality classification;
crossings are marked `unknown / not_inspected`, never good or defective.
Pinned source and SHA-256 manifest: `candidates/egg_segmentor.json`.
Publisher model license: CC BY-NC 4.0, retained after export.

`candidates/kaggle_egg_yolo11n.pt` and its 640 OpenVINO export are staged for
offline comparison only. Source, checksum, and evaluation limits are recorded
in `candidates/kaggle_egg_yolo11n.json` and `../KAGGLE_COMPARISON.md`.

For full defect inspection, real mode requires a second trained classifier.
Stock COCO YOLO
weights do **not** detect eggs, and ImageNet classifiers do **not** classify
egg cracks or dirt. Public trained candidates are now downloaded locally.

- `egg_seg_openvino_model/`: one `.xml`, matching `.bin`, `metadata.yaml`, task `segment`, class `egg`.
- `egg_cls_openvino_model/`: one `.xml`, matching `.bin`, `metadata.yaml`, task `classify`, labels `normal` and `defect`, or `normal`, `cracked`, `dirty`.

Use `backend/export_models.py` to export your trained checkpoints. Keep image
sizes aligned with `backend/.env`. FP32 is the default for accuracy; calibrated
INT8 is optional and must be compared against the held-out egg validation set.
Do not commit proprietary camera footage or trained weights to public repos.
