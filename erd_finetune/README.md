# Empty-Road Detector (ERD) fine-tuning

The empty-scene screen uses the lightweight ERD classifier of
Liu and Kang, "Filtering Empty Video Frames for Efficient Real-Time
Object Detection" (Sensors, 2024). Original implementation and weights:
https://github.com/Real-Time-Lab/Filtering-Empty-Video-Frames-for-Efficient-Real-Time-Object-Detection (MIT license).

We fine-tune the upstream ERD on BDD100K MOT training clips to classify
dashcam frames as empty / non-empty:

- `generate_erd_labels.py` — produces empty/non-empty labels for BDD100K
  frames using a YOLO11x teacher at conf 0.50.
- `train_erd_bdd100k.py` — fine-tunes ERD on those labels.
- `weights/erd_bdd100k.pt` — the fine-tuned weights used in the paper.

Credit for the ERD architecture and original weights belongs to Liu and
Kang; the fine-tuned weights are released here under MIT with attribution.
