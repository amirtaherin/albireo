# Dataset: BDD100K MOT (validation split)

We evaluate on the BDD100K multi-object tracking validation split
(200 dashcam clips, 7 annotated classes). The dataset is licensed by the
BDD100K project and must be downloaded from the official source:
https://doc.bdd100k.com/download.html (images + MOT labels).

Expected layout is documented in `experiments/bdd_loader.py`.
`clips_seed42.txt` lists the exact 200 validation clips selected with
seed 42 — every experiment in the paper runs this same clip set.
