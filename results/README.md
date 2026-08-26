# Per-clip result sets (paper data)

`summary.csv` files contain the per-clip metrics behind every table and
figure in the paper, organized as `<platform>/<system>/<model>/summary.csv`
(200 BDD100K MOT validation clips, seed 42, detector conf 0.50).

Mapping to the paper:
- Table I (YOLO26x headline, Thor/Orin): `*/albireo/yolo26x` (vs.\ Vanilla rows within the same files)
- Baseline comparison: `*/ctd/*`, `*/statues/*`, `*/fixedskip/*`, `*/erd_only/*`
- Detector-agnostic results: `yolo11x` and `rfdetr-large` subdirectories
- Threshold sweep / Pareto frontier: `*/albireo_sweep/u*/yolo26x`

Column definitions are printed by `figures/plot_results.py`; energy and
power columns are produced by the telemetry pipeline in `telemetry/`.
