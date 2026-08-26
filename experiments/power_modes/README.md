# Power-mode runs (SEC 2026 camera-ready, rebuttal RB-Q3)

Vanilla + Albireo (Albireo, default u=3e-4) on the primary detector (yolo26x),
200 BDD clips, at three power modes per board. One folder per mode; results
land inside the mode folder — paper results under results/ are never touched.

## Workflow per mode (reboot required to switch modes)

```bash
sudo nvpmodel -q --verbose        # find the index for the target mode
sudo nvpmodel -m <index>          # set it
sudo reboot
# after reboot:
cd ~/albireo/experiments/power_modes/<board>/<mode>
sudo -E env "PATH=$PATH" ./run.sh ~/bdd100k
```

The script refuses to run if `nvpmodel -q` does not report the folder's mode,
records `power_mode_info.txt` (mode + jetson_clocks provenance), resumes via
summary.csv row count (400 = 200 clips x 2 systems), and parses tegrastats at
the end. ~2-2.5 h per mode.

## Order suggestion
Run modes lowest-to-highest so the board ends the night at its default mode.
Note which mode the original paper runs used — that is the baseline for the
camera-ready one-liner comparison.

## Collect back (from Polaris)
```bash
rsync -avz <board>:~/albireo/experiments/power_modes/<board>/ \
    experiments/power_modes/<board>/
```
