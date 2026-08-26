"""Tuned default parameters used in the SEC 2026 paper (primary operating point).

These are the values behind every headline number in the paper; the
uncertainty threshold `U` is the primary accuracy--efficiency knob.
"""
U = 3e-4                # uncertainty threshold on tr(H P H^T); primary knob
T_MAX = 15              # max consecutive prediction-only frames per object
PERIODIC_CAP = 30       # force a detector call at least once every N frames
C_RESCUE = 0.5          # min last-match confidence for rescue eligibility
R_MAX = 2               # max consecutive rescues per object state
YOLO_CONF = 0.5         # detector confidence threshold used throughout
