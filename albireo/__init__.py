"""Albireo: adaptive, energy-efficient inference for edge video object detection.

Albireo wraps an off-the-shelf object detector and decides, per frame,
whether detector invocation can be safely skipped based on per-object
Kalman-filter uncertainty, with detector-flicker rescue and an
empty-scene screen. See the SEC 2026 paper for details.
"""
from .detector import AdaptiveTracker, Track
from . import defaults
