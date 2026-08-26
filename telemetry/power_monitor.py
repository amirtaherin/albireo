"""
Script Name: power_monitor.py
Description: Thread-safe power monitor for NVIDIA Jetson AGX Xavier using tegrastats.
             Parses power rail data in background and provides average watts on demand.

Author: Amir Taherin
Email: amirtaherin@gmail.com
Email: taherin.a@northeastern.edu
Date Created: 2026-03-27
Last Modified: 2026-03-27
Version: 1.0

License: MIT License

Usage:
    Imported as a module by run_experiment.py.

        monitor = PowerMonitor()
        monitor.start()
        # ... do work ...
        monitor.stop()
        avg_watts = monitor.get_average_power_watts()  # None if unavailable

Notes:
    - Primary: parses VDD_* power rails from tegrastats stdout
    - Fallback: reads INA3221 voltage/current from sysfs hwmon entries
    - If both fail, get_average_power_watts() returns None (experiment continues)
    - tegrastats must be available at /usr/bin/tegrastats
"""

import re
import subprocess
import threading
import warnings
warnings.filterwarnings("ignore")

# Jetson tegrastats power rail format (requires sudo):
#   Xavier: GPU 0mW/0mW CPU 465mW/465mW SOC 1242mW/1242mW CV 0mW/0mW VDDRQ 465mW/465mW SYS5V 2760mW/2760mW
#   Orin:   VDD_CPU_CV 465mW/465mW VDD_GPU_SOC 3210mW/3210mW VDD_IN 5234mW/5234mW
# Pattern captures RAIL current_mW/avg_mW pairs.
_RAIL_PATTERN = re.compile(r'(\w+)\s+(\d+)mW/(\d+)mW')

# Total input power rail name by platform — tried in order, first match wins.
#   Xavier → SYS5V, Orin → VDD_IN
_TOTAL_RAIL_CANDIDATES = ["SYS5V", "VDD_IN", "VIN_SYS_5V0", "VDD_SYS"]

# Fallback: legacy VDD_* format on some Jetson firmware
_VDD_PATTERN = re.compile(r'VDD_\w+\s+(\d+)/\d+')


class PowerMonitor:
    """
    Background power monitor. Wraps tegrastats (with sudo) in a subprocess
    and reads power data from stdout via a daemon thread.

    Requires passwordless sudo for tegrastats:
        echo "$USER ALL=(ALL) NOPASSWD: /usr/bin/tegrastats" | sudo tee /etc/sudoers.d/tegrastats
        sudo chmod 440 /etc/sudoers.d/tegrastats
    """

    def __init__(
        self,
        tegrastats_path: str = "/usr/bin/tegrastats",
        interval_ms: int = 500,
        use_sudo: bool = True,
    ):
        self.tegrastats_path = tegrastats_path
        self.interval_ms = interval_ms
        self.use_sudo = use_sudo

        self._proc: subprocess.Popen = None
        self._thread: threading.Thread = None
        self._lock = threading.Lock()
        self._samples: list = []          # SYS5V watts per sample
        self._rail_samples: dict = {}     # {rail_name: [watts, ...]} for all rails
        self._running: bool = False
        self._parse_mode: str = "unknown"  # 'tegrastats-sudo', 'tegrastats', 'unavailable'

    def start(self) -> None:
        """Launch tegrastats (optionally with sudo) and start background reader thread."""
        self._samples = []
        self._rail_samples = {}
        self._running = True

        import os
        if not os.path.exists(self.tegrastats_path):
            self._parse_mode = "unavailable"
            self._running = False
            return

        cmd = (["sudo", "-n", self.tegrastats_path] if self.use_sudo
               else [self.tegrastats_path])
        cmd += ["--interval", str(self.interval_ms)]

        try:
            self._proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
                bufsize=1,
            )
        except (FileNotFoundError, PermissionError) as e:
            print(f"[PowerMonitor] Could not start tegrastats: {e}")
            self._parse_mode = "unavailable"
            self._running = False
            return

        self._thread = threading.Thread(target=self._read_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Stop the background reader and terminate tegrastats."""
        self._running = False
        if self._proc is not None:
            try:
                self._proc.terminate()
                self._proc.wait(timeout=3)
            except Exception:
                pass
        if self._thread is not None:
            self._thread.join(timeout=3)

    def get_average_power_watts(self):
        """
        Returns mean SYS5V power in watts (total board power), or None if unavailable.
        Falls back to sum of all identified rails if SYS5V not present.
        """
        with self._lock:
            if not self._samples:
                return None
            return sum(self._samples) / len(self._samples)

    def get_rail_averages(self) -> dict:
        """
        Returns {rail_name: avg_watts} for all power rails seen.
        Empty dict if no data collected.
        """
        with self._lock:
            return {
                rail: sum(vals) / len(vals)
                for rail, vals in self._rail_samples.items()
                if vals
            }

    def get_sample_count(self) -> int:
        """Returns number of power samples collected."""
        with self._lock:
            return len(self._samples)

    def get_parse_mode(self) -> str:
        """Returns how power is being read."""
        return self._parse_mode

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _read_loop(self) -> None:
        """Background thread: read tegrastats lines and parse power."""
        while self._running and self._proc and self._proc.poll() is None:
            try:
                line = self._proc.stdout.readline()
            except Exception:
                break
            if not line:
                break
            self._parse_line(line.strip())

    def _parse_line(self, line: str) -> None:
        """
        Parse one tegrastats line for power rail data.

        Strategy A (sudo tegrastats, AGX Xavier):
          Format: 'GPU 0mW/0mW CPU 465mW/465mW SOC 1242mW/1242mW SYS5V 2760mW/2760mW'
          Extract all RAIL current_mW/avg_mW pairs.
          Use SYS5V as the total; fall back to sum of all rails if absent.

        Strategy B (non-sudo, older firmware):
          Format: 'VDD_CPU_GPU_CV 4000/4000'
          Sum all VDD_* current values.

        If neither matches, mark as unavailable.
        """
        # Strategy A: RAIL XmW/YmW pairs (requires sudo on AGX Xavier)
        rail_matches = _RAIL_PATTERN.findall(line)
        if rail_matches:
            rails = {name: int(current_mw) for name, current_mw, _ in rail_matches}
            with self._lock:
                # Record each rail individually
                for name, mw in rails.items():
                    self._rail_samples.setdefault(name, []).append(mw / 1000.0)
                # Use the first recognised total-input rail (platform-agnostic).
                # Xavier → SYS5V, Orin → VDD_IN. Falls back to sum of all rails.
                total_rail = next(
                    (r for r in _TOTAL_RAIL_CANDIDATES if r in rails), None
                )
                if total_rail:
                    self._samples.append(rails[total_rail] / 1000.0)
                else:
                    self._samples.append(sum(rails.values()) / 1000.0)
                self._parse_mode = "tegrastats-sudo"
            return

        # Strategy B: VDD_* legacy format
        vdd_matches = _VDD_PATTERN.findall(line)
        if vdd_matches:
            total_mw = sum(int(m) for m in vdd_matches)
            with self._lock:
                self._samples.append(total_mw / 1000.0)
                self._parse_mode = "tegrastats"
            return

        # No power data on this line (normal — tegrastats lines without load)
        with self._lock:
            if self._parse_mode == "unknown":
                self._parse_mode = "unavailable"
