"""
Script Name: clip_tegrastats.py
Description: Per-clip tegrastats logger and parser. Replaces both the global
             TegrastatsLogger and per-clip PowerMonitor with a single mechanism
             that captures the full tegrastats output per clip, parses it into
             structured metrics (power, temperature, utilization, memory), and
             computes accurate energy via trapezoidal integration.

             Works cross-platform: Xavier (JP5), Orin (JP6), Thor (JP7).
             On Thor, augments tegrastats with NVML GPU utilization since
             GR3D_FREQ does not report load% on that platform.

Author: Amir Taherin
Email: amirtaherin@gmail.com
Email: taherin.a@northeastern.edu
Date Created: 2026-04-19
Last Modified: 2026-04-19
Version: 1.0

License: MIT License

Usage:
    from clip_tegrastats import ClipTegrastats

    tgs = ClipTegrastats(log_path="results/thor/albireo/yolo11x/tegrastats/clip_vanilla.log")
    tgs.start()
    # ... process clip frames ...
    tgs.stop()
    summary = tgs.get_summary()
    # summary["energy_j"], summary["avg_power_w"], summary["avg_gpu_temp_c"], ...
"""

import ctypes
import os
import re
import subprocess
import threading
import time
import warnings
warnings.filterwarnings("ignore")


TEGRASTATS_AVAILABLE = os.path.exists("/usr/bin/tegrastats")

# --- NVML for Thor GPU utilization ---

class _NVMLUtilization(ctypes.Structure):
    _fields_ = [("gpu", ctypes.c_uint), ("memory", ctypes.c_uint)]

try:
    _nvml = ctypes.CDLL("libnvidia-ml.so.1")
    _NVML_AVAILABLE = True
except (OSError, AttributeError):
    _nvml = None
    _NVML_AVAILABLE = False

# --- Tegrastats line parsing patterns ---

_RAIL_PATTERN = re.compile(r'(\w+)\s+(\d+)mW/(\d+)mW')
_TOTAL_RAIL_CANDIDATES = ["SYS5V", "VDD_IN", "VIN", "VDD_SYS"]

_RAIL_TO_GROUP = {
    # Xavier
    "GPU": "gpu", "CPU": "cpu",
    "SOC": "io", "CV": "io", "VDDRQ": "io",
    # Orin
    "VDD_GPU_SOC": "gpu", "VDD_CPU_CV": "cpu",
    "VIN_SYS_5V0": "io",
    # Thor
    "VDD_GPU": "gpu", "VDD_CPU_SOC_MSS": "cpu",
}

_TEMP_PATTERN = re.compile(r'(\w+)@([\d.]+)C', re.IGNORECASE)
_GPU_UTIL_PATTERN = re.compile(r'GR3D_FREQ\s+(\d+)%')
_GPU_UTIL_NVML_PATTERN = re.compile(r'GPU_UTIL\s+(\d+)%')
_CPU_PATTERN = re.compile(r'CPU\s+\[([^\]]*)\]')
_CORE_LOAD_PATTERN = re.compile(r'(\d+)%@\d+')
_EMC_PATTERN = re.compile(r'EMC_FREQ\s+(\d+)%')
_RAM_PATTERN = re.compile(r'RAM\s+(\d+)/(\d+)MB')

_POWER_GROUPS = ["gpu", "cpu", "io"]

_SUMMARY_KEYS = [
    "energy_j", "avg_power_w", "max_power_w",
    "avg_power_gpu_w", "energy_gpu_j",
    "avg_power_cpu_w", "energy_cpu_j",
    "avg_power_io_w", "energy_io_j",
    "avg_gpu_temp_c", "max_gpu_temp_c",
    "avg_cpu_temp_c", "max_cpu_temp_c",
    "max_tj_temp_c",
    "avg_gpu_util_pct", "avg_cpu_util_pct", "avg_emc_util_pct",
    "avg_ram_used_mb", "max_ram_used_mb",
    "n_tegrastats_samples",
]


class _Sample:
    __slots__ = (
        'timestamp', 'total_power_w', 'rail_power_w', 'group_power_w',
        'temps_c',
        'gpu_util', 'cpu_util', 'emc_util', 'ram_used_mb', 'ram_total_mb',
    )

    def __init__(self):
        self.timestamp = 0.0
        self.total_power_w = 0.0
        self.rail_power_w = {}
        self.group_power_w = {}
        self.temps_c = {}
        self.gpu_util = None
        self.cpu_util = None
        self.emc_util = None
        self.ram_used_mb = None
        self.ram_total_mb = None


class ClipTegrastats:
    """
    Per-clip tegrastats logger.

    Starts a tegrastats subprocess, collects timestamped lines in a background
    thread, optionally saves raw output to a log file, and on stop() parses all
    collected lines into structured samples for summary statistics.
    """

    def __init__(self, log_path=None, interval_ms=100):
        self.log_path = log_path
        self.interval_ms = interval_ms
        self._proc = None
        self._thread = None
        self._running = False
        self._lines = []
        self._samples = []
        self._log_file = None
        self._nvml_handle = None

    def start(self):
        if not TEGRASTATS_AVAILABLE:
            return

        self._lines = []
        self._samples = []
        self._running = True

        if self.log_path:
            os.makedirs(os.path.dirname(self.log_path), exist_ok=True)
            self._log_file = open(self.log_path, 'w')

        if _NVML_AVAILABLE:
            try:
                _nvml.nvmlInit_v2()
                handle = ctypes.c_void_p()
                _nvml.nvmlDeviceGetHandleByIndex_v2(0, ctypes.byref(handle))
                self._nvml_handle = handle
            except Exception:
                self._nvml_handle = None

        cmd = ["sudo", "-n", "/usr/bin/tegrastats",
               "--interval", str(self.interval_ms)]
        try:
            self._proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
                bufsize=1,
            )
        except (FileNotFoundError, PermissionError):
            self._running = False
            if self._log_file:
                self._log_file.close()
                self._log_file = None
            return

        self._thread = threading.Thread(target=self._read_loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False
        if self._proc is not None:
            try:
                self._proc.terminate()
                self._proc.wait(timeout=3)
            except Exception:
                try:
                    self._proc.kill()
                except Exception:
                    pass
        if self._thread is not None:
            self._thread.join(timeout=3)
        if self._log_file is not None:
            self._log_file.close()
            self._log_file = None
        if self._nvml_handle is not None:
            try:
                _nvml.nvmlShutdown()
            except Exception:
                pass
            self._nvml_handle = None

        self._parse_all()

    def get_summary(self):
        if not self._samples:
            return {k: None for k in _SUMMARY_KEYS}

        # Trapezoidal integration for energy (total + per-group)
        energy_j = 0.0
        group_energy = {g: 0.0 for g in _POWER_GROUPS}
        for i in range(1, len(self._samples)):
            dt = self._samples[i].timestamp - self._samples[i - 1].timestamp
            avg_p = (self._samples[i].total_power_w
                     + self._samples[i - 1].total_power_w) / 2.0
            energy_j += avg_p * dt
            for g in _POWER_GROUPS:
                avg_gp = (self._samples[i].group_power_w.get(g, 0.0)
                          + self._samples[i - 1].group_power_w.get(g, 0.0)) / 2.0
                group_energy[g] += avg_gp * dt

        powers = [s.total_power_w for s in self._samples]
        group_powers = {g: [s.group_power_w.get(g, 0.0) for s in self._samples]
                        for g in _POWER_GROUPS}
        gpu_temps = [s.temps_c['gpu'] for s in self._samples
                     if 'gpu' in s.temps_c]
        cpu_temps = [s.temps_c['cpu'] for s in self._samples
                     if 'cpu' in s.temps_c]
        tj_temps = [s.temps_c['tj'] for s in self._samples
                    if 'tj' in s.temps_c]
        gpu_utils = [s.gpu_util for s in self._samples
                     if s.gpu_util is not None]
        cpu_utils = [s.cpu_util for s in self._samples
                     if s.cpu_util is not None]
        emc_utils = [s.emc_util for s in self._samples
                     if s.emc_util is not None]
        ram_used = [s.ram_used_mb for s in self._samples
                    if s.ram_used_mb is not None]

        def _mean(lst):
            return sum(lst) / len(lst) if lst else None

        def _max(lst):
            return max(lst) if lst else None

        def _rnd(val, n=1):
            return round(val, n) if val is not None else None

        return {
            "energy_j":            _rnd(energy_j, 2),
            "avg_power_w":         _rnd(_mean(powers), 3),
            "max_power_w":         _rnd(_max(powers), 3),
            "avg_power_gpu_w":     _rnd(_mean(group_powers["gpu"]), 3),
            "energy_gpu_j":        _rnd(group_energy["gpu"], 2),
            "avg_power_cpu_w":     _rnd(_mean(group_powers["cpu"]), 3),
            "energy_cpu_j":        _rnd(group_energy["cpu"], 2),
            "avg_power_io_w":      _rnd(_mean(group_powers["io"]), 3),
            "energy_io_j":         _rnd(group_energy["io"], 2),
            "avg_gpu_temp_c":      _rnd(_mean(gpu_temps)),
            "max_gpu_temp_c":      _rnd(_max(gpu_temps)),
            "avg_cpu_temp_c":      _rnd(_mean(cpu_temps)),
            "max_cpu_temp_c":      _rnd(_max(cpu_temps)),
            "max_tj_temp_c":       _rnd(_max(tj_temps)),
            "avg_gpu_util_pct":    _rnd(_mean(gpu_utils)),
            "avg_cpu_util_pct":    _rnd(_mean(cpu_utils)),
            "avg_emc_util_pct":    _rnd(_mean(emc_utils)),
            "avg_ram_used_mb":     _rnd(_mean(ram_used), 0),
            "max_ram_used_mb":     _rnd(_max(ram_used), 0),
            "n_tegrastats_samples": len(self._samples),
        }

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _read_loop(self):
        while self._running and self._proc and self._proc.poll() is None:
            try:
                line = self._proc.stdout.readline()
            except Exception:
                break
            if not line:
                break
            ts_mono = time.monotonic()
            ts_wall_ns = time.time_ns()
            line = line.strip()

            if self._nvml_handle is not None:
                try:
                    rates = _NVMLUtilization()
                    _nvml.nvmlDeviceGetUtilizationRates(
                        self._nvml_handle, ctypes.byref(rates))
                    line += f" GPU_UTIL {rates.gpu}% MEM_UTIL {rates.memory}%"
                except Exception:
                    pass

            self._lines.append((ts_mono, line))
            if self._log_file:
                sec = ts_wall_ns // 1_000_000_000
                ns = ts_wall_ns % 1_000_000_000
                t = time.gmtime(sec)
                ts_str = time.strftime("%m-%d-%Y %H:%M:%S", t) + f".{ns:09d}"
                self._log_file.write(f"{ts_str} {line}\n")
                self._log_file.flush()

    def _parse_all(self):
        self._samples = []
        for ts, line in self._lines:
            sample = self._parse_line(ts, line)
            if sample.total_power_w > 0:
                self._samples.append(sample)

    @staticmethod
    def _parse_line(ts, line):
        sample = _Sample()
        sample.timestamp = ts

        # Power rails
        rails = {}
        for name, cur_mw, _ in _RAIL_PATTERN.findall(line):
            rails[name] = int(cur_mw) / 1000.0
        sample.rail_power_w = rails
        total_rail = next(
            (r for r in _TOTAL_RAIL_CANDIDATES if r in rails), None)
        if total_rail:
            sample.total_power_w = rails[total_rail]
        elif rails:
            sample.total_power_w = sum(rails.values())

        groups = {g: 0.0 for g in _POWER_GROUPS}
        for rail_name, watts in rails.items():
            grp = _RAIL_TO_GROUP.get(rail_name)
            if grp:
                groups[grp] += watts
        sample.group_power_w = groups

        # Temperatures — normalize keys to {cpu, gpu, tj, soc, tboard, tdiode}
        for name, val in _TEMP_PATTERN.findall(line):
            key = name.lower()
            if key in ('tj', 'tboard', 'tdiode'):
                sample.temps_c[key] = float(val)
            elif key.startswith('cpu'):
                sample.temps_c['cpu'] = float(val)
            elif key.startswith('gpu'):
                sample.temps_c['gpu'] = float(val)

        # GPU utilization: GR3D_FREQ X% on Xavier/Orin, GPU_UTIL X% on Thor
        m = _GPU_UTIL_PATTERN.search(line)
        if m:
            sample.gpu_util = int(m.group(1))
        else:
            m = _GPU_UTIL_NVML_PATTERN.search(line)
            if m:
                sample.gpu_util = int(m.group(1))

        # CPU utilization: average of online cores
        m = _CPU_PATTERN.search(line)
        if m:
            core_loads = _CORE_LOAD_PATTERN.findall(m.group(1))
            if core_loads:
                sample.cpu_util = (
                    sum(int(x) for x in core_loads) / len(core_loads))

        # EMC utilization
        m = _EMC_PATTERN.search(line)
        if m:
            sample.emc_util = int(m.group(1))

        # RAM
        m = _RAM_PATTERN.search(line)
        if m:
            sample.ram_used_mb = int(m.group(1))
            sample.ram_total_mb = int(m.group(2))

        return sample
