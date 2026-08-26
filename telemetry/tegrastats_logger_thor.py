# Copyright (c) 2025 Amir Taherin
# Author: Amir Taherin
# Last Modified: 2026-04-07
# __________________________________
import subprocess
import csv
import ctypes
import time
import os
import signal
import threading

"""
Thor-specific tegrastats logger for JetPack 7.1.
Reads tegrastats output line-by-line in Python and appends NVML GPU utilization
(gpu_util%, mem_util%) to each line before writing to file.

This replaces the shell pipeline approach used in tegrastats.py so we can
interleave NVML reads with each tegrastats sample.
"""


class NVMLReader:
    """Lightweight NVML wrapper using ctypes to read GPU utilization."""

    def __init__(self):
        self.nvml = None
        self.handle = None

    def init(self):
        try:
            self.nvml = ctypes.CDLL('libnvidia-ml.so.1')
            self.nvml.nvmlInit_v2()
            self.handle = ctypes.c_void_p()
            self.nvml.nvmlDeviceGetHandleByIndex_v2(0, ctypes.byref(self.handle))
        except Exception as e:
            print(f"Warning: NVML init failed: {e}")
            self.nvml = None

    def get_utilization(self):
        """Returns (gpu_util%, mem_util%) or (None, None) if unavailable."""
        if self.nvml is None:
            return None, None
        try:

            class Utilization(ctypes.Structure):
                _fields_ = [('gpu', ctypes.c_uint), ('memory', ctypes.c_uint)]

            util = Utilization()
            self.nvml.nvmlDeviceGetUtilizationRates(self.handle, ctypes.byref(util))
            return util.gpu, util.memory
        except Exception:
            return None, None

    def shutdown(self):
        if self.nvml:
            try:
                self.nvml.nvmlShutdown()
            except Exception:
                pass


class Tegrastats:
    def __init__(self, log_file="output_log.txt", interval=1, verbose=False):
        self.interval = interval
        self.log_file = log_file
        self.verbose = verbose
        self._stop_event = threading.Event()

    def start_tegrastats(self):
        """Start tegrastats reader in a background thread.
        Returns the thread object (use stop_tegrastats to stop)."""
        self._stop_event.clear()
        t = threading.Thread(target=self._reader_loop, daemon=True)
        t.start()
        print(f"Tegrastats started (thread-based with NVML)")
        return t

    def stop_tegrastats(self, thread):
        """Signal the reader loop to stop and wait for it."""
        self._stop_event.set()
        thread.join(timeout=5)
        if thread.is_alive():
            print("Warning: tegrastats thread did not stop cleanly")
        else:
            print("Tegrastats stopped successfully")

    def _reader_loop(self):
        """Read tegrastats line-by-line, append NVML data and nanosecond timestamp, write to file."""
        nvml = NVMLReader()
        nvml.init()

        # Start tegrastats subprocess
        process = subprocess.Popen(
            ['sudo', 'tegrastats', '--interval', str(self.interval)],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True
        )

        try:
            with open(self.log_file, 'w') as f:
                while not self._stop_event.is_set():
                    line = process.stdout.readline()
                    if not line:
                        break
                    line = line.strip()

                    # Get nanosecond timestamp
                    now = time.time_ns()
                    ts_sec = now // 10**9
                    ts_ns = now % 10**9
                    t = time.localtime(ts_sec)
                    timestamp = f"{t.tm_mon:02d}-{t.tm_mday:02d}-{t.tm_year} {t.tm_hour:02d}:{t.tm_min:02d}:{t.tm_sec:02d}.{ts_ns:09d}"

                    # Get NVML utilization
                    gpu_util, mem_util = nvml.get_utilization()
                    nvml_str = f"GPU_UTIL {gpu_util}% MEM_UTIL {mem_util}%"

                    # Write: timestamp + tegrastats line + NVML data
                    f.write(f"{timestamp} {line} {nvml_str}\n")
                    f.flush()
        finally:
            # Clean up
            nvml.shutdown()
            try:
                process.terminate()
                process.wait(timeout=2)
            except Exception:
                process.kill()


# Defining helper functions
def write_to_csv(output_path, header, values):
    with open(output_path, mode='w') as gtFile:
        writer = csv.writer(gtFile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        writer.writerow(header)
        for value in values:
            writer.writerow(value)


if __name__ == '__main__':
    tegrastats = Tegrastats(interval=100)
    thread = tegrastats.start_tegrastats()
    time.sleep(5)
    tegrastats.stop_tegrastats(thread)

    # Print the output
    with open('output_log.txt') as f:
        for line in f:
            print(line.strip())
