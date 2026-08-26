# Copyright (c) 2025 Amir Taherin
# Author: Amir Taherin
# Last Modified: 2026-04-05
# __________________________________
import argparse
import csv
import os
import re
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

"""
This script parses the tegrastats log file for Thor (JetPack 7.0) and converts it to a CSV file.
Based on tegrastats_parser.py (Orin / JP6.2).

Thor tegrastats differences from Orin (requires sudo for full output):
  - 14 CPU cores (vs 12)
  - No SWAP field
  - EMC_FREQ format: "EMC_FREQ 33%@4266" (same as Orin)
  - GR3D_FREQ format: "@[freq,freq,freq]" with 3 GPC values, no load% prefix (Orin has "0%@[305,305]")
  - NVENC0_FREQ, NVENC1_FREQ, NVDEC0_FREQ, NVDEC1_FREQ report as "@freq" (Orin has on/off)
  - NVJPG0_FREQ, OFA_FREQ report as "@freq" (Orin has on/off)
  - No NVDLA0, NVDLA1 fields
  - Temperature sensors: cpu@, tj@, soc012@, soc345@, gpu@ (gpu@ only appears under load)
    (Orin has cpu@, soc0@, soc1@, soc2@, gpu@, tj@, Tboard@, Tdiode@)
  - Power rails: VDD_GPU, VDD_CPU_SOC_MSS, VIN_SYS_5V0, VIN in curr/peak mW format
    (Orin has VDD_GPU_SOC, VDD_CPU_CV, VIN_SYS_5V0, VDDQ_VDD2_1V8AO in curr/avg mW format)
"""


def line_parser(line):
    """
    Parse a line of tegrastats output (Thor / JetPack 7.0) and return a dictionary of the values.
    """

    parsed_line = {}

    # DATE and TIME
    # The Hydra wrapper prepends a nanosecond timestamp via date command.
    # Line format: "MM-DD-YYYY HH:MM:SS.NNNNNNNNN MM-DD-YYYY HH:MM:SS RAM ..."
    # Pick the timestamp with fractional seconds (nanosecond precision).
    matches = re.findall(r'\d{2}-\d{2}-\d{4} \d{2}:\d{2}:\d{2}(?:\.\d{1,9})?', line)

    ns_time_str = next((m for m in matches if '.' in m), None)
    if ns_time_str:
        dt, ns = ns_time_str.split('.')
        ns = ns.ljust(9, '0')[:9]
        dt_obj = datetime.strptime(dt, '%m-%d-%Y %H:%M:%S').replace(tzinfo=ZoneInfo("America/New_York"))
        dt_obj = dt_obj.astimezone(timezone.utc)
        timestamp_ns = int(dt_obj.timestamp()) * 10**9 + int(ns)
        parsed_line['timestamp'] = timestamp_ns
    else:
        print("No high-precision timestamp found.")
        exit(1)

    # RAM
    ram = re.findall(r'RAM ([0-9]+)/([0-9]+)MB \(lfb ([0-9]+)x([0-9]+)MB\)', line)
    parsed_line['ram_used'] = float(ram[0][0]) if ram else None
    parsed_line['ram_total'] = float(ram[0][1]) if ram else None
    parsed_line['ram_lfb_count'] = float(ram[0][2]) if ram else None
    parsed_line['ram_lfb_size'] = float(ram[0][3]) if ram else None

    # CPU — Thor has 14 cores
    # Use character class to grab contents up to the closing bracket
    cpus = re.findall(r'CPU \[([^\]]*)\]', line)
    if cpus:
        cpus = cpus[0].split(",")
        for i in range(len(cpus)):
            if cpus[i].strip() == 'off':
                cpus[i] = 'inf%@inf'
            cpu = cpus[i].strip().split('%@')
            parsed_line[f'cpu_{i}_load'] = float(cpu[0]) if len(cpu) == 2 else None
            parsed_line[f'cpu_{i}_freq'] = float(cpu[1]) if len(cpu) == 2 else None

    # EMC_FREQ — memory controller load and frequency
    emc = re.findall(r'EMC_FREQ ([0-9]+)%@?([0-9]*)?', line)
    parsed_line['emc_load'] = float(emc[0][0]) if emc else None
    parsed_line['emc_freq'] = float(emc[0][1]) if emc and emc[0][1] else None

    # GR3D_FREQ — GPU frequency, Thor format: "@[freq,freq,freq]" (3 GPCs)
    # Also handles optional load%: "99%@[freq,freq,freq]" if added in future JetPack
    gpu = re.findall(r'GR3D_FREQ (?:([0-9]+)%)?@\[([0-9]+),([0-9]+),([0-9]+)\]', line)
    parsed_line['gpc1_load'] = float(gpu[0][0]) if gpu and gpu[0][0] else float('nan')
    parsed_line['gpc2_load'] = float(gpu[0][0]) if gpu and gpu[0][0] else float('nan')
    parsed_line['gpc3_load'] = float(gpu[0][0]) if gpu and gpu[0][0] else float('nan')
    parsed_line['gpc1_freq'] = float(gpu[0][1]) if gpu else None
    parsed_line['gpc2_freq'] = float(gpu[0][2]) if gpu else None
    parsed_line['gpc3_freq'] = float(gpu[0][3]) if gpu else None

    # NVENC0_FREQ, NVENC1_FREQ
    nvenc0 = re.findall(r'NVENC0_FREQ @([0-9]+)', line)
    parsed_line['nvenc0_freq'] = float(nvenc0[0]) if nvenc0 else None

    nvenc1 = re.findall(r'NVENC1_FREQ @([0-9]+)', line)
    parsed_line['nvenc1_freq'] = float(nvenc1[0]) if nvenc1 else None

    # NVDEC0_FREQ, NVDEC1_FREQ
    nvdec0 = re.findall(r'NVDEC0_FREQ @([0-9]+)', line)
    parsed_line['nvdec0_freq'] = float(nvdec0[0]) if nvdec0 else None

    nvdec1 = re.findall(r'NVDEC1_FREQ @([0-9]+)', line)
    parsed_line['nvdec1_freq'] = float(nvdec1[0]) if nvdec1 else None

    # NVJPG0_FREQ
    nvjpg0 = re.findall(r'NVJPG0_FREQ @([0-9]+)', line)
    parsed_line['nvjpg0_freq'] = float(nvjpg0[0]) if nvjpg0 else None

    # VIC
    vic = re.findall(r'VIC (off)', line)
    parsed_line['vic'] = 'off' if vic else None

    # OFA_FREQ
    ofa = re.findall(r'OFA_FREQ @([0-9]+)', line)
    parsed_line['ofa_freq'] = float(ofa[0]) if ofa else None

    # PVA0_FREQ
    pva0 = re.findall(r'PVA0_FREQ (off)', line)
    parsed_line['pva0_freq'] = 'off' if pva0 else None

    # APE
    ape = re.findall(r'APE ([0-9]+)', line)
    parsed_line['ape'] = float(ape[0]) if ape else None

    # Temperatures
    cpu_temp = re.findall(r'cpu@([(-|)][0-9.]+)C', line)
    parsed_line['cpu_temp'] = float(cpu_temp[0]) if cpu_temp else None

    tj_temp = re.findall(r'tj@([(-|)][0-9.]+)C', line)
    parsed_line['tj_temp'] = float(tj_temp[0]) if tj_temp else None

    soc012_temp = re.findall(r'soc012@([(-|)][0-9.]+)C', line)
    parsed_line['soc012_temp'] = float(soc012_temp[0]) if soc012_temp else None

    # gpu@ only appears when GPU is under load
    gpu_temp = re.findall(r'gpu@([(-|)][0-9.]+)C', line)
    parsed_line['gpu_temp'] = float(gpu_temp[0]) if gpu_temp else None

    soc345_temp = re.findall(r'soc345@([(-|)][0-9.]+)C', line)
    parsed_line['soc345_temp'] = float(soc345_temp[0]) if soc345_temp else None

    # Power Rails (curr/peak mW format)
    vdd_gpu = re.findall(r'VDD_GPU ([0-9.]+)mW/([0-9.]+)mW', line)
    parsed_line['vdd_gpu_cur'] = float(vdd_gpu[0][0]) if vdd_gpu else None
    parsed_line['vdd_gpu_peak'] = float(vdd_gpu[0][1]) if vdd_gpu else None

    vdd_cpu_soc_mss = re.findall(r'VDD_CPU_SOC_MSS ([0-9.]+)mW/([0-9.]+)mW', line)
    parsed_line['vdd_cpu_soc_mss_cur'] = float(vdd_cpu_soc_mss[0][0]) if vdd_cpu_soc_mss else None
    parsed_line['vdd_cpu_soc_mss_peak'] = float(vdd_cpu_soc_mss[0][1]) if vdd_cpu_soc_mss else None

    vin_sys_5v0 = re.findall(r'VIN_SYS_5V0 ([0-9.]+)mW/([0-9.]+)mW', line)
    parsed_line['vin_sys_5v0_cur'] = float(vin_sys_5v0[0][0]) if vin_sys_5v0 else None
    parsed_line['vin_sys_5v0_peak'] = float(vin_sys_5v0[0][1]) if vin_sys_5v0 else None

    vin = re.findall(r'\bVIN\s+([0-9.]+)mW/([0-9.]+)mW', line)
    parsed_line['vin_cur'] = float(vin[0][0]) if vin else None
    parsed_line['vin_peak'] = float(vin[0][1]) if vin else None

    # NVML GPU Utilization (appended by tegrastats_jp71_thor.py logger)
    gpu_util = re.findall(r'GPU_UTIL ([0-9]+)%', line)
    parsed_line['gpu_util'] = float(gpu_util[0]) if gpu_util else None

    mem_util = re.findall(r'MEM_UTIL ([0-9]+)%', line)
    parsed_line['mem_util'] = float(mem_util[0]) if mem_util else None

    return parsed_line


def parser(input_file, output_file):
    """
    Parse the input file and write the output to the output file.
    The input file is the tegrastats log file.
    The output file is the csv file that has values in float.
    """
    with open(output_file, 'w') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        header = False
        with open(input_file, 'r') as file:
            while True:
                line = file.readline()
                if not line:
                    break
                parsed_line = line_parser(line)

                if not header:
                    writer.writerow(parsed_line.keys())
                    header = True
                writer.writerow(parsed_line.values())


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument('--inp', '-i', type=str, help="Input Log File")
    args.add_argument('--out', '-o', type=str, help="Output CSV File")

    args = args.parse_args()
    if args.out is None:
        args.out = os.path.splitext(args.inp)[0] + '_parsed.csv'
    parser(args.inp, args.out)
