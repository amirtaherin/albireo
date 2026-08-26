# Copyright (c) 2025 Amir Taherin
# Author: Amir Taherin
# Last Modified: 2025-06-06 15:13:23
import argparse
import csv
import os
import re
from datetime import datetime, timezone
from zoneinfo import ZoneInfo


"""
This script parses the tegrastats log file and converts it to a csv file.
I adopted this script from the tegrastats_parser.py file in the utils folder.
The original tegrastat parser (designed for MIRA) worked for tegrastats of JP5.
This version works for tegrastats of JP6.
"""

def line_parser(line):
    """
    Parse a line of tegrastats output and return a dictionary of the values.
    To parse the line reqular expressions are used.
    """

    parsed_line = {}

    # DATE and TIME
    # The following line is commented out because it does not work with the new tegrastats
    # dtime = re.findall(r'\d{2}[-]\d{2}[-]\d{4}\s\d{2}:\d{2}:\d{2}', line)
    # dtime = datetime.strptime(dtime[0], '%m-%d-%Y %H:%M:%S')
    # parsed_line['timestamp'] = dtime.timestamp()

    # The following line works with the new tegrastats
    # Step 1: Extract all datetime-like strings (with and without fractional seconds)
    matches = re.findall(r'\d{2}-\d{2}-\d{4} \d{2}:\d{2}:\d{2}(?:\.\d{1,9})?', line)

    # Step 2: Pick the one with fractional seconds (i.e., nanosecond precision)
    ns_time_str = next((m for m in matches if '.' in m), None)
    # Step 3: Parse that string with nanosecond support (pad right to 9 digits if needed)
    if ns_time_str:
        dt, ns = ns_time_str.split('.')
        ns = ns.ljust(9, '0')[:9]  # Ensure 9 digits
        dt_obj = datetime.strptime(dt, '%m-%d-%Y %H:%M:%S').replace(tzinfo=ZoneInfo("America/New_York"))
        dt_obj = dt_obj.astimezone(timezone.utc)
        timestamp_ns = int(dt_obj.timestamp()) * 10**9 + int(ns)
        parsed_line['timestamp'] = timestamp_ns
    else:
        print("No high-precision timestamp found.")
        exit(1)

    # RAM
    ram = re.findall(r'RAM ([0-9]*)\/([0-9]*)MB \(lfb ([0-9]*)x([0-9]*)MB\)', line)
    parsed_line['ram_used'] = float(ram[0][0]) if ram else None
    parsed_line['ram_total'] = float(ram[0][1]) if ram else None
    parsed_line['ram_lfb_count'] = float(ram[0][2]) if ram else None
    parsed_line['ram_lfb_size'] = float(ram[0][3]) if ram else None

    # SWAP
    swap = re.findall(r'SWAP ([0-9]*)\/([0-9]*)MB \(cached ([0-9]*)MB\)', line)
    parsed_line['swap_used'] = float(swap[0][0]) if swap else None
    parsed_line['swap_total'] = float(swap[0][1]) if swap else None
    parsed_line['swap_cached'] = float(swap[0][2]) if swap else None

    # CPU
    # cpus = re.findall(r'CPU \[(.*)\]', line) => this worked for older version of tegrastats
    # In the newer version of tegrastats there is one input for GPU which uses [] and messed up the regex
    cpus = re.findall(r'CPU \[(.*)\] EMC', line)
    cpus = cpus[0].split(",")
    for i in range(len(cpus)):
        if cpus[i] == 'off':
            cpus[i] = 'inf%@inf'
        cpu = cpus[i].split('%@')
        parsed_line[f'cpu_{i}_load'] = float(cpu[0]) if cpu else None
        parsed_line[f'cpu_{i}_freq'] = float(cpu[1]) if cpu else None


    # EMC_FREQ
    emc = re.findall(r'EMC_FREQ ([0-9]*)%@?([0-9]*)?', line)
    parsed_line['emc_load'] = float(emc[0][0]) if emc else None
    parsed_line['emc_freq'] = float(emc[0][1]) if emc else None



    # GRD_FREQ
    # The following two lines won't work with the new tegrastats.
    # new tegrastats has "GR3D_FREQ 0%@[305,305] NVENC" for GPUs
    #gpu1 = re.findall(r'GR3D_FREQ ([0-9]*)%@?([0-9]*)?', line)
    #gpu2 = re.findall(r'GR3D2_FREQ ([0-9]*)%@?([0-9]*)?', line)
    # Parsing  "GR3D_FREQ 0%@[305,305] NVENC
    gpu = re.findall(r'GR3D_FREQ ([0-9]*)%@?\[([0-9]*),([0-9]*)\]', line)
    parsed_line['gpc1_load'] = float(gpu[0][0]) if gpu else None
    parsed_line['gpc1_freq'] = float(gpu[0][1]) if gpu else None
    parsed_line['gpc2_load'] = float(gpu[0][0]) if gpu else None
    parsed_line['gpc2_freq'] = float(gpu[0][2]) if gpu else None

    # NVENC
    nvenc = re.findall(r'NVENC (off+[0-9]*)', line)
    parsed_line['NVENC'] = 'off' if nvenc == ['off'] else float(nvenc[0]) if nvenc else None
     
    # NVDEC
    nvdec = re.findall(r'NVDEC (off+[0-9]*)', line)
    parsed_line['NVDEC'] = 'off' if nvdec == ['off'] else float(nvdec[0]) if nvdec else None

    # NVJPG
    nvjpg = re.findall(r'NVJPG (off+[0-9]*)', line)
    parsed_line['NVJPG0'] = 'off' if nvjpg == ['off'] else float(nvjpg[0]) if nvjpg else None

    # NVJPG1
    nvjpg1 = re.findall(r'NVJPG1 (off+[0-9]*)', line)
    parsed_line['NVJPG1'] = 'off' if nvjpg1 == ['off'] else float(nvjpg1[0]) if nvjpg1 else None

    # VIC
    vic = re.findall(r'VIC (off+[0-9]*)', line)
    parsed_line['VIC'] = 'off' if vic == ['off'] else float(vic[0]) if vic else None

    # OFA
    ofa = re.findall(r'OFA (off+[0-9]*)', line)
    parsed_line['OFA'] = 'off' if ofa == ['off'] else float(ofa[0]) if ofa else None

    # NVDLA0
    nvdla0 = re.findall(r'NVDLA0 (off+[0-9]*)', line)
    parsed_line['NVDLA0'] = 'off' if nvdla0 == ['off'] else float(nvdla0[0]) if nvdla0 else None

    # NVDLA1
    nvdla1 = re.findall(r'NVDLA1 (off+[0-9]*)', line)
    parsed_line['NVDLA1'] = 'off' if nvdla1 == ['off'] else float(nvdla1[0]) if nvdla1 else None

    # PVA0_FREQ
    pva0_freq = re.findall(r'PVA0_FREQ (off+[0-9]*)', line)
    parsed_line['pva0_freq'] = 'off' if pva0_freq == ['off'] else float(pva0_freq[0]) if pva0_freq else None


    # VIC_FREQ
    vic_freq = re.findall(r'VIC_FREQ ([0-9]*)', line)
    parsed_line['vic_freq'] = float(vic_freq[0]) if vic_freq else None

    # APE
    ape = re.findall(r'APE ([0-9]*)', line)
    parsed_line['ape'] = float(ape[0]) if ape else None

    # Wont work on the new tegrastats 
    #CV0, CV1, CV2
    #cv0 = re.findall(r'CV0@([(-|)][0-9.]*)', line)
    #cv1 = re.findall(r'CV1@([(-|)][0-9.]*)', line)
    #cv2 = re.findall(r'CV2@([(-|)][0-9.]*)', line)
    #parsed_line['cv0'] = float(cv0[0]) if cv0 else None
    #parsed_line['cv1'] = float(cv1[0]) if cv1 else None
    #parsed_line['cv2'] = float(cv2[0]) if cv2 else None

    # Tempratures
    cpu_temp = re.findall(r'cpu@([(-|)][0-9.]*)C', line)
    parsed_line['cpu_temp'] = float(cpu_temp[0]) if cpu_temp else None

    # not in the new tegrastats 
    tboard_temp = re.findall(r'Tboard@([(-|)][0-9.]*)C', line)
    parsed_line['tboard_temp'] = float(tboard_temp[0]) if tboard_temp else None

    # not in the new tegrastats
    tdiode_temp = re.findall(r'Tdiode@([(-|)][0-9.]*)C', line)
    parsed_line['tdiode_temp'] = float(tdiode_temp[0]) if tdiode_temp else None

    soc0_temp = re.findall(r'soc0@([(-|)][0-9.]*)C', line)
    parsed_line['soc_0_temp'] = float(soc0_temp[0]) if soc0_temp else None

    soc1_temp = re.findall(r'soc1@([(-|)][0-9.]*)C', line)
    parsed_line['soc_1_temp'] = float(soc1_temp[0]) if soc1_temp else None

    soc2_temp = re.findall(r'soc2@([(-|)][0-9.]*)C', line)
    parsed_line['soc_2_temp'] = float(soc2_temp[0]) if soc2_temp else None

    gpu_temp = re.findall(r'gpu@([(-|)][0-9.]*)C', line)
    parsed_line['gpu_temp'] = float(gpu_temp[0]) if gpu_temp else None

    tj_temp = re.findall(r'tj@([(-|)][0-9.]*)C', line)
    parsed_line['tj_temp'] = float(tj_temp[0]) if tj_temp else None

    # Power
    vdd_gpu_soc = re.findall(r'VDD_GPU_SOC ([0-9.]*)mW\/([0-9.]*)mW', line)
    parsed_line['vdd_gpu_soc_cur'] = float(vdd_gpu_soc[0][0]) if vdd_gpu_soc else None
    parsed_line['vdd_gpu_soc_avg'] = float(vdd_gpu_soc[0][1]) if vdd_gpu_soc else None

    vdd_cpu_cv = re.findall(r'VDD_CPU_CV ([0-9.]*)mW\/([0-9.]*)mW', line)
    parsed_line['vdd_cpu_cv_cur'] = float(vdd_cpu_cv[0][0]) if vdd_cpu_cv else None
    parsed_line['vdd_cpu_cv_avg'] = float(vdd_cpu_cv[0][1]) if vdd_cpu_cv else None

    vin_sys_5V0 = re.findall(r'VIN_SYS_5V0 ([0-9.]*)mW\/([0-9.]*)mW', line)
    parsed_line['vin_sys_5v0_cur'] = float(vin_sys_5V0[0][0]) if vin_sys_5V0 else None
    parsed_line['vin_sys_5v0_avg'] = float(vin_sys_5V0[0][1]) if vin_sys_5V0 else None

    vddq_vdd2_1v8ao = re.findall(r'VDDQ_VDD2_1V8AO ([0-9.]*)mW\/([0-9.]*)mW', line)
    parsed_line['vddq_vdd2_1v8ao_cur'] = float(vddq_vdd2_1v8ao[0][0]) if vddq_vdd2_1v8ao else None
    parsed_line['vddq_vdd2_1v8ao_avg'] = float(vddq_vdd2_1v8ao[0][1]) if vddq_vdd2_1v8ao else None

    return parsed_line


def parser(input_file, output_file):
    """
    Parse the input file and write the output to the output file
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
