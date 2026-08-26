# Copyright (c) 2025 Amir Taherin
# Author: Amir Taherin
# Last Modified: 2025-04-10 20:49:28
import argparse
import csv
import os
import re
from datetime import datetime


def line_parser(line):
    """
    Parse a line of tegrastats output and return a dictionary of the values.
    To parse the line reqular expressions are used.
    """

    parsed_line = {}

    #print(line)

    # DATE and TIME - done
    dtime = re.findall(r'\d{2}[-]\d{2}[-]\d{4}\s\d{2}:\d{2}:\d{2}', line)
    dtime = datetime.strptime(dtime[0], '%m-%d-%Y %H:%M:%S')
    parsed_line['timestamp'] = dtime.timestamp()

    # RAM - done
    ram = re.findall(r'RAM ([0-9]*)\/([0-9]*)MB \(lfb ([0-9]*)x([0-9]*)MB\)', line)
    parsed_line['ram_used'] = float(ram[0][0]) if ram else None
    parsed_line['ram_total'] = float(ram[0][1]) if ram else None
    parsed_line['ram_lfb_count'] = float(ram[0][2]) if ram else None
    parsed_line['ram_lfb_size'] = float(ram[0][3]) if ram else None

    # SWAP - done
    swap = re.findall(r'SWAP ([0-9]*)\/([0-9]*)MB \(cached ([0-9]*)MB\)', line)
    parsed_line['swap_used'] = float(swap[0][0]) if swap else None
    parsed_line['swap_total'] = float(swap[0][1]) if swap else None
    parsed_line['swap_cached'] = float(swap[0][2]) if swap else None

    # CPU - done
    #cpus = re.findall(r'CPU \[(.*)\]', line)
    cpus = re.findall(r'CPU \[(.*?)\]', line)
    cpus = cpus[0].split(",")
    for i in range(len(cpus)):
        if cpus[i] == 'off':
            cpus[i] = 'inf%@inf'
        cpu = cpus[i].split('%@')
        parsed_line[f'cpu_{i}_load'] = float(cpu[0]) if cpu else None
        parsed_line[f'cpu_{i}_freq'] = float(cpu[1]) if cpu else None


    # EMC_FREQ -done 
    emc = re.findall(r'EMC_FREQ (\d+)%', line)
    parsed_line['emc_load'] = float(emc[0]) if emc else None
    #parsed_line['emc_freq'] = float(emc[0][1]) if emc else None


    # GRD_FREQ
    gpu1 = re.findall(r'GR3D_FREQ (\d+)%@\[(\d+)\]', line)
    gpu2 = re.findall(r'GR3D2_FREQ ([0-9]*)%@?([0-9]*)?', line)
    parsed_line['gpu1_load'] = float(gpu1[0][0]) if gpu1 else None
    parsed_line['gpu1_freq'] = float(gpu1[0][1]) if gpu1 else None
    parsed_line['gpu2_load'] = float(gpu2[0][0]) if gpu2 else None
    parsed_line['gpu2_freq'] = float(gpu2[0][1]) if gpu2 else None

    #NVJPG1
    nvjpg1 = re.findall(r'NVJPG1 ([0-9]*)', line)
    parsed_line['nvjpg1'] = float(nvjpg1[0]) if nvjpg1 else None

    #VIC_FREQ
    vic_freq = re.findall(r'VIC_FREQ ([0-9]*)', line)
    parsed_line['vic_freq'] = float(vic_freq[0]) if vic_freq else None

    #APE
    ape = re.findall(r'APE ([0-9]*)', line)
    parsed_line['ape'] = float(ape[0]) if ape else None

    # CV0, CV1, CV2
    cv0 = re.findall(r'CV ([0-9.]*)mW\/([0-9.]*)mW', line)
    #cv0 = re.findall(r'CV0@([(-|)][0-9.]*)', line)
    cv1 = re.findall(r'CV1@([(-|)][0-9.]*)', line)
    cv2 = re.findall(r'CV2@([(-|)][0-9.]*)', line)
    
    parsed_line['cv0'] = float(cv0[0][0]) if cv0 else None
    parsed_line['cv1'] = float(cv1[0]) if cv1 else None
    parsed_line['cv2'] = float(cv2[0]) if cv2 else None

    # Tempratures
    cpu_temp = re.findall(r'CPU@([(-|)][0-9.]*)C', line)
    parsed_line['cpu_temp'] = float(cpu_temp[0]) if cpu_temp else None

    tboard_temp = re.findall(r'Tboard@([(-|)][0-9.]*)C', line)
    parsed_line['tboard_temp'] = float(tboard_temp[0]) if tboard_temp else None

    tdiode_temp = re.findall(r'Tdiode@([(-|)][0-9.]*)C', line)
    parsed_line['tdiode_temp'] = float(tdiode_temp[0]) if tdiode_temp else None

    soc0_temp = re.findall(r'SOC0@([(-|)][0-9.]*)C', line)
    parsed_line['soc_0_temp'] = float(soc0_temp[0]) if soc0_temp else None

    soc1_temp = re.findall(r'SOC1@([(-|)][0-9.]*)C', line)
    parsed_line['soc_1_temp'] = float(soc1_temp[0]) if soc1_temp else None

    soc2_temp = re.findall(r'SOC2@([(-|)][0-9.]*)C', line)
    parsed_line['soc_2_temp'] = float(soc2_temp[0]) if soc2_temp else None

    gpu_temp = re.findall(r'GPU@([(-|)][0-9.]*)C', line)
    parsed_line['gpu_temp'] = float(gpu_temp[0]) if gpu_temp else None

    tj_temp = re.findall(r'tj@([(-|)][0-9.]*)C', line)
    parsed_line['tj_temp'] = float(tj_temp[0]) if tj_temp else None

    # Power
    vdd_gpu_soc = re.findall(r'GPU ([0-9.]*)mW\/([0-9.]*)mW', line)
    parsed_line['vdd_gpu_soc_cur'] = float(vdd_gpu_soc[0][0]) if vdd_gpu_soc else None
    parsed_line['vdd_gpu_soc_avg'] = float(vdd_gpu_soc[0][1]) if vdd_gpu_soc else None

    vdd_cpu_cv = re.findall(r'CPU ([0-9.]*)mW\/([0-9.]*)mW', line)
    parsed_line['vdd_cpu_cv_cur'] = float(vdd_cpu_cv[0][0]) if vdd_cpu_cv else None
    parsed_line['vdd_cpu_cv_avg'] = float(vdd_cpu_cv[0][1]) if vdd_cpu_cv else None

    vin_sys_5V0 = re.findall(r'SYS5V ([0-9.]*)mW\/([0-9.]*)mW', line)
    parsed_line['sys5v0_cur'] = float(vin_sys_5V0[0][0]) if vin_sys_5V0 else None
    parsed_line['sys5v0_avg'] = float(vin_sys_5V0[0][1]) if vin_sys_5V0 else None

    vddq_vdd2_1v8ao = re.findall(r'VDDRQ ([0-9.]*)mW\/([0-9.]*)mW', line)
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
