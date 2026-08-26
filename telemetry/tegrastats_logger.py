# Copyright (c) 2025 Amir Taherin
# Author: Amir Taherin
# Last Modified: 2025-05-21 19:37:12
# __________________________________
import subprocess
import csv
from time import sleep
import os
import signal

""" This script is used to run tegrastats command on a Jetson device."""
""" The original script was designed for MIRA project. I modified it for better usability. """

class Tegrastats:
    def __init__(self, log_file="output_log.txt", interval=1, verbose=False):
        self.interval = interval
        self.log_file = log_file
        self.verbose = verbose

    def compose_command(self):
        #command = f"sudo tegrastats --interval {self.interval} --logfile {self.log_file}"  # I did not use this since it prints in the output
        #command = f"tegrastats --interval {self.interval} > {self.log_file}"   # Redirecting output to a file

        command = (
            f"sudo tegrastats --interval {self.interval} | "
            "while IFS= read -r line; do "
            "printf '%s %s\\n' \"$(date '+%m-%d-%Y %H:%M:%S.%9N')\" \"$line\"; "
            f"done > {self.log_file}"
        )  # Redirecting output to a file with timestamp in nanoseconds format
        # Note: sudo required for full output (EMC, GPU freq, NVENC/NVDEC, etc.)
        # Ensure tegrastats is in sudoers with NOPASSWD

        return command

    def start_tegrastats(self):
        command = self.compose_command()
        process = None
        try:
            process = subprocess.Popen(command, shell=True, preexec_fn=os.setsid)
            print(f"Tegrastats started with PID: {process.pid}")
        except subprocess.CalledProcessError:
            print("Error: tegrastats failed to start")
        return process

    def stop_tegrastats(self, process):
        try:
            process.terminate()  # Terminate the process
            os.killpg(os.getpgid(process.pid), signal.SIGTERM)  # Send the SIGTERM signal to the process group
            sleep(1)  # Give it a moment to terminate

            if process.poll() is None:
                print("Warning: tegrastats did not terminate, killing it")
                os.killpg(os.getpgid(process.pid), signal.SIGKILL)

            print("Tegrastats stopped successfully")
        except subprocess.CalledProcessError:
            print("Error: tegrastats failed to stop")


# Defining helper functions
def write_to_csv(output_path,header,values):
	with open(output_path, mode='w') as gtFile:
		writer = csv.writer(gtFile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
		writer.writerow(header)
		for value in values:
			writer.writerow(value)


if __name__ == '__main__':
    tegrastats = Tegrastats()
    command = tegrastats.compose_command()
    print(command)
    process = tegrastats.start_tegrastats()
    sleep(5)
    tegrastats.stop_tegrastats(process)


