import csv
import os

def init_log_file(filepath, header):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)

def init_log_file(filepath, header):
    dir_name = os.path.dirname(filepath)
    if dir_name:  # Only create directory if it's not empty
        os.makedirs(dir_name, exist_ok=True)
    with open(filepath, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)


def append_log(filepath, row):
    with open(filepath, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(row)
