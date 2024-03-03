#! /usr/bin/env python3

"""
Convert a ULog file into CSV file(s)
"""

from __future__ import print_function
import os

# change the working directory to the DeepNav_data directory
os.chdir(os.path.dirname(os.path.abspath(__file__)))


ulg_dir =  os.path.join(os.path.pardir, "DeepNav_data", "ulg_files")
csvs_dir = os.path.join(os.path.pardir, "DeepNav_data", "flight_csvs")

if not os.path.isdir(csvs_dir) :
    os.mkdir(csvs_dir)

# Iterate on the flights (one folder per flight)
log_list = sorted(os.listdir(ulg_dir))
for flight_number, flight_name in enumerate(log_list):
    
    print("processing flight number ", flight_number, "\t", flight_name)
    trimmed_flight_name = flight_name
    if len(flight_name) > 42:
        print ("Removing file hash: " + flight_name)
        os.rename(os.path.join(ulg_dir, flight_name), os.path.join(ulg_dir, flight_name[:-41]))
        trimmed_flight_name = flight_name[:-41]
    

    os.system("ulog2csv " + os.path.join(ulg_dir, trimmed_flight_name) + " -o " + os.path.join(csvs_dir, trimmed_flight_name))  
