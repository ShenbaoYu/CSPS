# -*- coding: utf-8 -*-
"""
translate the continuous data into binary one.
"""


import os, sys
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)
import numpy as np
import re




DATASET_FROM = 'Sync2-2000'  # Sync1-xxx, Sync2-xxx, Alg0506

if re.match(r'.*Sync.*', DATASET_FROM):
# if DATASET_FROM in ['Sync1', 'Sync2']:
    data = np.loadtxt(BASE_DIR + "/Data/" + DATASET_FROM + "/data.txt")
    aves = np.average(data, axis=0)
    data_binary = np.zeros(shape=(data.shape[0], data.shape[1]), dtype=float)

    for col in range(data.shape[1]):
        ave = aves[col]
        copy = data[:, col].copy()
        copy[copy <= ave], copy[copy > ave] = 0, 1  # using the meamn value
        data_binary[:, col] = copy
    np.savetxt(BASE_DIR + "/Data/" + DATASET_FROM + ' binary' + "/data.txt", data_binary, fmt='%d', delimiter='\t')
    if not os.path.exists(BASE_DIR + "/Data/" + DATASET_FROM + ' binary' + "/json"):
        os.mkdir(BASE_DIR + "/Data/" + DATASET_FROM + ' binary' + "/json")

if DATASET_FROM in ['Alg0506']:
    data = np.loadtxt(BASE_DIR + "/Data/" + DATASET_FROM + "/data.txt")
    data[data < 1] = 0
    np.savetxt(BASE_DIR + "/Data/" + DATASET_FROM + ' binary' + "/data.txt", data, delimiter='\t')


