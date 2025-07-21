# -*- coding: utf-8 -*-
"""
randomly divide the data into training and testing
"""

import os
import sys
import numpy as np
import random

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)
DATASET = 'FrcSub'

ratio = 0.8  # 0.2, 0.4, 0.6, 0.8
data = BASE_DIR + "/Data/" + DATASET + "/data.txt"
stu_exe = np.loadtxt(data, delimiter='\t')
# divide data
stu_num = stu_exe.shape[0]
count = int(ratio * stu_num)
rows = list()  # rows used to form test data
while count != 0:
    row = random.randint(0, stu_num-1)
    if not row in rows:
        rows.append(row)
        count -= 1
    else:
        continue
train = (np.delete(stu_exe, rows, axis=0)).astype(int)  # build the training data
test = (stu_exe[rows, :]).astype(int)  # build the testing data
# save files
save_dir = BASE_DIR + "/Data/" + DATASET + str(ratio)
np.savetxt(save_dir + '/train.txt', train, fmt='%i')
np.savetxt(save_dir + '/test.txt', test, fmt='%i')




