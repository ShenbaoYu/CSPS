"""
The main entry for the GaussSEM EM based on Latent Gaussian Network
"""

import os, sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

import numpy as np
from DataPre import data_pre as dp
from GaussSEM import em_gaussian
from Evaluation import evaluation as ev




if __name__ == '__main__':

    DATASET = "Sync1"
    MISS_RATE = 0.0

    stu_exe_file = BASE_DIR + "/Data/" + DATASET + "/data.txt"
    q_file = BASE_DIR + "/Data/" + DATASET + "/q.txt"

    stu_exe = np.loadtxt(stu_exe_file, dtype=float, delimiter='\t').T  # read the Student-Exercise Data; ROW:exercise, COL:student
    q_m = np.loadtxt(q_file, dtype=int)  # read the Q-Matrix; ROW:exercise, COL:knowledge concept

    stu_exe_miss, miss_coo = dp.missing_stu_exe(stu_exe, MISS_RATE)

    """ --- MODEL TRAINING --- """
    T = 1  # the dimensionality of latent variable (knowledge concept)
    gn = em_gaussian.em_gaussian(stu_exe_miss, q_m, T)
    nm = gn.structure_em(max_iter=500, cri=10, R=1)
    print(nm)

    """ --- MODEL TESTING --- """
    # tg = ev.true_graph(BASE_DIR + "/Data/" + DATASET + "/triggernames.txt")
    # lg = dp.translate_to_graph(nm)
    # ev.adj_ori_rate(true=tg, learned=lg)



