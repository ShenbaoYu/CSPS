# -*- coding: utf-8 -*-
"""
The main entry point for the proposed CPSP model
"""

import os, sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

import numpy as np
import igraph as ig
import networkx as nx
import matplotlib.pyplot as plt
from CSPS import gaus_network, spsl
from Evaluation import evaluation as ev
from DataPre import data_pre as dp


def out_to_file(path, model_name):

    class logger(object):
        
        def __init__(self, file_name, path):
            self.terminal = sys.stdout
            self.log = open(os.path.join(path, file_name), mode='a', encoding='utf8')
        
        def write(self, message):
            self.terminal.write(message)
            self.log.write(message)
        
        def flush(self):
            pass
    
    sys.stdout = logger(model_name + '.log', path=path)




if __name__ == '__main__':
    
    DATASET = "Sync1-150"
    MISS_RATE = 0  # the ratio of randomly missing values
    stu_exe_file = BASE_DIR + "/Data/" + DATASET + "/data.txt"
    q_file = BASE_DIR + "/Data/" + DATASET + "/q.txt"
    stu_exe = ((np.loadtxt(stu_exe_file)).astype(float)).T  # read the Student-Exercise Data; ROW:exercise, COL:student
    q_m = np.loadtxt(q_file, dtype=int)  # read the Q-Matrix; ROW:exercise, COL:knowledge concept
    stu_exe_miss, miss_coo = dp.missing_stu_exe(stu_exe, MISS_RATE)  # randomly miss the data

    T = 1  # the dimensionality of latent variable (knowledge concept)
    # out_to_file(BASE_DIR + "/CSPS/results/", '@' + DATASET)
    print('\n')
    print("---- The dataset is : %s ----" % DATASET)
    print("Dimension of latent skill T : %d" % T)

    """ --- SKILLS' CORRELATION RELATIONSHIPS DISCOVERY --- """
    # GAUSSIAN MODEL TRAINING
    gn = gaus_network.gaussian_network(stu_exe_miss, q_m, T, rotation="varimax")
    gn.train_em(max_iter=500, cri=0.1)
    # np.set_printoptions(formatter={'float': '{: 0.6f}'.format})
    # print("The skill covariance matrix is:\n", gn.sigma_Zs_pos)
    # gn.save(BASE_DIR + "/CSPS/params/" + DATASET)
    # gn.load(BASE_DIR + "/CSPS/params/" + DATASET)
    """ --- SKILLS' PREREQUISITE STRUCTURE LEARNING --- """
    mb = spsl.spsl(gn)
    lg = mb.searching()

    """ --- TESTING --- """
    tg = ev.true_graph(BASE_DIR + "/Data/" + DATASET + "/triggernames.txt")
    res = ev.adj_ori_rate(true=tg, learned=lg)