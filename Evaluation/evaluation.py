# -*- coding: utf-8 -*-

from decimal import DivisionByZero
import numpy as np
import math
import re


def rmse(stu_exe_true, U, PHI, THETA, mu_Zs_cond_obs_lis, miss_coo):
    """
    FUNCTION: Evaluation of the Student-Exercise predicting matrix

    Inputs:
    -----------------
    :param stu_exe_true --> numpy.ndarray
        the true Student-Exercise Data Matrix
        row : exercise
        col : student

    :param U --> numpy.ndarray
    
    :param PHI --> numpy.ndarray

    :param THETA --> numpy.ndarray
        the variance of the Gaussian noise epsilon
    
    :param mu_Zs_cond_obs_lis --> dict{d: mu_Zs_cond_obs --> numpy.ndarray(), ...}
        the conditional mean vector of the probability distribution of the joint latent variables conditioned on observations
    
    :param miss_coo --> tuple()
        the missing coordinates
    
    Outputs:
    -----------------
    :return rmse --> float
    """

    rmse = 0

    for _ in miss_coo:
        n, d = _[0], _[1]
        obs_true = stu_exe_true[n][d]
        mean = (np.dot(U[n], mu_Zs_cond_obs_lis[d]) + PHI[n])[0]
        obs_pre = np.random.normal(loc=mean, scale=math.sqrt(THETA[n]))
        rmse += math.pow(abs(obs_true - obs_pre), 2)
    try: 
        rmse = math.sqrt(rmse / len(miss_coo))
    except BaseException:
        print("The test set is empty")
        return None

    return rmse


def true_graph(filename):
    """
    FUNCTION: build the true graph based on the experience of experts,
    each skills with its possibly triggers labelled by experts

    Inputs:
    -----------------
    :params filename --> str

    Outputs:
    -----------------
    :return true_graph  --> dict()
        skill id with its trigger skills(id)
    """
    meta_data = dict()
    qnames = dict()  # skill name: skill id
    true_graph = dict()
    _ = 0
    with open(filename, "r") as file:
        for line in file.readlines()[1:]:  # skip the first line
            line = line.strip('\n').split('\t')
            meta_data[_] = [line[0], line[1], line[2]]
            qnames[line[1]] = int(line[0])
            _ += 1
             
    for meta in meta_data.values():
        skill_id = int(meta[0])

        if not skill_id in true_graph:
            true_graph[skill_id] = list()

        if len(re.findall(re.compile(r"\[(.+?)\]"), meta[2])):
            _ = re.findall(re.compile(r"\[(.+?)\]"), meta[2])[0]
            triggers = re.findall(re.compile(r"\'(.+?)\'"), _)
        else:
            continue  
        for trigger_name in triggers:
            trigger_id = qnames[trigger_name]
            true_graph[skill_id].append(trigger_id)  # add the trigger (parent)

    return true_graph


def adj_ori_rate(true, learned):
    """
    FUNCTION: Evaluation of the learned structure
    metrics:
    1. True positive for adjacency rate (TPAR)
    2. True discovery for adjacency rate (TDAR)
    3. F1-AR
    4. True positive for orientation rate (TPOR)
    5. True discovery for orientation rate (TDOR)
    6. F1-OR

    Inputs:
    -----------------
    :param true --> dict(id : [parent id])
        the true graph
    
    :param learned --> dict(id : [parent id])
        the learned graph

    Outputs:
    -----------------
    """

    cor_adj, adj_tru, adj_lea = 0, 0, 0
    cor_dir, dir_tru, dir_lea = 0, 0, 0

    flag = False
    for node, edges in learned.items():
        if len(edges) != 0:
            flag = True
    
    if not flag:
        print("there is no edge in the learned graph...")
        return

    # --- find all directed edges in true graph and learned graph respectively --- 
    dir_edges_tru = list()
    for node, parents_tru in true.items():  # directed edges in true graph
        for pa in parents_tru:
            dir_edges_tru.append((pa, node))

    dir_edges_lea = list()
    for node, parents_lea in learned.items():  # directed edges in learned graph
        for pa in parents_lea:
            # edges_lea.append((node, pa)) if node < pa else edges_lea.append((pa, node))
            dir_edges_lea.append((pa, node))
    
    # TPOR, TDOR, F1-OR
    dir_tru = len(dir_edges_tru)
    dir_lea = len(dir_edges_lea)
    cor_dir = len(set(dir_edges_tru).intersection(set(dir_edges_lea)))
    tpor = cor_dir / dir_tru
    tdor = cor_dir / dir_lea
    try:
        f1_or = 2 * tpor * tdor / (tpor + tdor)
    except ZeroDivisionError:
        f1_or = 0


    # --- undirected edges ---
    undir_edges_tru, undir_edges_lea = list(), list()
    for edge in dir_edges_tru:
        undir_edges_tru.append((edge[0], edge[1])) if edge[0] < edge[1] \
        else undir_edges_tru.append((edge[1], edge[0]))
    for edge in dir_edges_lea:
        undir_edges_lea.append((edge[0], edge[1])) if edge[0] < edge[1] \
        else undir_edges_lea.append((edge[1], edge[0]))

    # TPAR, TDAR, F1-AR
    adj_tru = len(undir_edges_tru)
    adj_lea = len(undir_edges_lea)
    cor_adj = len(set(undir_edges_tru).intersection(set(undir_edges_lea)))
    tpar = cor_adj / adj_tru
    tdar = cor_adj / adj_lea
    try:
        f1_ar = 2 * tpar * tdar / (tpar + tdar)
    except ZeroDivisionError:
        f1_ar = 0

    print("TPAR:%.4f, TDAR:%.4f, F1-AR:%.4f, TPOR:%.4f, TDOR:%.4f, F1-OR:%.4f" %(tpar, tdar, f1_ar, tpor, tdor, f1_or))

    return [round(tpar, 4), round(tdar, 4), round(f1_ar, 4), round(tpor, 4), round(tdor, 4), round(f1_or, 4)]