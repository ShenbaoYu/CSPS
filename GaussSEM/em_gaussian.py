# -*- coding: utf-8 -*-
"""
Constructing the Latent - GAUSSIAN NETWORK model based on Structure EM Algorithm 
    The latent variables: the knowledge concept
    The observed variables: the records of students doing exercises

ALGORITHM: Structure EM Algorithm
"""

import os, sys
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

import numpy as np
import math
from random import choice, shuffle, random
from GaussSEM import modules, gaus_net


class gaus_node():
    """
    The CLASS of latent variable
    """

    def __init__(self, id, dim) -> None:
        """
        FUNCTION: Initializing a CLASS of variable subject to Gaussian distribution
        including the mean vector and the covariance matrix.
          
        Inputs:
        -----------------
        :param id --> int
            The ID
        
        :param dim --> int
            the dimensionality of the latent variable

        NOTE: we assume the prior distribution of the latent variable follow the
              Gaussian distribution with ZERO mean and UNIT variance.
        """

        self.id = id  # ID
        self.mu = np.zeros(shape=[dim, 1], dtype=float)  # mean vector (Zero mean)
        self.sigma = np.identity(dim)  # covariance matrix (Identity Matrix)


class em_gaussian():
    """
    Structure EM based Gaussian Network
    """

    def __init__(self, stu_exe, q_matrix, dim) -> None:
        
        self.stu_exe, self.q_matrix, self.dim = stu_exe, q_matrix, dim

        obs_num, latent_num = q_matrix.shape

        # initialize each latent variables
        self.latent_set = dict()  # the set of CLASS of latent variables
        for _ in range(latent_num):
            self.latent_set[_] = gaus_node(_, self.dim)
        
        self.mu_Zs_pos_lis = dict()  # the posterior mean conditioned on observations

        # initialize the loading matrix,
        # where each entry is the feature vector for each latent variable with a particular observed node,
        # for example:
        # U = 
        #    +--------+--------+-------+--------+
        #    | U[1,1] | U[1,2] |  ...  | U[1,K] |
        #    | U[2,1] | U[2,2] |  ...  | U[2,K] |
        #    |  ...   |  ...   |  ...  |  ...   |
        #    | U[N,1] | U[N,2] |  ...  | U[N,K] |
        #    +--------+--------+-------+--------+
        #    U[n,k] = [Unk[1], Unk[2], ..., Unk[T]].T
        self.U = np.random.rand(obs_num*self.dim, latent_num) * self.q_matrix

        # initializing the exercise feature vector
        self.PHI = np.random.uniform(size=obs_num).reshape(obs_num, 1)  #  # observation equals # exercises

        # Initializing the Guassian noise
        self.THETA = np.random.uniform(size=obs_num)

        # initialize the adjacency matrix of latent variables
        # for example:
        # +----+----+----+
        # |    | Z1 | Z2 |
        # +----+----+----+
        # | Z1 | 0  | -1 |
        # +----+----+----+
        # | Z2 | 1  |  0 |
        # +----+----+----+
        # Z1 is the father of Z2, so Z2 is the child of Z1, and diagonal elements are set to 0.
        
        self.nm = np.zeros(shape=(latent_num, latent_num), dtype=int)  # initialization as the empty graph

        # latent_list = list()
        # for _ in self.latent_set.values():
        #     latent_list.append(_.id)
        # shuffle(latent_list)  # randomly sort the latent variables as the topological structure
        # for i in range(0, len(latent_list)):
        #     id_i = latent_list[i]
        #     for j in range(i+1, len(latent_list)):
        #         id_j = latent_list[j]
        #         is_edging = choice([True, False])  # whether add an edge between i and j
        #         if is_edging:
        #             self.nm[id_i][id_j] = -1
        #             self.nm[id_j][id_i] = 1
        
        # initialize the weight matrix (directed acyclic graph, DAG) of latent variables
        # for example:
        # +----+-----+-----+
        # |    | Z1  |  Z2 |
        # +----+-----+-----+
        # | Z1 | 1   |  0  |
        # +----+-----+-----+
        # | Z2 | W12 |  0  |
        # +----+-----+-----+
        # NOTE: if there is no parent for Zk, we set W[Zk][Zk] = 1

        self.W = np.zeros(shape=(latent_num, latent_num), dtype=float)  # initialization as the empty matrix
        for k in range(latent_num):
             self.W[k][k] = 1
        
        # row, col = self.nm.shape
        # for i in range(row):
        #     if not np.any(self.nm[i] == 1):  # latent i has no parents
        #         self.W[i][i] = 1
        #     else:
        #         for j in range(col):
        #             if self.nm[i][j] == 1:
        #                 self.W[i][j] = random()

        self.obj = -np.inf


    def structure_em(self, max_iter, cri, R):
        """
        FUNCTION: Estimating the parameter and structure simultaneously
                  based on the Structure EM algorithm
        
        Inputs:
        -----------------
        :param max_iter --> int
            maximum iterations of parameters estimation for a particular structure

        :param cri --> float
            termination cirteria
        
        :param R --> int
            maximum iterations for choosing the best Q function
        """

        t = 0
        while True:
            
            print("*** STRUCTURE SEARCHING [%d] ***" %t)
            # --- finding the best parameters ---
            mu_Zs_pos_lis, U, W, PHI, THETA = self.mu_Zs_pos_lis, self.U, self.W, self.PHI, self.THETA  # initialization
            for r in range(R):
                print("--- PARAMETERS UPDATING:ROUND[%d] ---" %r)
                obj, U, W, PHI, THETA, mu_Zs_pos_lis = \
                    update_parameters(max_iter, cri, self.stu_exe, self.q_matrix, self.latent_set, self.dim, self.nm, mu_Zs_pos_lis, U, W, PHI, THETA)
                if self.obj < obj:  # choose the highest Q function
                    self.U, self.W, self.PHI, self.THETA, self.mu_Zs_pos_lis = U, W, PHI, THETA, mu_Zs_pos_lis  # fine the best parameters
                    self.obj = obj
            
            # --- searching candidate structures among latent variables ---
            gauss_net_list = dict()  # the candidate structures set
            latent_num = self.q_matrix.shape[1]
            # walk through the lower triangle of the neighbor matrix
            id = 0
            for j in range(latent_num):
                for i in range(j+1, latent_num):

                    gs = gaus_net.gauss_net(id, self.q_matrix, self.dim)

                    nm_temp = self.nm.copy()

                    # randomly choose an operation (add, delete or reverse) for every edge
                    if nm_temp[i][j] == 0:  # there is no edge
                        opt = "add"
                    elif nm_temp[i][j] == 1:
                        opt = choice(["reverse", "delete"])
                    elif nm_temp[i][j] == -1:
                        opt = choice(["reverse", "delete"])

                    if opt == 'add':
                        nm_temp[i][j] = 1  # set j as the father of i
                        nm_temp[j][i] = nm_temp[i][j] * -1
                    elif opt == 'reverse':
                        nm_temp[i][j] = nm_temp[i][j] * -1
                        nm_temp[j][i] = nm_temp[i][j] * -1
                    elif opt == 'delete':
                        nm_temp[i][j] = 0
                        nm_temp[j][i] = 0

                    # check whether the current structure is directed acyclic graph
                    is_dag = check_dag(nm_temp)
                    if not is_dag:
                        continue

                    gs.nm = nm_temp
                    # initialize the weight matrix
                    edges = np.argwhere(gs.nm == 1).tolist()
                    for edge in edges:
                        gs.W[edge[0]][edge[1]] = np.random.rand()
                    for k in range(latent_num):
                        if np.all(gs.W[k] == 0):
                            gs.W[k][k] = 1
                    
                    gauss_net_list[id] = gs  # add to the candidate structure set

                    id += 1
            
            # --- find the best candidate structure ---
            obj_best = -np.inf
            id_best = 0
            for id, gs in gauss_net_list.items():
                print("--- CANDIDATE STRUCTURE ID [%d]" %id)
                gs.obj, gs.U, gs.W, gs.PHI, gs.THETA, gs.mu_Zs_pos_lis = \
                    update_parameters(max_iter, cri, self.stu_exe, self.q_matrix, self.latent_set, self.dim, gs.nm, gs.mu_Zs_pos_lis, gs.U, gs.W, gs.PHI, gs.THETA)
                if obj_best < gs.obj:  # the best candidate net
                    id_best = id
                    obj_best = gs.obj
            gs_best = gauss_net_list[id_best]
            
            # --- comparing based on the scoring function --- 
            D = self.stu_exe.shape[1]
            
            str_cost_self = (self.nm.shape[0]*self.nm.shape[1] - np.sum(self.nm == 0)) / 2
            # score_self = cal_rss(self.stu_exe, self.mu_Zs_pos_lis, self.U, self.W, self.PHI, self.THETA) + 0.5 * str_cost_self * math.log(D, 2)
            score_self = self.obj - 0.5*str_cost_self*math.log(D, 2)

            str_cost_cand = (gs_best.nm.shape[0]*gs_best.nm.shape[1] - np.sum(gs_best.nm == 0)) / 2
            # score_cand = cal_rss(self.stu_exe, gs.mu_Zs_pos_lis, gs.U, gs.W, gs.PHI, gs.THETA) + 0.5 * str_cost_cand * math.log(D, 2)
            score_cand = gs_best.obj - 0.5*str_cost_cand*math.log(D, 2)

            if score_cand > score_self:  # replace the best structure
                self.nm = gs_best.nm
                self.U, self.W, self.PHI, self.THETA, self.mu_Zs_pos_lis = gs_best.U, gs_best.W, gs_best.PHI, gs_best.THETA, gs.mu_Zs_pos_lis
                self.obj = gs.obj
            else:
                return self.nm
            
            t += 1


def check_dag(nm):
    """
    Function: check whether the structure is directed acyclic graph based on Kahn algorthm

    Inputs:
    -----------------
    :param nm --> numpy.ndarray()

    Outputs:
    -----------------
    :return is_dag --> boolean
    """

    is_dag = None
    row = nm.shape[1]
    in_degrees = dict()  # the in-degree for all latents
    for _ in range(row):
        in_degrees[_] = len(np.argwhere(nm[_] == 1))
    
    Q = [x for x in in_degrees if in_degrees[x] == 0]  # extract the latent of which in-degree is zero
    seq = []  # the topological sort list
    while Q:
        x = Q.pop()  # pop the latent to be deleted
        seq.append(x)  # add to the topological sort list
        # find the children of x
        children = np.argwhere(nm[x] == -1).tolist()
        for child in children:
            c = child[0]
            in_degrees[c] -= 1  # delete the edge
            if in_degrees[c] == 0:
                Q.append(c)
    
    if len(seq) == row:
        is_dag = True
    else:
        is_dag = False

    return is_dag


def update_parameters(max_iter, cri, stu_exe, q_matrix, latent_set, dim, nm, mu_Zs_pos_lis, U, W, PHI, THETA):
    """
    FUNCTION: Updating parameters, including:
        1. loading matrix:  U
        2. weight matrix:   W
        3. mean vector:     PHI
        4. variance vector: THETA
    
    Inputs:
    -----------------
    :param max_iter --> int
        maximum iterations of parameters estimation for a particular structure
            
    :param cri --> float
        termination cirteria
    
    :param stu_exe --> numpy.ndarray()
        the Student-Exercise Data Matrix
        row : exercise
        col : student
    
    :param q_matrix --> numpy.ndarray()
        The Q-Matrix
        row : exercise
        col : knowledge concept
    
    :param latent_set --> dict()
        the set of CLASS of latent variables
    
    :param dim --> int
        the dimensionality of the latent variable
    
    :param nm --> numpy.ndarray()
        the neighbor matrix of latent variables
    
    :param mu_Zs_pos_lis --> dict()
        the posterior mean conditioned on observations
    
    :param U --> numpy.ndarray()
    
    :param W --> numpy.ndarray()

    :param PHI --> numpy.ndarray()

    :param THETA --> numpy.ndarray()

    Outputs:
    -----------------
    :return objective value

    :return U

    :return W

    :return PHI

    :return THETA

    :return mu_Zs_pos_lis --> dict()
    """

    obs_num, D =  stu_exe.shape  # the number of observed variables (exercises) and samples (students), respectively
    latent_num = q_matrix.shape[1]  # the number of latent variables (knowledge concepts)

    i = 0
    convergence = False  # FLAG deciding whether stop iteration
    objective_value = -np.inf  # initializing the Objective Value

    while(not convergence) and (i < max_iter):
        # calculating the joint marginal distribution of the set of latent variables
        mu_Zs, sigma_Zs = modules.cal_marginal_distribution(latent_set, dim, nm, W)

        # calculate the posterior (partial) covariance of the latent variables conditioned on observations
        A = np.dot(U, W)
        sigma_Zs_pos = np.linalg.inv(np.linalg.inv(sigma_Zs) + np.dot(np.dot(A.T, np.linalg.inv(np.diag(THETA))), A))   # the partial covariance
        # calculate the posterior mean of the latent variables conditioned on observations
        for d in range(D):
            obs = stu_exe[:,d].reshape(obs_num, 1)
            mu_Zs_cond_obs = modules.cal_latent_cond_obs(obs, mu_Zs, sigma_Zs, sigma_Zs_pos, U, PHI, THETA, W)
            mu_Zs_pos_lis[d] = mu_Zs_cond_obs  # storing the conditional information
        
        # calculating the objective value
        objective_value_new = modules.cal_objective_value(stu_exe, q_matrix, mu_Zs_pos_lis, sigma_Zs_pos, U, PHI, THETA, dim, latent_num, nm, W)

        # --- updating parameters ---
        # updating THETA
        THETA_new = modules.update_THETA(stu_exe, q_matrix, mu_Zs_pos_lis, sigma_Zs_pos, U, PHI, dim, nm, W)
        # updating PHI
        PHI_new = modules.update_PHI(stu_exe, q_matrix, mu_Zs_pos_lis, U, latent_num, dim, nm, W)
        # updating U
        U_new = modules.update_U(stu_exe, q_matrix, mu_Zs_pos_lis, sigma_Zs_pos, U, PHI, latent_num, dim, nm, W)
        # updating W
        W_new = modules.update_W(stu_exe, q_matrix, mu_Zs_pos_lis, sigma_Zs_pos, U, PHI, THETA, latent_num, dim, nm, W)

        # Whether the convergence condition is reached
        convergence = abs(objective_value_new - objective_value) < cri

        objective_value = objective_value_new  # update the objective value
        # Updating all parameters
        THETA = THETA_new
        PHI = PHI_new
        U = U_new
        W = W_new

        i += 1
        print("Iteration %d = %s" % (i, objective_value))
        if i == max_iter:
            print('Maximum iterations reached.')

    return objective_value, U, W, PHI, THETA, mu_Zs_pos_lis


def cal_rss(stu_exe, mu_Zs_pos_lis, U, W, PHI, THETA):
    """
    FUNCTION: Calculate the residual sum of squares

    Inputs:
    -----------------
    :param stu_exe --> numpy.ndarray
        the Student-Exercise Data Matrix
        row : exercise
        col : student
    
    :param mu_Zs_pos_lis --> dict()
        the posterior mean conditioned on observations
    
    :param U --> numpy.ndarray()

    :param W --> numpy.ndarray()
    
    :param PHI --> numpy.ndarray()

    :param THETA --> numpy.ndarray()

    Outputs:
    -----------------
    :return rss --> float
    """

    obs_num, D = stu_exe.shape
    A = np.dot(U, W)

    rss = 0
    for n in range(obs_num):
        for d in range(D):
            if not np.isnan(stu_exe[n][d]):
                obs_true = stu_exe[n][d]  # true observation
                mean = (np.dot(A[n], mu_Zs_pos_lis[d]) + PHI[n])[0]
                obs_pre = np.random.normal(loc=mean, scale=math.sqrt(THETA[n]))  # predicted value
                rss += math.pow(abs(obs_true - obs_pre), 2)

    return rss