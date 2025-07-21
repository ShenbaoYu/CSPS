# -*- coding: utf-8 -*-
"""
Constructing the GAUSSIAN NETWORK (FACTOR ANALYSIS) model 
    The latent variables: the knowledge concept
    The observed variables: the records of students doing exercises

ALGORITHM: EM Algorithm
"""

import os, sys

from matplotlib import rc_context

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)
from CSPS import gaus_node, modules, spsl
from GaussSEM import modules as sem
from DataPre import data_pre
from Evaluation import evaluation

import numpy as np
import pickle
import logging
import math

class gaussian_network():
    """
    Knowledge concepts guassian network model.
    """

    def __init__(self, stu_exe, q_matrix, dim, rotation) -> None:
        """
        FUNCTION: Initialization

        Inputs:
        -----------------
        :param stu_exe --> numpy.ndarray
            the Student-Exercise Data Matrix
            row : exercise
            col : student
    
        :param q_matrix --> numpy.ndarray
            the Q-Matrix
            row : exercise
            col : knowledge concept
        
        :param dim --> int
            the dimensionality of the latent variable
        
        :param rotation --> string
            the rotation method
        """

        self.stu_exe, self.q_matrix, self.dim, self.rotation = stu_exe, q_matrix, dim, rotation
        
        obs_num, latent_num = q_matrix.shape

        # Initializing each latent variable
        self.latent_var_set = dict()  # the set of CLASS of latent variables
        for _ in range(latent_num):
            self.latent_var_set[_] = gaus_node.gaus_node(_, self.dim)
        
        # Initializing the posterior distrubiton of the latent variables condtioned on observations
        self.mu_Zs_pos_lis = dict()  # the posterior mean
        self.sigma_Zs_pos = None  # the partial covariance 
        
        # initializing the feature matrix,
        # where each entry is the feature vector for each knowledge concept with a particular observed node,
        # For example: U = 
        #    +--------+--------+-------+--------+
        #    | U[1,1] | U[1,2] |  ...  | U[1,K] |
        #    | U[2,1] | U[2,2] |  ...  | U[2,K] |
        #    |  ...   |  ...   |  ...  |  ...   |
        #    | U[N,1] | U[N,2] |  ...  | U[N,K] |
        #    +--------+--------+-------+--------+
        #    
        #    U[n,k] = [Unk[1], Unk[2], ..., Unk[T]]
        q = np.zeros(shape=(obs_num, latent_num*self.dim), dtype=int)
        for i in range(latent_num):
            for j in range(self.dim):
                q[:, i * self.dim + j] = self.q_matrix[:, i]

        # self.U = np.random.rand(obs_num*self.dim, latent_num) * q
        self.U = np.random.rand(obs_num, latent_num*self.dim) * q

        # initializing the exercise feature vector
        self.PHI = np.random.uniform(size=obs_num).reshape(obs_num, 1)  #  # observation equals # exercises

        # Initializing the Guassian noise
        self.THETA = np.random.uniform(size=obs_num)
    

    def train_em(self, max_iter, cri):
        """
        FUNCTION: Estimating the paramters in the GAUSSIAN NETWORK (FACTOR ANALYSIS) model 
                  based on EM algorithm.

        Inputs:
        -----------------        
        :param max_iter --> int
            maximum iterations
        
        :param cri --> float
            termination cirteria
        """

        obs_num, D = self.stu_exe.shape  # the number of observed variables (exercises) and samples (students), respectively
        latent_num = self.q_matrix.shape[1]  # the number of latent variables (knowledge concepts)

        # calculating the joint marginal distribution of the set of latent variables
        mu_Zs, sigma_Zs = modules.cal_marginal_distribution(self.latent_var_set, self.dim)

        """ ---- Iterate over the E-M steps EM Algorithm ---- """
        i = 0
        convergence = False  # FLAG deciding whether stop iteration

        objective_value = -np.inf  # initializing the Objective Value

        best_elbo = -np.inf  #  initializing the ELBO

        while(not convergence) and (i < max_iter):

            # calculating the joint conditional probability distribution of the latent variables conditioned on observations
            self.sigma_Zs_pos = np.linalg.inv(np.linalg.inv(sigma_Zs) + np.dot(np.dot(self.U.T, np.linalg.inv(np.diag(self.THETA))), self.U))   # the partial covariance
            self.mu_Zs_pos_lis = dict()  # the mean conditioned on each observation
            for d in range(D):
                obs = self.stu_exe[:,d].reshape(obs_num, 1)
                mu_Zs_cond_obs = modules.cal_latent_cond_obs(obs,mu_Zs,sigma_Zs,self.sigma_Zs_pos,self.U,self.PHI,self.THETA)
                self.mu_Zs_pos_lis[d] = mu_Zs_cond_obs  # storing the conditional information
            
            # calculating the objective value
            objective_value_new = modules.cal_objective_value(self.stu_exe,self.q_matrix,mu_Zs,self.mu_Zs_pos_lis,self.sigma_Zs_pos,self.U,self.PHI,self.THETA,self.dim,latent_num)

            # updating parameters
            # updating THETA
            THETA_new = modules.update_THETA(self.stu_exe,self.q_matrix,self.mu_Zs_pos_lis,self.sigma_Zs_pos,self.U,self.PHI,self.dim)
            # updating U
            U_new = modules.update_U(self.stu_exe,self.q_matrix,self.mu_Zs_pos_lis,self.sigma_Zs_pos,self.U,self.PHI,latent_num,self.dim)
            # updating PHI
            PHI_new = modules.update_PHI(self.stu_exe,self.q_matrix,self.mu_Zs_pos_lis,self.U,latent_num,self.dim)

            # Whether the convergence condition is reached
            convergence = abs(objective_value_new - objective_value) < cri

            # Updating all parameters
            objective_value = objective_value_new  # update the objective value

            self.THETA = THETA_new
            self.U = U_new
            self.PHI = PHI_new

            i += 1
            # print("Iteration %d = %s" % (i, objective_value))
            if i == max_iter:
                print('Maximum iterations reached.')
            
        # if the dimensionality of the latent variable (T) is larger than 1
        # average the diagonal elements of the T*T sub-matrix of Sigma_{i,j},
        # to represent the covariance value between i and j.
        T = self.dim
        if T > 1:
            _sigma_reshape = np.zeros(shape=(latent_num, latent_num), dtype=float) # initialization
            for i in range(latent_num):
                for j in range(latent_num):
                    sub_m = self.sigma_Zs_pos[i*T:(i+1)*T, j*T:(j+1)*T]
                    _sigma_reshape[i][j] = np.mean(abs(np.diagonal(sub_m)))
            self.sigma_Zs_pos = _sigma_reshape
        
        return
    

    def _rotate(self, components, n_components=None, tol=1e-6):
        "Rotate the factor analysis solution."
        # note that tol is not exposed
        implemented = ("varimax", "quartimax")
        method = self.rotation
        if method in implemented:
            return _ortho_rotation(components.T, method=method,
                                   tol=tol)
        else:
            raise ValueError("'method' must be in %s, not %s"
                             % (implemented, method))
    

    def eval(self, stu_exe_true, mu_Zs_cond_obs_lis, miss_coo):
        """
        FUNCTION: Model Evaluation

        Inputs:
        -----------------
        :param stu_exe_true --> numpy.ndarray
            the true matrix
        
        :param mu_Zs_cond_obs --> dict{d: mu_Zs_cond_obs --> numpy.ndarray(), ...}
            the conditional mean vector of the probability distribution of the joint latent variables conditioned on observations
        
        :param miss_coo --> tuple()
            the missing coordinates

        Outputs:
        -----------------
        :return rmse --> float
            root mean square error
        """

        rmse = evaluation.rmse(stu_exe_true, self.U, self.PHI, self.THETA, mu_Zs_cond_obs_lis, miss_coo)
        if not rmse is None: 
            print('rmse: %.5f' % rmse)
    
    
    def log_likelihood(self, learned_graph, stu_exe, q_matrix):
        """
        FUNCTION: Calculate the likelihood value for test data

        Inputs:
        -----------------
        :param learned_graph --> dict(skill id : parents ids)

        :param stu_exe --> numpy.ndarray()
            the test matrix
            ROW: exercises
            COL: student
        
        :param q_matrix --> numpy.ndarray()
            the Q matrix

        Outputs:
        -----------------
        :return ll --> float
            the log-ikelihood value
        """

        # build the adjacency matrix
        nm = data_pre.translate_to_nm(learned_graph)

        # calculate the coefficient matrix W
        skill_num = len(learned_graph)
        W = np.zeros(shape=(skill_num, skill_num), dtype=float)
        for sk, parents in learned_graph.items():
            # find the potential Markov blanket of the sk
            pc = self.latent_var_set[sk].neighbors
            pcpc = spsl.find_pcpc(self.latent_var_set[sk], self.latent_var_set, pc)
            if len(parents):
                for pa in parents:
                    W[sk][pa] = cal_reg_coeff(self.sigma_Zs_pos, sk, pa, pcpc)
            else:
                W[sk][sk] = 1
        
        # calculate the likelihood value
        latent_num = q_matrix.shape[1]
        ll = cal_log_likelihood_val(stu_exe, q_matrix, self.mu_Zs_pos_lis, self.sigma_Zs_pos, self.U, self.PHI, self.THETA, self.dim, latent_num, nm, W)
        print('The log-likelihood value is:%.4f' %ll)
        return
    
    def save(self, filepath):
        with open(filepath, 'wb') as file:
            # pickle.dump({"U": self.U, "PHI": self.PHI, "THETA": self.THETA}, file)
            pickle.dump({"mu_Zs_pos_lis": self.mu_Zs_pos_lis, "sigma_Zs_pos": self.sigma_Zs_pos}, file)
            logging.info("save parameters to %s" % filepath)
    

    def load(self, filepath):
        with open(filepath, 'rb') as file:
            # self.U, self.PHI, self.THETA = pickle.load(file).values()
            self.mu_Zs_pos_lis, self.sigma_Zs_pos = pickle.load(file).values()
            logging.info("load parameters from %s" % filepath)


def _ortho_rotation(components, method='varimax', tol=1e-6, max_iter=100):
    """Return rotated components."""
    nrow, ncol = components.shape
    rotation_matrix = np.eye(ncol)
    var = 0

    for _ in range(max_iter):
        comp_rot = np.dot(components, rotation_matrix)
        if method == "varimax":
            tmp = comp_rot * np.transpose((comp_rot ** 2).sum(axis=0) / nrow)
        elif method == "quartimax":
            tmp = 0
        u, s, v = np.linalg.svd(
            np.dot(components.T, comp_rot ** 3 - tmp))
        rotation_matrix = np.dot(u, v)
        var_new = np.sum(s)
        if var != 0 and var_new < var * (1 + tol):
            break
        var = var_new

    return np.dot(components, rotation_matrix).T


def cal_reg_coeff(covariance, dv, iv, pcpc):
    """
    FUNCTION: Calculate the regression coefficient between depedent variable and independent variable

    Inputs:
    -----------------
    :param covariance --> numpy.ndarray()

    :param dv --> int
        depedent variable

    :param iv --> int
        independent variable

    :param pcpc --> list()
        the potential Markov blanket of dv

    Outputs:
    -----------------
    :return rc --> float
        the regression coefficient
    """

    rc = 0  # initialize
    pcpc.append(dv)
    # calculate the partial covariance between dv and iv
    par_cov, cond_num = spsl.cal_partial_covariance(dv, iv, pcpc, covariance)
    # calculate the partial variance of iv
    pcpc.remove(dv)
    pcpc.remove(iv)
    par_var = spsl.cal_partial_variance(iv, pcpc, covariance)
    # calculate the regression coefficient
    rc = par_cov[0][1] / par_var

    return rc


def cal_log_likelihood_val(stu_exe, q_matrix, mu_Zs_pos_lis, sigma_Zs_pos, U, PHI, THETA, dim, latent_num, nm, W):
    """
    FUNCTION: Calculating the Objective Value.

    Inputs:
    -----------------
    :param stu_exe --> numpy.ndarray
        the Student-Exercise Data Matrix
        row : exercise
        col : student
    
    :param q_matrix --> numpy.ndarray
        The Q-Matrix
        row : exercise
        col : knowledge concept
    
    :param mu_Zs_pos_lis --> dict{d: mu_Zs_cond_obs --> numpy.ndarray(), ...}
        the conditional mean vector of the probability distribution of the joint latent variables conditioned on observations

    :param sigma_Zs_pos --> numpy.ndarray()
        the conditional partial covariance matrix of the probability distribution of the joint latent variables conditioned on observations
    
    :param U --> numpy.ndarray
    
    :param PHI --> numpy.ndarray

    :param THETA --> numpy.ndarray
        the variance of the Gaussian noise epsilon
    
    :param dim --> int
        the dimensionality of the latent variable
    
    :param latent_num --> int
        the number of latent variables

    Outputs:
    -----------------
    :return likelihood value --> float
    """

    ll =-np.inf  # initialization
    obs_num, D = stu_exe.shape  # the number of observed nodes (exercises) and samples (students), respectively

    mu_Zs_pos_mean = np.zeros(shape=(latent_num, 1), dtype=float)  # the average expectation value
    for mu_Zs_pos in mu_Zs_pos_lis.values():
        mu_Zs_pos_mean += mu_Zs_pos
    mu_Zs_pos_mean = mu_Zs_pos_mean / len(mu_Zs_pos_lis)

    # 1. Σ(d)Σ(n) logθn
    sum_theta_n = 0
    # 2. Σ(d)Σ(n) 0.5*(1/θn)*E[Δ]
    sum_e_del = 0
    B = 0
    for d in range(D):
         for n in range(obs_num):
             B += 1
             sum_theta_n += math.log(THETA[n], 2)
             sum_e_del += 0.5 * (1/THETA[n]) * sem.cal_delta_obs_d(n, stu_exe[n][d], U, PHI[n][0], q_matrix, mu_Zs_pos_mean, sigma_Zs_pos, dim, nm, W)
    
    # 3. calculate Σ(k) log𝜔(k)
    # 4. calculate Σ(k) 1/𝜔(k) * E[Zk.T, Zk]
    sum_log_omega = 0  # Σ(k) log𝜔(k)
    sum_omega_e_ztz = 0  # Σ(k) 1/𝜔(k) * E[Zk.T, Zk]
    for k in range(latent_num):
        omega_k = sem.cal_omega_k(k, latent_num, nm, W)
        e_ztz = sem.cal_e_ztz_cond_obs(k, mu_Zs_pos_lis, sigma_Zs_pos, dim, len(mu_Zs_pos_lis))
        sum_log_omega += math.log(omega_k, 2)
        sum_omega_e_ztz += 1/omega_k * e_ztz
    
    # 5. calculate the final likelihood value
    ll = 1/D * (- 0.5*B*math.log(2*math.pi, 2) - 0.5*sum_theta_n - sum_e_del) \
         - 0.5*latent_num*dim*math.log(2*math.pi, 2) - 0.5*dim*sum_log_omega - 0.5*sum_omega_e_ztz
    
    return ll