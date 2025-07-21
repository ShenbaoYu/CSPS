"""
modules in gaussian network
"""

import numpy as np
import math


def cal_marginal_distribution(latent_var_set, dim):
    """
    FUNCTION: Calculating the joint marginal distribution of the latent variables,
    including:
    1. the mean vector where each entry represents the mean of a particular latent variable.
    2. the covariance matrix where each entry denotes the covariance for a pair of latent variables.

    Inputs:
    -----------------
    :param latent_var_set --> dict(latent_var_id : Gaussian node --> class, ...)
        the set of CLASS of latent variables
        key   : id
        value : the CLASS of latent variable
    
    :param dim --> int
        the dimensionality of the latent variable
    
    Outputs:
    -----------------
    :return mu_Zs --> numpy.ndarray
        the mean vector of joint latent variables marginal probability distribution

    :return sigma_Zs --> numpy.ndarray
        the covariance matrix joint latent variables marginal probability distribution

    ATTENTION: 
    The default index in mu_Zs represents the corresponding ID of the latent variable.
    """

    
    # Becasuse the number of dimensions for each latent variable is T,
    # So:
    #
    # 1. We set the size of mu_Zs is: (T*K) * 1
    # For example:
    # mu_Zs = [[0,0,...,0] [0,0,...,0] ... [0,0,...0]],
    # which means there are K [0,0,...,0], the size of each of which is T, 
    # and the index of Zk in mu_Zs is: [k*T, (k+1)*T-1], such as,
    # Z0 = mu_Zs[0] to mu_Zs[T-1],
    # Z1 = mu_Zs[T] to mu_Zs[2T-1], ..., and so on
    #
    # 2. The size of sigma_Zs is: (T*K) * (T*K)

    # initialize
    mu_Zs = np.zeros(shape=[dim*len(latent_var_set), 1], dtype=float)  
    sigma_Zs = np.zeros(shape=[dim*len(latent_var_set), dim*len(latent_var_set)], dtype=float)

    for k, zk in latent_var_set.items():
        mu_Zs[k*dim:(k+1)*dim] = zk.mu
        sigma_Zs[k*dim:(k+1)*dim, k*dim:(k+1)*dim] = zk.sigma

    return mu_Zs, sigma_Zs


def cal_W(q_matrix, U, dim):
    """
    Calculate the sparse regression matrix
    
    Inputs:
    -----------------
    :param q_matrix --> numpy.ndarray
        The Q-Matrix
        row : exercise
        col : knowledge concept
    
    :param U --> numpy.ndarray

    :param dim --> int
        the dimensionality of the latent variable
    
    Outputs:
    -----------------
    :return W

    """
    # calculate the sparse regression matrix --> W
    # For example:
    #   Z1  Z2
    #   /\  /\
    # O1  O2  O3
    # W =
    # +--------+--------+
    # |  U1.T  |  0     |
    # +--------+--------+
    # |  U1.T  |  U2.T  |
    # +--------+--------+
    # |  0     |  U2.T  |
    # +--------+--------+
    obs_num, latent_num = q_matrix.shape
    W = np.zeros(shape=[obs_num, latent_num*dim], dtype=float)  # initialization

    for o in range(obs_num):
        for k in range(latent_num):
            if q_matrix[o][k] == 1:
                # Unk = U[o*dim:(o+1)*dim, k].reshape(dim, 1)
                Unk = U[o, k*dim:(k+1)*dim].reshape(dim, 1)
                for t in range(dim):
                    W[o][k*dim + t] = Unk[t][0]
    
    return W


def cal_latent_cond_obs(obs, mu_Zs, sigma_Zs, sigma_Zs_cond_obs, U, PHI, THETA):
    """
    FUNCTION: Calculating the joint conditional probability distribution of latent variables based on (d-th) observation via precision matrix.

    Inputs:
    -----------------
    :param observation --> numpy.ndarray
        the d-th column of the Student-Exercise Data Matrix
        row : exercise
        col : student
    
    :param mu_Zs --> numpy.ndarray
        the mean vector of joint latent variables marginal probability distribution

    :param sigma_Zs --> numpy.ndarray
        the covariance matrix joint latent variables marginal probability distribution

    :param sigma_Zs_cond_obs --> numpy.ndarray
        the partial covariance matrix conditioned on observations
    
    :param U --> numpy.ndarray
    
    :param PHI --> numpy.ndarray

    :param THETA --> numpy.ndarray
        the variances of the Gaussian noise epsilon
    
    Outputs:
    -----------------
    :return mu_Zs_cond_obs --> numpy.ndarray()
        the conditional mean vector of the probability distribution of the joint latent variables conditioned on a particular observation
    """

    obs_num = len(obs)
    index = []
    if np.isnan(obs).any():
        for _ in range(obs_num):
            if np.isnan(obs[_][0]):
                index.append(_)
    
    obs_cut = np.delete(obs, index).reshape(obs_num - len(index), 1)
    U_cut = np.delete(U, index, axis=0)
    PHI_cut = np.delete(PHI, index).reshape(obs_num - len(index), 1)
    THETA_cut = np.delete(THETA, index)

    mu_Zs_cond_obs = np.dot(sigma_Zs_cond_obs, np.dot(np.dot(U_cut.T, np.linalg.inv(np.diag(THETA_cut))), obs_cut-PHI_cut) 
                            + np.dot(np.linalg.inv(sigma_Zs), mu_Zs))

    return mu_Zs_cond_obs


def cal_objective_value(stu_exe, q_matrix, mu_Zs, mu_Zs_cond_obs_lis, sigma_Zs_cond_obs, U, PHI, THETA, dim, latent_num):
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
    
    :param mu_Zs --> numpy.ndarray
        the mean vector of joint latent variables marginal probability distribution
    
    :param mu_Zs_cond_obs_lis --> dict{d: mu_Zs_cond_obs --> numpy.ndarray(), ...}
        the conditional mean vector of the probability distribution of the joint latent variables conditioned on observations

    :param sigma_Zs_cond_obs --> numpy.ndarray()
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
    :return objective value --> float
    """

    _objective_value = -np.inf

    obs_num, D = stu_exe.shape  # the number of observed nodes (exercises) and samples (students), respectively

    # 1. calculate E[np.dot(Z.T,Z)], where Z~g(Z|O)
    exp_ZTZ = cal_exp_ZTZ_cond_obs(mu_Zs_cond_obs_lis, sigma_Zs_cond_obs, dim, D, latent_num)
    # 2. calculate Σ(d)np.dot(μ(Z).T, E[Z])
    mu_expZ = cal_mu_expZ(mu_Zs, mu_Zs_cond_obs_lis, D)

    # 3. calculate Σ(d)Σ(n⊂I(d)) E[Δ] = Σ(d)Σ(n⊂I(d))E[math.pow(Ond - Σ(k)δ(nk)np.dot(U(nk).T,Zk)+PHI(n))], where Z~g(Z|O)
    # 4. calculate Σ(d)Σ(n⊂I(d)) logθn
    sum_theta_n = 0
    _B = 0
    e_delta = 0  # 0.5*(1/θn)*E[Δ]
    for d in range(D):  # walk through all samples
        for n in range(obs_num):
            if not np.isnan(stu_exe[n][d]):
                _B += 1
                sum_theta_n += math.log(THETA[n], 2)
                e_delta += 0.5 * (1/THETA[n]) * cal_delta_obs_d(n, stu_exe[n][d], U, PHI[n][0], q_matrix, mu_Zs_cond_obs_lis[d], sigma_Zs_cond_obs, dim)
    
    # 5. calculate the final objective value
    _objective_value = - 0.5 * _B * obs_num * math.log(2*math.pi, 2) \
                       - 0.5 * sum_theta_n \
                       - e_delta \
                       - 0.5 * latent_num * dim * math.log(2*math.pi, 2) \
                       - 0.5 * exp_ZTZ \
                       +       mu_expZ \
                       - 0.5 * np.dot(mu_Zs.T, mu_Zs)[0][0]
    
    return _objective_value


def cal_exp_ZTZ_cond_obs(mu_Zs_cond_obs_lis, sigma_Zs_cond_obs, dim, D, latent_num):
    """ 
    FUNCTION: Calculate E[np.dot(Z.T,Z)], where Z~g(Z|O)

    Inputs:
    -----------------
    :param mu_Zs_cond_obs_lis --> dict{d: mu_Zs_cond_obs --> numpy.ndarray(), ...}
        the conditional mean vector of the probability distribution of the joint latent variables conditioned on observations

    :param sigma_Zs_cond_obs --> numpy.ndarray()
        the conditional partial covariance matrix of the probability distribution of the joint latent variables conditioned on observations
    
    :param dim --> int
        the dimensionality of the latent variable
    
    :param D --> int
        the number of samples
    
    :param latent_num --> int
        the number of latent variables

    Outputs:
    -----------------
    :return Σ(d)E[np.dot(Z.T,Z)], where Z~g(Z|O)
    """

    # ATTENTION:
    # Because Z = [Z1, Z2, ..., ZK], So we have:
    # E[Z.T,Z] = E[Z1.T,Z1] + ... + E[Zk.T,Zk] + ... + E[ZK.T,ZK],
    # where E[Zk.T,Zk] = E[math.pow(Zk1,2)] + ... + E[math.pow(ZkT,2)],
    # where E[math.pow(Zki,2)] = var(Zki) + math.pow(E[Zki], 2).

    # NOTE: Because Z~g(Z|Od), where d is the d-th observation, so
    #       we average the conditional expectation.

    exp_ZTZ = 0   # E[Z.T,Z]

    for d in range(D):  # walk through all observations
        exp_ZsZs = 0  # initialize E[Z.T,Z] for d-th observation
        for k in range(latent_num):
            exp_ZkZk = 0  # initialize E[np.dot(Zk.T,Zk)]
            # get the conditional mean vector of Zk based on d-th observation
            mu_Zk_d = mu_Zs_cond_obs_lis[d][k*dim:(k+1)*dim]
            # get the partial covariance matrix of Zk
            sigma_Zk_d = sigma_Zs_cond_obs[k*dim:(k+1)*dim, k*dim:(k+1)*dim]
            for t in range(dim):
                exp_Zki2 = sigma_Zk_d[t][t] + math.pow(mu_Zk_d[t], 2)
                exp_ZkZk += exp_Zki2
            
            exp_ZsZs += exp_ZkZk
        exp_ZTZ += exp_ZsZs
    
    return exp_ZTZ / D


def cal_mu_expZ(mu_Zs, mu_Zs_cond_obs_lis, D):
    """
    FUNCTION: Calculate μ(Z)E[Z]

    Inputs:
    -----------------
    :param mu_Zs --> numpy.ndarray
        the mean vector of joint latent variables marginal probability distribution
    
    :param mu_Zs_cond_obs_lis --> dict{d: mu_Zs_cond_obs --> numpy.ndarray(), ...}
        the conditional mean vector of the probability distribution of the joint latent variables conditioned on observations
    
    :param D --> int
        the number of samples

    Outputs:
    -----------------
    :return μ(Z)E[Z]
    """

    mu_expZ = 0  # initialization

    for d in range(D):
        mu_expZ += np.dot(mu_Zs.T, mu_Zs_cond_obs_lis[d])[0][0]

    return mu_expZ / D


def cal_delta_obs_d(n, obs, U, PHI_n, q_matrix, mu_Zs_cond_obs, sigma_Zs_cond_obs, dim):
    """
    FUNCTION: Calculating the EXCEPTION of the error of the observation: E[math.pow(Ond - Σ(k)δ(nk)np.dot(U(nk).T,Zk)+PHI(n))], where Z~g(Z|O) 
    
    Inputs:
    -----------------
    :param n --> int
        the n-th observation
    
    :param obs --> float
        the observed value
    
    :param U --> numpy.ndarray

    :param PHI_n --> float
        the n-th of PHI
    
    :param q_matrix --> numpy.ndarray
        the Q matrix
    
    :param mu_Zs_cond_obs --> numpy.ndarray()
        the conditional mean vector of the probability distribution of the joint latent variables conditioned on observations
    
    :param sigma_Zs_cond_obs --> numpy.ndarray()
        the conditional partial covariance matrix of the probability distribution of the joint latent variables conditioned on observations
    
    :param dim --> int
        the dimensionality of the latent variable

    Outputs:
    -----------------
    :return delta_obs_d
    """

    delta_obs_d = 0

    # initialzation
    sum_u_exp_Z = 0  # Σ(k)δ(nk)U(nk).TE[Zk]
    sum_u_exp_Z_u = 0  # Σ(i)Σ(j)δ(nij)U(ni).TE[Zi Zj.T]U(nj)

    latent_num = q_matrix.shape[1]

    for ki in range(latent_num):
        if q_matrix[n][ki] == 1:
            
            """ 1. calculate Σ(k)δ(nk)U(nk).TE[Zk] """
            # Uni = U[n*dim:(n+1)*dim, ki].reshape(dim, 1)
            Uni = U[n, ki*dim:(ki+1)*dim].reshape(dim, 1)  # get the feature vector of i-th latent variable with n-th observation
            mu_Zi = mu_Zs_cond_obs[ki*dim:(ki+1)*dim]  # get the conditional mean vector of i-th latent variable
            sum_u_exp_Z += np.dot(Uni.T, mu_Zi)[0][0]  # cumulate calculations

            for kj in range(latent_num):
                if q_matrix[n][kj] == 1:
                    """ 2. calculate Σ(i)Σ(j)δ(nij)U(ni).TE[Zi Zj.T]U(nj) """
                    # Unj = U[n*dim:(n+1)*dim, kj].reshape(dim, 1)
                    Unj = U[n, kj*dim:(kj+1)*dim].reshape(dim, 1)  # get the feature vector of j-th latent variable with n-th observation
                    mu_Zj = mu_Zs_cond_obs[kj*dim:(kj+1)*dim] # get the conditional mean vector of j-th latent variable
                    # 2.1 calculate the E[Zi, Zj.T]
                    sigma_Zi_Zj = sigma_Zs_cond_obs[ki*dim:(ki+1)*dim, kj*dim:(kj+1)*dim]
                    exp_Zi_Zj = sigma_Zi_Zj + np.dot(mu_Zi, mu_Zj.T)
                    # 2.2 cumulate calculations
                    sum_u_exp_Z_u += np.dot(np.dot(Uni.T, exp_Zi_Zj), Unj)[0][0]
    
    delta_obs_d = math.pow(obs, 2) - 2*obs*PHI_n + math.pow(PHI_n, 2) - 2*obs*sum_u_exp_Z + 2*PHI_n*sum_u_exp_Z + sum_u_exp_Z_u

    return delta_obs_d


def update_THETA(stu_exe, q_matrix, mu_Zs_cond_obs_lis, sigma_Zs_cond_obs, U, PHI, dim):
    """
    FUNCTION: Updating THETA

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
    
    :param mu_Zs_cond_obs_lis --> dict{d: mu_Zs_cond_obs --> numpy.ndarray(), ...}
        the conditional mean vector of the probability distribution of the joint latent variables conditioned on observations
    
    :param sigma_Zs_cond_obs --> numpy.ndarray()
        the conditional partial covariance matrix of the probability distribution of the joint latent variables conditioned on observations
    
    :param U --> numpy.ndarray
    
    :param PHI --> numpy.ndarray

    :param dim --> int
        the dimensionality of the latent variable

    Outputs:
    -----------------
    :return THETA 
    """

    obs_num, D = stu_exe.shape  # the number of observed nodes (exercises) and samples (students), respectively

    THETA_new = np.zeros(shape=[obs_num], dtype=float)
    _C = np.zeros(shape=[obs_num], dtype=int)
    for d in range(D):
        for n in range(obs_num):
            if not np.isnan(stu_exe[n][d]):
                _C[n] += 1
                e_delta = cal_delta_obs_d(n, stu_exe[n][d], U, PHI[n][0], q_matrix, mu_Zs_cond_obs_lis[d], sigma_Zs_cond_obs, dim)
                THETA_new[n] += e_delta

    return (1/_C) * THETA_new


def update_U(stu_exe, q_matrix, mu_Zs_cond_obs_lis, sigma_Zs_cond_obs, U, PHI, latent_num, dim):
    """
    FUNCTION: Updating U

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
    
    :param mu_Zs_cond_obs_lis --> dict{d: mu_Zs_cond_obs --> numpy.ndarray(), ...}
        the conditional mean vector of the probability distribution of the joint latent variables conditioned on observations

    :param sigma_Zs_cond_obs --> numpy.ndarray()
        the conditional partial covariance matrix of the probability distribution of the joint latent variables conditioned on observations
    
    :param U --> numpy.ndarray

    :param PHI --> numpy.ndarray

    :param latent_num --> int
        the number of latent variables
    
    :param dim --> int
        the dimensionality of the latent variable

    Outputs:
    -----------------
    :return U
    """

    obs_num, D = stu_exe.shape  # the number of observed nodes (exercises) and samples (students), respectively

    # U_new = np.zeros(shape=[obs_num*dim, latent_num], dtype=float)
    U_new = np.zeros(shape=[obs_num, latent_num*dim], dtype=float)

    for n in range(obs_num):
        for k in range(latent_num):
            if q_matrix[n][k] == 1:
                # initialization
                sum_ZkZk = 0        # Σ(d)δ(nd)δ(nk) E[ZkZk.T]
                sum_obs_expZk = 0   # Σ(d)δ(nd)δ(nk) Σ(d)Oid*E[Zk]
                sum_PHIn_expZk = 0  # Σ(d)δ(nd)δ(nk) Phin*E[Zk]
                sum_ZkZj_u= 0       # Σ(d)δ(nd)Σ(k)δ(nkj) E[ZkZj.T]Unj

                for d in range(D):
                    if not np.isnan(stu_exe[n][d]):
                        # get the conditional mean vector of Zk with the d-th sample
                        mu_Zk = mu_Zs_cond_obs_lis[d][k*dim:(k+1)*dim]
                        
                        """ 1. cumulate Σ(d)δ(nk)δ(nd) E[ZkZk.T] """
                        # get the partial covariance matrix between Zk and Zk
                        sigma_ZkZk = sigma_Zs_cond_obs[k*dim:(k+1)*dim, k*dim:(k+1)*dim]
                        # calculate E[ZkZk.T]
                        exp_ZkZk = sigma_ZkZk + np.dot(mu_Zk, mu_Zk.T)
                        # cumulative calculation
                        sum_ZkZk += exp_ZkZk
                    
                        """ 2. cumulate Σ(d)δ(nd)δ(nk) Σ(d)OndE[Zk] """
                        sum_obs_expZk += stu_exe[n][d]*mu_Zk

                        """ 3. cumulate Σ(d)δ(nd)δ(nk) Phi_nE[Zk] """
                        sum_PHIn_expZk += PHI[n][0]*mu_Zk

                        for j in range(latent_num):
                            if q_matrix[n][j] == 1 and k != j:
                                """ 4. cumulate Σ(d)δ(nd)Σ(k)δ(nkj) E[ZkZj.T]Unj """
                                # get the partial covariance matrix between Zk and Zj
                                sigma_ZkZj = sigma_Zs_cond_obs[k*dim:(k+1)*dim, j*dim:(j+1)*dim]
                                # get the conditional mean vector of Zj
                                mu_Zj = mu_Zs_cond_obs_lis[d][j*dim:(j+1)*dim]
                                # calculate E[ZkZj.T]
                                exp_ZkZj = sigma_ZkZj + np.dot(mu_Zk, mu_Zj.T)
                                # get the feature vector of j-th latent variable with the n-th observation
                                # Unj = U[n*dim:(n+1)*dim, j].reshape(dim, 1)
                                Unj = U[n, j*dim:(j+1)*dim].reshape(dim, 1)
                                # cumulative calculation
                                sum_ZkZj_u += np.dot(exp_ZkZj, Unj)

                # U_new[n*dim:(n+1)*dim, k] = np.dot(np.linalg.inv(sum_ZkZk), sum_obs_expZk - sum_PHIn_expZk - sum_ZkZj_u).reshape(dim)
                U_new[n, k*dim:(k+1)*dim] = np.dot(np.linalg.inv(sum_ZkZk), sum_obs_expZk - sum_PHIn_expZk - sum_ZkZj_u).reshape(dim)

    return U_new


def update_PHI(stu_exe, q_matrix, mu_Zs_cond_obs_lis, U, latent_num, dim):
    """
    FUNCTION: Updating PHI

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
    
    :param mu_Zs_cond_obs_lis --> dict{d: mu_Zs_cond_obs --> numpy.ndarray(), ...}
        the conditional mean vector of the probability distribution of the joint latent variables conditioned on observations

    :param U --> numpy.ndarray

    :param latent_num --> int
        the number of latent variables
    
    :param dim --> int
        the dimensionality of the latent variable

    Outputs:
    -----------------
    :return PHI
    """

    obs_num, D = stu_exe.shape  # the number of observed nodes (exercises) and samples (students), respectively
    PHI_new = np.zeros(shape=[obs_num, 1], dtype=float)

    for n in range(obs_num):
        # Initialization
        sum_obs = 0  # Σ(d)δ(nd) Oid
        sum_u_exp_Z = 0  # Σ(d)δ(nd)Σ(k)δ(nk)U(nk).TE[Zk]
        _C = 0
        for d in range(D):
            if not np.isnan(stu_exe[n][d]):
                _C += 1
                
                """ 1. cumulate Σ(d)δ(nd) Oid """
                sum_obs += stu_exe[n][d]
            
                for k in range(latent_num):
                    if q_matrix[n][k] == 1:
                        """ 2. cumulate Σ(d)δ(nd)Σ(k)δ(nk)U(nk).TE[Zk] """
                        # Unk = U[n*dim:(n+1)*dim, k].reshape(dim, 1)
                        Unk = U[n, k*dim:(k+1)*dim].reshape(dim, 1)  # get the feature vector of k-th latent variable with n-th observation
                        mu_Zk = mu_Zs_cond_obs_lis[d][k*dim:(k+1)*dim]  # get the conditional mean vector of k-th latent variable with the d-th sample
                        sum_u_exp_Z += np.dot(Unk.T, mu_Zk)[0][0]  # cumulate calculations
        
        PHI_new[n][0] = (sum_obs - sum_u_exp_Z) / _C

    return PHI_new


def normalization(matrix):
    """
    FUNCTION: Matrix normalization

    Inputs:
    -----------------
    :param matrix --> numpy.array()
        the matrix to be normalized
    
    Outputs:
    -----------------
    :return matrix_norm
    """
    row, col = matrix.shape
    matrix_norm = np.zeros(shape=[row, col], dtype=float)

    _has = matrix[matrix != 0]
    mu = np.mean(_has)
    mx = np.max(_has)
    mn = np.min(_has)

    el = np.argwhere(matrix > 0)
    for _ in el:
        matrix_norm[_[0]][_[1]] = (matrix[_[0]][_[1]] - mu) / (mx - mn)

    return matrix_norm