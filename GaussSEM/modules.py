import numpy as np
import math


def cal_marginal_distribution(latent_set, dim, nm, W):
    """
    FUNCTION: Calculating the joint marginal distribution of the latent variables,
    including:
    1. the mean vector where each entry represents the mean of a particular latent variable.
    2. the covariance matrix where each entry denotes the covariance for a pair of latent variables.

    Inputs:
    -----------------
    :param latent_set --> dict(latent_var_id : Gaussian node --> class, ...)
        the set of CLASS of latent variables
        key   : id
        value : the CLASS of latent variable
    
    :param dim --> int
        the dimensionality of the latent variable
    
    :param nm --> numpy.ndarray
        the neighbor matrix of latent variables
    
    :param W --> numpy.ndarray
        the weight matrix of latent variables
    
    Outputs:
    -----------------
    :return mu_Zs --> numpy.ndarray
        the mean vector of joint latent variables marginal probability distribution

    :return sigma_Zs --> numpy.ndarray
        the covariance matrix joint latent variables marginal probability distribution

    NOTE: The default index in mu_Zs represents the corresponding ID of the latent variable.
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

    l_num = len(latent_set)

    mu_Zs = np.zeros(shape=[dim*l_num, 1], dtype=float)  # mean vector

    sigma_Zs = np.zeros(shape=[dim*l_num, dim*l_num], dtype=float)  # covariance matrix

    for k, zk in latent_set.items():  # the variance of latent variable k
        omega_k = cal_omega_k(k, l_num, nm, W)
        sigma_Zs[k*dim:(k+1)*dim, k*dim:(k+1)*dim] = omega_k * zk.sigma
    
    # calculate the covariance between i and j (i≠j)
    for i in range(l_num):
        for j in range(i+1, l_num):
            if nm[i][j] == 1 or nm[j][i] == 1:
                sigma_Zs[i*dim:(i+1)*dim, j*dim:(j+1)*dim] = (W[i][j] + W[j][i]) * np.identity(dim)
                sigma_Zs[j*dim:(j+1)*dim, i*dim:(i+1)*dim] = (W[i][j] + W[j][i]) * np.identity(dim)

    return mu_Zs, sigma_Zs


def cal_omega_k(k, l_num, nm, W):
    omega = 1
    for i in range(l_num):
        if nm[k][i] == 1 and i != k:  # i is the parent of k
            for j in range(l_num):
                if nm[k][j] == 1 and j != k:  # j is the parent of k
                    if i == j:
                        omega += W[k][i] * W[k][i] * 1
                    elif nm[i][j] == 1 or nm[j][i] == 1:  # i and j are neighbors
                        omega += W[k][i] * W[k][j] * (W[i][j] + W[j][i])
    return omega


def cal_latent_cond_obs(obs, mu_Zs, sigma_Zs, sigma_Zs_cond_obs, U, PHI, THETA, W):
    """
    FUNCTION: Calculating the joint conditional probability distribution of latent variables based on (d-th) observation via precision matrix.

    Inputs:
    -----------------
    :param obs --> numpy.ndarray
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
    
    :param W --> numpy.ndarray
        the weight matrix of latent variables
    
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
    A_cut = np.dot(U_cut, W)
    PHI_cut = np.delete(PHI, index).reshape(obs_num - len(index), 1)
    THETA_cut = np.delete(THETA, index)

    mu_Zs_cond_obs = np.dot(sigma_Zs_cond_obs, np.dot(np.dot(A_cut.T, np.linalg.inv(np.diag(THETA_cut))), obs_cut-PHI_cut) 
                            + np.dot(np.linalg.inv(sigma_Zs), mu_Zs))
    
    return mu_Zs_cond_obs


def cal_objective_value(stu_exe, q_matrix, mu_Zs_cond_obs_lis, sigma_Zs_cond_obs, U, PHI, THETA, dim, latent_num, nm, W):
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

    # 1. calculate Σ(d)Σ(n⊂I(d)) logθn
    sum_theta_n = 0
    # 2. calculate  Σ(d)Σ(n⊂I(d)) 0.5*(1/θn)*E[Δ]
    sum_e_del = 0
    B = 0
    for d in range(D):
        for n in range(obs_num):
            if not np.isnan(stu_exe[n][d]):
                B += 1
                sum_theta_n += math.log(THETA[n], 2)
                sum_e_del += 0.5 * (1/THETA[n]) * cal_delta_obs_d(n, stu_exe[n][d], U, PHI[n][0], q_matrix, mu_Zs_cond_obs_lis[d], sigma_Zs_cond_obs, dim, nm, W)
    
    # 3. calculate Σ(k) log𝜔(k)
    # 4. calculate Σ(k) 1/𝜔(k) * E[Zk.T, Zk]
    sum_log_omega = 0  # Σ(k) log𝜔(k)
    sum_omega_e_ztz = 0  # Σ(k) 1/𝜔(k) * E[Zk.T, Zk]
    for k in range(latent_num):
        omega_k = cal_omega_k(k, latent_num, nm, W)
        e_ztz = cal_e_ztz_cond_obs(k, mu_Zs_cond_obs_lis, sigma_Zs_cond_obs, dim, D)
        sum_log_omega += math.log(omega_k, 2)
        sum_omega_e_ztz += 1/omega_k * e_ztz

    # 5. calculate the final objective value
    _objective_value = - 0.5 * B * math.log(2*math.pi, 2) \
                       - 0.5 * sum_theta_n \
                       - sum_e_del \
                       - 0.5 * latent_num * dim * math.log(2*math.pi, 2) \
                       - 0.5 * dim * sum_log_omega \
                       - 0.5 * sum_omega_e_ztz
    
    return _objective_value


def cal_delta_obs_d(n, obs, U, PHI_n, q_matrix, mu_Zs_cond_obs, sigma_Zs_cond_obs, dim, nm, W):
    """
    FUNCTION: Calculating the EXCEPTION of the error of the observation: E[math.pow(obs - Σ(k)δ(nk)np.dot(U(nk).T,Zk)+PHI(n))], where Z~g(Z|O) 
    
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
    
    :param nm --> numpy.ndarray
        the neighbor matrix of latent variables
    
    :param W --> numpy.ndarray
        the weight matrix of latent variables

    Outputs:
    -----------------
    :return delta_obs_d
    """

    delta_obs_d = 0

    # initialzation
    sum_omega_u_z = 0    # Σ(k)Σ(i≠k) δ(nk)δ(ik) 𝜔(ik)U(nk)E[Zi]
    sum_omega_u_z_u = 0  # Σ(i)Σ(p≠i)Σ(j)Σ(q≠j) δ(nij)δ(pi)δ(qj) 𝜔(pi)𝜔(qj)U(ni)E[Zp Zq]U(nj)

    latent_num = q_matrix.shape[1]

    for i in range(latent_num):
        if q_matrix[n][i] == 1:
            
            """ calculate Σ(i)Σ(_≠i) δ(ni)δ(_i) 𝜔(_i)U(ni)E[Z_]"""
            uni = U[n*dim:(n+1)*dim, i].reshape(dim, 1)  # get the feature vector of i-th latent variable with n-th observation
            
            if np.any(nm[i] == 1):  # i-latent variable has parent
                for _ in range(latent_num):
                    if nm[i][_] == 1 and _ != i:  # _ is the parent of i
                        mu_z_ = mu_Zs_cond_obs[_*dim:(_+1)*dim]  # get the conditional mean vector of _-th latent variable
                        sum_omega_u_z += W[i][_] * np.dot(uni.T, mu_z_)[0][0]
            else:  # if i-latent variable has not any parent, Σ(k)Σ(i≠k) δ(nk)δ(ik) 𝜔(ik)U(nk)E[Zi] = Σ(i) U(ni)E[Zi]
                mu_zi = mu_Zs_cond_obs[i*dim:(i+1)*dim]  # get the conditional mean vector of i-th latent variable
                sum_omega_u_z += np.dot(uni.T, mu_zi)[0][0]

            for j in range(latent_num):
                if q_matrix[n][j] == 1:
                    """ calculate Σ(i)Σ(p≠i)Σ(j)Σ(q≠j) δ(nij)δ(pi)δ(qj) 𝜔(pi)𝜔(qj)U(ni)E[Zp Zq]U(nj) """
                    unj = U[n*dim:(n+1)*dim, j].reshape(dim, 1)

                    if np.any(nm[i] == 1) and np.any(nm[j] == 1):  # both i and j have parents

                        for p in range(latent_num):
                            if nm[i][p] == 1 and p != i:  # p is the parent of i
                                mu_zp = mu_Zs_cond_obs[p*dim:(p+1)*dim]
                                for q in range(latent_num):
                                    if nm[j][q] == 1 and q != j:  # q is the parent of j
                                        mu_zq = mu_Zs_cond_obs[q*dim:(q+1)*dim]
                                        # calculate the E[zp zq]
                                        sigma_zp_zq = sigma_Zs_cond_obs[p*dim:(p+1)*dim, q*dim:(q+1)*dim]
                                        e_zp_zq = sigma_zp_zq + np.dot(mu_zp, mu_zq.T)
                                        # cumulation
                                        sum_omega_u_z_u += W[i][p] * W[j][q] * np.dot(np.dot(uni.T, e_zp_zq), unj)[0][0]
                    
                    elif np.any(nm[i] == 1) and np.any(nm[j] != 1):  # i has parent while j has not any
                        
                        mu_zj = mu_Zs_cond_obs[j*dim:(j+1)*dim]
                        for p in range(latent_num):
                            if nm[i][p] == 1 and p != i:  # p is the parent of i
                                mu_zp = mu_Zs_cond_obs[p*dim:(p+1)*dim]
                                # calculate the E[zp zj]
                                sigma_zp_zj = sigma_Zs_cond_obs[p*dim:(p+1)*dim, j*dim:(j+1)*dim]
                                e_zp_zj = sigma_zp_zj + np.dot(mu_zp, mu_zj.T)
                                # cumulation
                                sum_omega_u_z_u += W[i][p] * np.dot(np.dot(uni.T, e_zp_zj), unj)[0][0]

                    elif np.any(nm[i] != 1) and np.any(nm[j] == 1):  # j has parent while i has not any
                        
                        mu_zi = mu_Zs_cond_obs[i*dim:(i+1)*dim]
                        for q in range(latent_num):
                            if nm[j][q] == 1 and q != j:  # q is the parent of j
                                mu_zq = mu_Zs_cond_obs[q*dim:(q+1)*dim]
                                # calculate the E[zi zq]
                                sigma_zi_zq = sigma_Zs_cond_obs[i*dim:(i+1)*dim, q*dim:(q+1)*dim]
                                e_zi_zq = sigma_zi_zq + np.dot(mu_zi, mu_zq.T)
                                # cumulation
                                sum_omega_u_z_u += W[j][q] * np.dot(np.dot(uni.T, e_zi_zq), unj)[0][0]

                    else:  # both i and j have not any parent
                        mu_zi = mu_Zs_cond_obs[i*dim:(i+1)*dim]
                        mu_zj = mu_Zs_cond_obs[j*dim:(j+1)*dim]
                        # calculate the E[zi zj]
                        sigma_zi_zj = sigma_Zs_cond_obs[i*dim:(i+1)*dim, j*dim:(j+1)*dim]
                        e_zi_zj = sigma_zi_zj + np.dot(mu_zi, mu_zj.T)
                        # cumulate calculations
                        sum_omega_u_z_u += np.dot(np.dot(uni.T, e_zi_zj), unj)[0][0]

    delta_obs_d = math.pow(obs,2) - 2*obs*PHI_n + math.pow(PHI_n,2) + 2*(PHI_n-obs)*sum_omega_u_z + sum_omega_u_z_u

    return delta_obs_d


def cal_e_ztz_cond_obs(k, mu_Zs_cond_obs_lis, sigma_Zs_cond_obs, dim, D):
    """ 
    FUNCTION: Calculate E[Zk.T,Zk], where Z~g(Z|O)

    Inputs:
    -----------------
    :param k --> int
        the k-th latent variable
    
    :param mu_Zs_cond_obs_lis --> dict{d: mu_Zs_cond_obs --> numpy.ndarray(), ...}
        the conditional mean vector of the probability distribution of the joint latent variables conditioned on observations

    :param sigma_Zs_cond_obs --> numpy.ndarray()
        the conditional partial covariance matrix of the probability distribution of the joint latent variables conditioned on observations
    
    :param dim --> int
        the dimensionality of the latent variable
    
    :param D --> int
        the number of samples

    Outputs:
    -----------------
    :return E[Zk.T,Zk]
    """

    # NOTE: E[Zk.T,Zk] = E[math.pow(Zk1,2)] + ... + E[math.pow(ZkT,2)],
    #       where E[math.pow(Zki,2)] = var(Zki) + math.pow(E[Zki], 2).

    #       Besides, Because Z~g(Z|Od), where d is the d-th observation,
    #       so we average the conditional expectation.

    e_ztz = 0  # E[Zk.T,Zk]

    for d in range(D):
        # get the conditional mean vector of Zk based on d-th observation
        mu_zk_d = mu_Zs_cond_obs_lis[d][k*dim:(k+1)*dim]
        # get the partial covariance matrix of Zk
        sigma_zk_d = sigma_Zs_cond_obs[k*dim:(k+1)*dim, k*dim:(k+1)*dim]
        for t in range(dim):
            e_zki2 = sigma_zk_d[t][t] + math.pow(mu_zk_d[t], 2)
            e_ztz += e_zki2
    
    return e_ztz / D


def update_THETA(stu_exe, q_matrix, mu_Zs_cond_obs_lis, sigma_Zs_cond_obs, U, PHI, dim, nm, W):
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
    
    :param nm --> numpy.ndarray
        the neighbor matrix of latent variables
    
    :param W --> numpy.ndarray
        the weight matrix of latent variables

    Outputs:
    -----------------
    :return THETA 
    """

    obs_num, D = stu_exe.shape  # the number of observed nodes (exercises) and samples (students), respectively
    THETA_new = np.zeros(shape=[obs_num], dtype=float)

    count = np.zeros(shape=[obs_num], dtype=int)
    for d in range(D):
        for n in range(obs_num):
            if not np.isnan(stu_exe[n][d]):
                count[n] += 1
                e_delta = cal_delta_obs_d(n, stu_exe[n][d], U, PHI[n][0], q_matrix, mu_Zs_cond_obs_lis[d], sigma_Zs_cond_obs, dim, nm, W)
                THETA_new[n] += e_delta

    return (1/count) * THETA_new


def update_PHI(stu_exe, q_matrix, mu_Zs_cond_obs_lis, U, latent_num, dim, nm, W):
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
    
    :param nm --> numpy.ndarray
        the neighbor matrix of latent variables
    
    :param W --> numpy.ndarray
        the weight matrix of latent variables

    Outputs:
    -----------------
    :return PHI
    """

    obs_num, D = stu_exe.shape  # the number of observed nodes (exercises) and samples (students), respectively
    PHI_new = np.zeros(shape=[obs_num, 1], dtype=float)

    for n in range(obs_num):
        sum_obs = 0  # Σ(d)δ(nd) Oid
        sum_omega_u_z = 0  # Σ(d)δ(nd)Σ(k)Σ(i≠k) δ(nk)δ(ik) 𝜔(ik)U(nk)E[Zi]
        
        count = 0
        for d in range(D):
            if not np.isnan(stu_exe[n][d]):
                count +=1 
                
                """ cumulate Σ(d)δ(nd) Oid """
                sum_obs += stu_exe[n][d]

                for k in range(latent_num):
                    if q_matrix[n][k] == 1:
                        """ cumulate Σ(d)δ(nd)Σ(k)Σ(i≠k) δ(nk)δ(ik) 𝜔(ik)U(nk)E[Zi] """
                        unk = U[n*dim:(n+1)*dim, k].reshape(dim, 1)  # get the feature vector of k-th latent variable with n-th observation

                        if np.any(nm[k] == 1):  # k-latent variable has parent
                            for i in range(latent_num):
                                if nm[k][i] == 1 and i != k:  # i is the parent of k
                                    mu_zi = mu_Zs_cond_obs_lis[d][i*dim:(i+1)*dim]  # get the conditional mean vector of i-th latent variable
                                    sum_omega_u_z += W[k][i] * np.dot(unk.T, mu_zi)[0][0]
                        else:  # if k-latent variable has not any parent
                            # Σ(d)δ(nd)Σ(k)Σ(i≠k) δ(nk)δ(ik) 𝜔(ik)U(nk)E[Zi] = Σ(d)δ(nd)Σ(k)δ(nk) U(nk)E[Zk]
                            mu_zk = mu_Zs_cond_obs_lis[d][k*dim:(k+1)*dim]
                            sum_omega_u_z += np.dot(unk.T, mu_zk)[0][0]
        
        PHI_new[n][0] = (sum_obs - sum_omega_u_z) / count
    
    return PHI_new


def update_U(stu_exe, q_matrix, mu_Zs_cond_obs_lis, sigma_Zs_cond_obs, U, PHI, latent_num, dim, nm, W):
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
    
    :param nm --> numpy.ndarray
        the neighbor matrix of latent variables
    
    :param W --> numpy.ndarray
        the weight matrix of latent variables

    Outputs:
    -----------------
    :return U
    """

    obs_num, D = stu_exe.shape  # the number of observed nodes (exercises) and samples (students), respectively

    U_new = np.zeros(shape=[obs_num*dim, latent_num], dtype=float)

    for n in range(obs_num):
        for k in range(latent_num):
            if q_matrix[n][k] == 1:  # δ(nk) = 1

                # initialization
                sum_omega_zpzq = 0       # Σ(d)δ(nd)δ(nk) Σ(p≠k)Σ(q≠k)δ(pk)δ(qk) 𝜔(pk)𝜔(qk)E[Zp Zq.T]
                sum_obs_phi_omega_z = 0  # Σ(d)δ(nd)δ(nk) (obs - phi) Σ(i≠k)δ(ik) 𝜔(ik)E[Zi]
                sum_omega_z_u = 0        # Σ(d)δ(nd)Σ(p≠k)Σ(j≠k)Σ(q≠j)δ(nkj)δ(pk)δ(qj) 𝜔(pk)𝜔(qj)E[Zp, Zq.T]Unj

                for d in range(D):
                    if not np.isnan(stu_exe[n][d]):  # δ(nd) = 1
                        obs = stu_exe[n][d]
                        phi_n = PHI[n][0]
                        
                        if np.any(nm[k] == 1):  # k has parent

                            for p in range(latent_num):
                                if nm[k][p] == 1 and p != k:  # p is the parent of k, δ(pk) = 1
                                    omega_pk = W[k][p]
                                    mu_zp = mu_Zs_cond_obs_lis[d][p*dim:(p+1)*dim]

                                    # cumulate Σ(d)δ(nd)δ(nk) (obs - phi) Σ(i≠k)δ(ik) 𝜔(ik)E[Zi]
                                    sum_obs_phi_omega_z += (obs - phi_n) * omega_pk * mu_zp

                                    for q in range(latent_num):
                                        if nm[k][q] == 1 and q != k:  # q is the parent of k, δ(qk) = 1
                                            omega_qk = W[k][q]
                                            mu_zq = mu_Zs_cond_obs_lis[d][q*dim:(q+1)*dim]
                                            sigma_zpzq = sigma_Zs_cond_obs[p*dim:(p+1)*dim, q*dim:(q+1)*dim]
                                            e_zpzq = sigma_zpzq + np.dot(mu_zp, mu_zq.T)

                                            # cumulate Σ(d)δ(nd)δ(nk) Σ(p≠k)Σ(q≠k)δ(pk)δ(qk) 𝜔(pk)𝜔(qk)E[Zp Zq.T]
                                            sum_omega_zpzq += omega_pk * omega_qk * e_zpzq

                        else:  # if k has not any parent
                            mu_zk = mu_Zs_cond_obs_lis[d][k*dim:(k+1)*dim]
                            sigma_zkzk = sigma_Zs_cond_obs[k*dim:(k+1)*dim, k*dim:(k+1)*dim]
                            e_zkzk = sigma_zkzk + np.dot(mu_zk, mu_zk.T)

                            # cumulate Σ(d)δ(nd)δ(nk)Σ(p≠k)Σ(q≠k)δ(pk)δ(qk) 𝜔(pk)𝜔(qk)E[Zp Zq.T] = Σ(d)δ(nd)δ(nk) E[Zk Zk.T]
                            sum_omega_zpzq += e_zkzk
                            
                            # cumulate Σ(d)δ(nd)δ(nk) (obs - phi) Σ(i≠k)δ(ik) 𝜔(ik)E[Zi] = Σ(d)δ(nd)δ(nk) (obs - phi) E[Zk]
                            sum_obs_phi_omega_z += (obs - phi_n) * mu_zk
                        
                        for j in range(latent_num):
                            if q_matrix[n][j] == 1 and j != k:  # both δ(nk) = 1 and δ(nj) = 1

                                unj = U[n*dim:(n+1)*dim, j].reshape(dim, 1)

                                if np.any(nm[k] == 1) and np.any(nm[j] == 1):  # both k and j have parents
                                    
                                    for p in range(latent_num):
                                        if nm[k][p] == 1 and p != k:  # p is the parent of k
                                            mu_zp = mu_Zs_cond_obs_lis[d][p*dim:(p+1)*dim]
                                            for q in range(latent_num):
                                                if nm[j][q] == 1 and q != j:  # q is the parent of j
                                                    mu_zq = mu_Zs_cond_obs_lis[d][q*dim:(q+1)*dim]
                                                    # calculate the E[zp zq]
                                                    sigma_zp_zq = sigma_Zs_cond_obs[p*dim:(p+1)*dim, q*dim:(q+1)*dim]
                                                    e_zp_zq = sigma_zp_zq + np.dot(mu_zp, mu_zq.T)
                                                    
                                                    # cumulation Σ(d)δ(nd)Σ(p≠k)Σ(j≠k)Σ(q≠j)δ(nkj)δ(pk)δ(qj) 𝜔(pk)𝜔(qj)E[Zp, Zq.T]Unj
                                                    sum_omega_z_u += W[k][p] * W[j][q] * np.dot(e_zp_zq, unj)[0][0]
                                    
                                elif np.any(nm[k] == 1) and np.any(nm[j] != 1):  # k has parent while j has not any
                                    
                                    mu_zj = mu_Zs_cond_obs_lis[d][j*dim:(j+1)*dim]
                                    for p in range(latent_num):
                                        if nm[k][p] == 1 and p != k:  # p is the parent of k
                                            mu_zp = mu_Zs_cond_obs_lis[d][p*dim:(p+1)*dim]
                                            # calculate the E[zp zj]
                                            sigma_zp_zj = sigma_Zs_cond_obs[p*dim:(p+1)*dim, j*dim:(j+1)*dim]
                                            e_zp_zj = sigma_zp_zj + np.dot(mu_zp, mu_zj.T)

                                            # cumulation Σ(d)δ(nd)Σ(p≠k)Σ(j≠k)Σ(q≠j)δ(nkj)δ(pk)δ(qj) 𝜔(pk)𝜔(qj)E[Zp, Zq.T]Unj
                                            sum_omega_z_u += W[k][p] * np.dot(e_zp_zj, unj)[0][0]

                                elif np.any(nm[k] != 1) and np.any(nm[j] == 1):  # j has parent while k has not any
                                    
                                    mu_zk = mu_Zs_cond_obs_lis[d][k*dim:(k+1)*dim]
                                    for q in range(latent_num):
                                        if nm[j][q] == 1 and q != j:  # q is the parent of j
                                            mu_zq = mu_Zs_cond_obs_lis[d][q*dim:(q+1)*dim]
                                            # calculate the E[zk zq]
                                            sigma_zk_zq = sigma_Zs_cond_obs[k*dim:(k+1)*dim, q*dim:(q+1)*dim]
                                            e_zk_zq = sigma_zk_zq + np.dot(mu_zk, mu_zq.T)

                                            # cumulation Σ(d)δ(nd)Σ(p≠k)Σ(j≠k)Σ(q≠j)δ(nkj)δ(pk)δ(qj) 𝜔(pk)𝜔(qj)E[Zp, Zq.T]Unj
                                            sum_omega_z_u += W[j][q] * np.dot(e_zk_zq, unj)[0][0]

                                else:  # both k and j have not any parent
                                    
                                    mu_zk = mu_Zs_cond_obs_lis[d][k*dim:(k+1)*dim]
                                    mu_zj = mu_Zs_cond_obs_lis[d][j*dim:(j+1)*dim]
                                    # calculate the E[zk zj]
                                    sigma_zk_zj = sigma_Zs_cond_obs[k*dim:(k+1)*dim, j*dim:(j+1)*dim]
                                    e_zk_zj = sigma_zk_zj + np.dot(mu_zk, mu_zj.T)

                                    # cumulation Σ(d)δ(nd)Σ(p≠k)Σ(j≠k)Σ(q≠j)δ(nkj)δ(pk)δ(qj) 𝜔(pk)𝜔(qj)E[Zp, Zq.T]Unj
                                    sum_omega_z_u += np.dot(e_zk_zj, unj)[0][0]

                U_new[n*dim:(n+1)*dim, k] = np.dot(np.linalg.inv(sum_omega_zpzq), sum_obs_phi_omega_z - sum_omega_z_u).reshape(dim)
    
    return U_new


def update_W(stu_exe, q_matrix, mu_Zs_cond_obs_lis, sigma_Zs_cond_obs, U, PHI, THETA, latent_num, dim, nm, W):
    """
    FUNCTION: Updating weight matrix of latent variables

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

    :param THETA --> numpy.ndarray

    :param latent_num --> int
        the number of latent variables
    
    :param dim --> int
        the dimensionality of the latent variable
    
    :param nm --> numpy.ndarray
        the neighbor matrix of latent variables
    
    :param W --> numpy.ndarray
        the weight matrix of latent variables

    Outputs:
    -----------------
    :return U
    """

    obs_num, D = stu_exe.shape
    W_new = np.identity(latent_num)

    for k in range(latent_num):
        for p in range(latent_num):
            if nm[k][p] == 1:  # p is the parent of k
                W_new[k][k] = 0  # k has parents, reset the corresponding diagonal element
                # update weight from p to k
                lambda_1, lambda_2, lambda_3, lambda_4 = cal_lambda(k, p, obs_num, D, latent_num, stu_exe, q_matrix, mu_Zs_cond_obs_lis, sigma_Zs_cond_obs, U, PHI, THETA, dim, nm, W)
                gamma_1, gamma_2, gamma_3, upsilon_1, upsilon_2, upsilon_3 = cal_Gamma_Upsilon(k, p, D, latent_num, mu_Zs_cond_obs_lis, sigma_Zs_cond_obs, dim, nm, W)
                W_new[k][p] = (lambda_1 - lambda_2 - lambda_3 - gamma_1 - gamma_2 + upsilon_1 + upsilon_2) / (lambda_4 + gamma_3 - upsilon_3)
    
    return W_new


def cal_lambda(k, p, obs_num, D, latent_num, stu_exe, q_matrix, mu_Zs_cond_obs_lis, sigma_Zs_cond_obs, U, PHI, THETA, dim, nm, W):
    """
    Calculate Λ1, Λ2, Λ3, Λ4
    Λ1 = Σ(d)Σ(n)δ(dn)δ(nk) 1/(θn) * (obs - phi)U(nk)E[Zp]
    Λ2 = Σ(d)Σ(n)δ(dn)δ(nk) 1/(θn) * Σ(j≠k)δ(nj)Σ(q≠j)δ(qj) 𝜔(qj)U(nk)E[Zp,Zq]U(nj)
    Λ3 = Σ(d)Σ(n)δ(dn)δ(nk) 1/(θn) * Σ(q≠k,p) δ(qk) 𝜔(qk)U(nk)E[Zp,Zq]U(nk)
    Λ4 = Σ(d)Σ(n)δ(dn)δ(nk) 1/(θn) * U(nk)E[Zp,Zp]U(nk)
    """

    lambda_1, lambda_2, lambda_3, lambda_4 = 0, 0, 0, 0  # initialization
    
    for d in range(D):
        for n in range(obs_num):
            if not np.isnan(stu_exe[n][d]) and q_matrix[n][k] == 1:  # δ(dn)δ(nk) = 1
                theta_n, phi_n, obs = THETA[n], PHI[n][0], stu_exe[n][d]
                unk = U[n*dim:(n+1)*dim, k].reshape(dim, 1)
                mu_zp = mu_Zs_cond_obs_lis[d][p*dim:(p+1)*dim]

                # cumulate Λ4
                # calculate the E[zp zp]
                sigma_zp_zp = sigma_Zs_cond_obs[p*dim:(p+1)*dim, p*dim:(p+1)*dim]
                e_zp_zp = sigma_zp_zp + np.dot(mu_zp, mu_zp.T)
                
                lambda_4 += (1/theta_n) * (np.dot(np.dot(unk.T, e_zp_zp), unk)[0][0])
                
                # cumulate Λ1
                lambda_1 += (1/theta_n) * ((obs - phi_n) * np.dot(unk, mu_zp)[0][0])

                for j in range(latent_num):

                    # cumulate Λ2
                    if q_matrix[n][j] == 1 and j != k:  # δ(nj) = 1 and j≠k
                        unj = U[n*dim:(n+1)*dim, j].reshape(dim, 1)
                        
                        if np.any(nm[j] == 1):  # j has parent
                            for q in range(latent_num):
                                if nm[j][q] == 1 and q != j:  # q is the father of j (δ(qj) = 1 and q≠j)
                                    mu_zq = mu_Zs_cond_obs_lis[d][q*dim:(q+1)*dim]
                                    # calculate the E[zp zq]
                                    sigma_zp_zq = sigma_Zs_cond_obs[p*dim:(p+1)*dim, q*dim:(q+1)*dim]
                                    e_zp_zq = sigma_zp_zq + np.dot(mu_zp, mu_zq.T)
                                    
                                    lambda_2 += (1/theta_n) * (W[j][q] * np.dot(np.dot(unk.T, e_zp_zq), unj)[0][0])
                        else:
                            # in Λ2, Σ(j≠k)δ(nj)Σ(q≠j)δ(qj) 𝜔(qj)U(nk)E[Zp,Zq]U(nj) = Σ(j≠k)δ(nj) U(nk)E[Zp,Zj]U(nj)
                            mu_zj = mu_Zs_cond_obs_lis[d][j*dim:(j+1)*dim]
                            # calculate the E[zp zj]
                            sigma_zp_zj = sigma_Zs_cond_obs[p*dim:(p+1)*dim, j*dim:(j+1)*dim]
                            e_zp_zj = sigma_zp_zj + np.dot(mu_zp, mu_zj.T)
                            
                            lambda_2 = (1/theta_n) * (np.dot(np.dot(unk.T, e_zp_zj), unj)[0][0])
                    
                    # cumulate Λ3
                    if nm[k][j] == 1 and j != k and j != p:  # besides p, j is another parent of k
                        mu_zj = mu_Zs_cond_obs_lis[d][j*dim:(j+1)*dim]
                        # calculate the E[zp zj]
                        sigma_zp_zj = sigma_Zs_cond_obs[p*dim:(p+1)*dim, j*dim:(j+1)*dim]
                        e_zp_zj = sigma_zp_zj + np.dot(mu_zp, mu_zj.T)

                        lambda_3 += (1/theta_n) * (W[k][j] * np.dot(np.dot(unk.T, e_zp_zj), unk)[0][0])

    return lambda_1, lambda_2, lambda_3, lambda_4


def cal_Gamma_Upsilon(k, p, D, latent_num, mu_Zs_cond_obs_lis, sigma_Zs_cond_obs, dim, nm, W):
    """
    Calculate Γ1, Γ2, Γ3 and ϓ1, ϓ2, ϓ3
    
    Γ1 = T * 1/𝜔(k) * Σ(j≠p,k)δ(jk)δ(pj) 𝜔(jk)𝜔(pj)  # j is both the parent of k and the neighbor of p
    Γ2 = 0.5 * T * Σ(i≠p,k)δ(pi)δ(ki) 1/𝜔(i) 𝜔(pi)𝜔(ki)  # i is both the child of p and k
    Γ3 = T * 1/𝜔(k)

    ϓ1 = 1/[𝜔(k)]^(-2) * E[Zk.T,Zk] * Σ(j≠p,k)δ(jk)δ(pj)𝜔(jk)𝜔(pj)  # j is both the parent of k and the neighbor of p
    ϓ2 = 0.5 * Σ(i≠p,k)δ(pi)δ(ki) 1/[𝜔(i)]^(-2) E[Zi.T,Zi] 𝜔(pi)𝜔(ki)  # i is both the child of p and k
    ϓ3 = 1/[𝜔(k)]^(-2) * E[Zk.T,Zk]
    """

    gamma_1, gamma_2, gamma_3 = 0, 0, 0
    upsilon_1, upsilon_2, upsilon_3 = 0, 0, 0

    omega_k = cal_omega_k(k, latent_num, nm, W)  # 𝜔(k)
    e_ztz_k = cal_e_ztz_cond_obs(k, mu_Zs_cond_obs_lis, sigma_Zs_cond_obs, dim, D)  # E[Zk.T,Zk]

    gamma_3 = dim * (1/omega_k)
    upsilon_3 = math.pow(omega_k, -2) * e_ztz_k

    for j in range(latent_num):
        if j != p and j != k and nm[k][j] == 1 and (nm[p][j]*nm[j][p] != 0):  # j is both the parent of k and the neighbor of p
            # cumulate Γ1
            omega_k = cal_omega_k(k, latent_num, nm, W)
            gamma_1 += dim * (1/omega_k) * W[k][j] * (W[p][j] + W[j][p])

            # cumulate ϓ1
            upsilon_1 += math.pow(omega_k, -2) * e_ztz_k * W[k][j] * (W[p][j] + W[j][p])

        if j != p and j != k and nm[j][p] == 1 and nm[j][k] == 1:  # j is both the child of p and k
            # cumulate Γ2
            omega_j = cal_omega_k(j, latent_num, nm, W)  # 𝜔(j)
            gamma_2 += 0.5 * dim * (1/omega_j) * W[j][p] * W[j][k]

            # cumulate ϓ2
            e_ztz_j = cal_e_ztz_cond_obs(j, mu_Zs_cond_obs_lis, sigma_Zs_cond_obs, dim, D)  # E[Zj.T,Zj]
            upsilon_2 += 0.5 * math.pow(omega_j, -2) * e_ztz_j * W[j][p] * W[j][k]

    return gamma_1, gamma_2, gamma_3, upsilon_1, upsilon_2, upsilon_3

