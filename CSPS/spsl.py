"""
The  SKILLS' PREREQUISITE STRUCTURE LEARNING (SPSL) algorithm

References
[1] https://www.ncbi.nlm.nih.gov/pmc/articles/PMC1480117/pdf/amia2003_0021.pdf

[2] Yang, J., Li, L., & Wang, A. (2011). 
    A partial correlation-based Bayesian network structure learning algorithm under linear SEM. 
    Knowledge-Based Systems, 24(7), 963-976.

[3] Opgen-Rhein, R., & Strimmer, K. (2007).
    From correlation to causation networks:
    a simple approximate learning algorithm and its application to high-dimensional plant gene expression data.
    BMC systems biology, 1(1), 1-10.
    https://bmcsystbiol.biomedcentral.com/track/pdf/10.1186/1752-0509-1-37.pdf
"""

import math
import numpy as np
from scipy import stats

SL = stats.norm.cdf(0.5)  # significance level
# SL = 0.95  # equals to p-value=0.05




class spsl():

    def __init__(self, gn) -> None:
        """
        FUNCTION: Initialization

        Inputs:
        -----------------
        :param gn --> CLASS gaussian_network()
            the CLASS of Gaussian Network
        """
        
        self.latent_var_set, self.mu_pos_lis, self.cov_pos, self.stu_exe, self.q, self.U, self.PHI, self.THETA, self.dim \
            = gn.latent_var_set, gn.mu_Zs_pos_lis, gn.sigma_Zs_pos, gn.stu_exe, gn.q_matrix, gn.U, gn.PHI, gn.THETA, gn.dim
        
        self.samples = gn.stu_exe.shape[1]
    

    def searching(self):
        """
        FUNCTION: Searching every latent variable's (partial)direcred neighbors, including
        1. The SKELETON SEARCHING STEP
        2. The DIRECTION SEARCHING STEP

        Outputs:
        -----------------
        :return learned graph --> dict(skill id : [parent id])
        """
        # 1. THE SKELETON SEARCHING STEP: Search the neighbours (parents and children) for each latent variable
        for latent_var in self.latent_var_set.values():
            pc_k = pc_searching(latent_var.id, self.latent_var_set, self.cov_pos, self.samples)
            for nb in pc_k:
                latent_var.neighbors.append(nb)
        # 2. THE DIRECTION SEARCHING STEP: Causal detection
        for latent_var in self.latent_var_set.values():
            if len(latent_var.neighbors):
                causal_detection(latent_var, self.latent_var_set, self.cov_pos)
                print("ID:", latent_var.id, "Parents:", latent_var.parents, "Children:", latent_var.children)
            else:
                print("ID:", latent_var.id, "has not any parents and children.")
        
        # integrity checking
        for latent_var in self.latent_var_set.values():
            latent_id = latent_var.id
            parents = latent_var.parents
            children = latent_var.children
            for pa in parents:
                if not latent_id in self.latent_var_set[pa].children:
                    self.latent_var_set[pa].children.append(latent_id)
            for ch in children:
                if not latent_id in self.latent_var_set[ch].parents:
                    self.latent_var_set[ch].parents.append(latent_id)
      
        # return the learned graph
        learned_graph = dict()
        for latent_var in self.latent_var_set.values():
            skill_id = latent_var.id
            triggers = latent_var.parents
            if not skill_id in learned_graph.keys():
                learned_graph[skill_id] = triggers 

        return learned_graph


def pc_searching(k, latent_var_set, cov, samples):
    """
    FUNCTION: Searching the potential parents and children of target latent variable k

    Inputs:
    -----------------
    :param k --> int
        the ID of the of target latent variable
    
    :param latent_var_set --> dict()
        the set of CLASS of latent variables
    
    :param cov --> numpy.ndarray
        the covariance matrix
    
    :param samples --> int
        the number of samples
    
    Outputs:
    -----------------
    :return current_pc --> list()
    """

    current_pc = list()  # the ID of potential parents and children of latent variable k
    ass_set = dict()  # the latent variable j's ID and the corresponding association value with k

    # calculate the (marginal) value of association of all latent variables with k
    for latent_var in latent_var_set.values():  # walk through all latent variables
        j = latent_var.id
        if j != k:  # j not equals to k
            _ass = cal_association(k, j, cov, samples, 0)
            ass_set[j] = _ass

    # sorted the parents and children of k based on association value
    pc_sorted = sorted(ass_set.items(), key = lambda x:(x[1], x[0]))

    while len(pc_sorted):
        current_pc.append(pc_sorted.pop()[0])

        current_vars = current_pc.copy()
        current_vars.append(k)  # add the target latent variable
        
        # verification of conditional independence between i (in current_pc) and k
        for i in current_pc:
            # calculate the partial covaraince matrix of i and k 
            # conditioned on all the remaining variables in current_pc
            par_cov, cond_num = cal_partial_covariance(k, i, current_vars, cov)
            # calculate the (conditional) value of association of i and k
            _ass = cal_association(0, 1, par_cov, samples, cond_num)
            if _ass <= SL:  # if the association is less than the threshold
                current_pc.remove(i)

    return current_pc


def cal_association(i, j, cov, samples, cond_num):
    """ 
    FUNCTION: Calculate the value of association of i and j

    Inputs:
    -----------------
    :param i --> int
        the ID of the latent variable
    
    :param j --> int
        the ID of the latent variable
    
    :param cov --> numpy.ndarray
        the covariance matrix
    
    :param samples --> int
        the number of the samples
    
    :param cond_num --> int
        the number of the variable that i and j to be conditioned

    :Outputs:
    -----------------
    :return ass --> float
        the association of i and j
    """
    # calculate the (conditional) correlation coefficient between i and j
    rho = abs(cov[i][j] / (math.sqrt(cov[i][i]*cov[j][j])))
    # calculate the value of association
    zeta = 0.5 * math.sqrt(samples - cond_num - 3) * (math.log((1+rho)/(1-rho)))
    # p_value = stats.norm.cdf(-zeta)*2
    ass = 1 - stats.norm.cdf(-zeta)*2
    return ass


def cal_partial_covariance(k, i, current_vars, cov):
    """
    FUNCTION: Calculate the PARTIAL COVARIANCE MATRIX of the variable k and i conditioned on other variables.

    Inputs:
    -----------------
    :param k --> int
        the ID of the of target latent variable
    
    :param i --> int
        the ID of the latent variable which to be blocked with k

    :param current_vars --> list()
        all latent variables in Current PC, including k

    :param cov --> numpy.matrix
        the covariance matrix

    Outputs:    
    -----------------
    :return  par_cov --> numpy.matrix
        the new covariance matrix of BLOCK 1 conditioned on BLOCK 4
    
    :return cond_num --> int
        the number of the variable that k and i to be conditioned
    """

    # NOTE: We block the original covariance matrix as follows:
    #            +---------------------+
    # BLOCK 1: = | cov(i, i) cov(i, k) |
    #            | cov(k, i) cov(k, k) |
    #            +---------------------+
    #            +---------------------+
    # BLOCK 2: = |...cov(i, m) m≠i,k...|
    #            |...cov(k, m) m≠i,k...|
    #            +---------------------+
    #            +---------------------------------+
    #            |     ...             ...         |
    # BLOCK 3: = | cov(m, i) m≠i,k cov(m, k) m≠i,k | 
    #            |     ...             ...         |
    #            +---------------------------------+
    #            +-------------------------------+
    #            |              ...              |
    # BLOCK 4: = |  ...  cov(p≠i,k; j≠i,k)  ...  |
    #            |              ...              |
    #            +-------------------------------+
    #            +--------+--------+
    #            | BLOCK1 | BLOCK2 |
    # cov_new: = +--------+--------+
    #            | BLOCK3 | BLOCK4 |
    #            +--------+--------+

    _ = [x for x in current_vars if x not in [i, k]]
    cond_num = len(_)

    # BLOCK 1
    block_1 = cov[np.ix_([i,k], [i,k])]
    # BLOCK 2
    block_2 = cov[np.ix_([i,k], _)]
    # BLOCK 3
    block_3 = cov[np.ix_(_, [i,k])]
    # BLOCK 4 
    block_4 = cov[np.ix_(_, _)]

    cov_new = np.vstack((np.hstack((block_1, block_2)), np.hstack((block_3, block_4))))
    precision_m = np.linalg.inv(cov_new)  # inverse to get the precision matrix
    par_cov = np.linalg.inv(precision_m[0:2, 0:2])  # the patrial covariance matrix of BLOCK 1 conditioned on BLOCK 4

    return par_cov, cond_num


def causal_detection(var, var_set, cov):
    """
    FUNCTION: Edge direction learning between the var and its neighbors.

    Inputs:    
    -----------------
    :param var --> CLASS gaus_node
        the target latent variables
    
    :param var_set --> dict()
        the set of CLASS of latent variables
    
    :param cov --> numpy.ndarray
        the covariance matrix

    Outputs:
    -----------------
    """

    pc_var = var.neighbors  # the neighbors (parents and children) of the var
    pcpc_var = find_pcpc(var, var_set, pc_var)  # the potential Markov blanket of the var

    # calculate the partial variance of var conditioned on all variables in its potential Markov blanket
    par_variance = cal_partial_variance(var.id, pcpc_var, cov)
    # calculate the Standardized Partial Variance (SPV) of the var
    spv_var = par_variance / cov[var.id][var.id]

    # edge direction decision
    # by comparing the SPV values of the var and that of its neighbors
    for nb in pc_var:
        
        if var.id in var_set[nb].parents:  # already has been detected
            var.children.append(nb)
            continue
            
        if var.id in var_set[nb].children:  # already has been detected
            var.parents.append(nb)
            continue

        pcpc_nb = find_pcpc(var_set[nb], var_set, var_set[nb].neighbors)  # find the potential Markov blanket of nb
        par_variance = cal_partial_variance(nb, pcpc_nb, cov)  # calculate the partial variance of nb
        spv_nb = par_variance / cov[nb][nb]

        if spv_nb > spv_var:
            var.parents.append(nb)
        elif spv_var > spv_nb:
            var.children.append(nb)

    return


def find_pcpc(var, var_set, pc):
    """
    Find the potential Markov blanket of the var,
    including the parents and children of the parents and children of the var
    """

    # find all neighbors (parents and children) of every neighbor of the var
    pcpc = [] + pc
    for nb in pc:
        pcpc = pcpc + var_set[nb].neighbors
    
    pcpc = list(set(pcpc))  # remove duplicated elenemts
    if var.id in pcpc:
        pcpc.remove(var.id)  # remove the var itself
    
    return pcpc


def cal_partial_variance(k, pcpc, cov):
    """
    FUNCTION: Calculate the partial variance of the variable k,
    conditioned on all variables in its potential Markov blanket

    Inputs:
    -----------------
    :param k --> int
        the ID of the target variable
    
    :param pcpc --> list()
        the potential Markov blanket of the varibale k,
        including parents and children of the parents and children of k
    
    :param cov --> numpy.ndarray
        the covariance matrix
    """

    # NOTE: We block the original covariance matrix as follows:
    #            +-----------+
    # BLOCK 1: = | cov(k, k) |
    #            +-----------+
    #            +---------------------+
    # BLOCK 2: = | ...cov(k, m) m≠k... |
    #            +---------------------+
    #            +---------------+
    #            |     ...       |
    # BLOCK 3: = | cov(m, k) m≠k | 
    #            |     ...       |
    #            +---------------+
    #            +-----------------------+
    #            |          ...          |
    # BLOCK 4: = | ... cov(p≠k; j≠k) ... |
    #            |          ...          |
    #            +-----------------------+
    #            +--------+--------+
    #            | BLOCK1 | BLOCK2 |
    # cov_new: = +--------+--------+
    #            | BLOCK3 | BLOCK4 |
    #            +--------+--------+
    #                                  +-----+-----+
    #                                  | A11 | A12 |
    # the inverse of the cov_new is: = +-----+-----+
    #                                  | A21 | A22 |
    #                                  +-----+-----+
    # So the partial covariance of BLOCK 1 condition BLOCK 4 is given:
    # Sigma (BLOCK 1 | BLOCK 4) = the inverse of the A11
    # see the Equation (2.73) and (2.78) in
    # Bishop, C. M., & Nasrabadi, N. M. (2006). Pattern recognition and machine learning (Vol. 4, No. 4, p. 738). New York: springer.

    block_1 = cov[np.ix_([k], [k])]
    block_2 = cov[np.ix_([k], pcpc)]
    block_3 = cov[np.ix_(pcpc, [k])]
    block_4 = cov[np.ix_(pcpc, pcpc)]

    cov_new = np.vstack((np.hstack((block_1, block_2)), np.hstack((block_3, block_4))))
    precision_m = np.linalg.inv(cov_new)  # the precision matrix
    par_variance = np.linalg.inv(precision_m[0:1, 0:1])[0][0]

    return par_variance