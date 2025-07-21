# -*- coding: utf-8 -*-
"""
The Gaussian network
"""

import numpy as np

class gauss_net:

    def __init__(self, id, q_matrix, dim) -> None:

        obs_num, latent_num = q_matrix.shape
        
        self.id = id
        self.obj = -np.inf
        self.nm = None
        self.W = np.zeros(shape=(latent_num, latent_num), dtype=float)
        self.U = np.random.rand(obs_num*dim, latent_num) * q_matrix
        self.PHI = np.random.uniform(size=obs_num).reshape(obs_num, 1)
        self.THETA = np.random.uniform(size=obs_num)
        self.mu_Zs_pos_lis = dict()  # the posterior mean conditioned on observations