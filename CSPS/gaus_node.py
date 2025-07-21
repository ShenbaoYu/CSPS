# -*- coding: utf-8 -*-
"""
The Gaussian nodes of latent variable
"""

import numpy as np


class gaus_node:
    """
    The CLASS of knowledge concept latent variable
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
        """

        self.id = id  # ID
        # self.mu = np.random.uniform(size=dim).reshape(dim, 1)  # mean vector
        self.mu = np.zeros(shape=[dim, 1], dtype=float)  # mean vector (Zero mean)
        self.sigma = np.identity(dim)  # covariance matrix (Identity Matrix)
        
        self.neighbors = list()  # the neighbors
        self.parents = list()  # the parents 
        self.children = list()  # the children
