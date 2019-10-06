import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
from Kshape_distance import SBD,NCCc_fft

class SOM:
    def __init__(self,m,n,input_dim,num_features,
                 num_iterations,eta=0.5,sigma=None):
        '''
        :param m:
        :param n:
        :param input_dim: the length of series
        :param eta: learning rate
        :param sigma: radius of neighbourhood
        num_features * input_dim for every stock
        '''
        self._m = m
        self._n = n
        self._num_iterations = int(num_iterations)
        self._neighbourhood = []
        self._topography = []

        if sigma is None:
            self.sigma = max(m,n)/2.0   #const radius
        else:
            self.sigma = float(sigma)

        self._graph = tf.Graph()

        with self._graph.as_default():
            #weight matrix
            self._W= tf.Variable(tf.random_normal([m*n,input_dim,num_features],seed=0))
            self._topography = tf.constant(
                np.array(list(self.grid_location(m,n)))
            )

            #placeholder for training data
            self._X = tf.placeholder(tf.float32,[num_features,input_dim])
            #placeholder to keep track of number of iters
            self._iter = tf.placeholder(tf.float32)



    def grid_location(self,m,n):
        for i in range(m):
            for j in range(n):
                yield np.array([i,j])