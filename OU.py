import random
import numpy as np 

class OU(object):

    def function(self, x, mu, theta, sigma):
        return theta * (mu - x) + sigma * np.random.randn(1)

    def random(self, x):
        return np.random.normal(0, 1) * x
