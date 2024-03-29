import numpy as np
from itertools import combinations


class BasisFunctions:
    def __init__(self, nb_stocks):
        self.nb_stocks = nb_stocks
        lst = list(range(self.nb_stocks))
        self.combs =  [list(x) for x in combinations(lst, 2)]
        self.nb_base_fcts = 3  #1 + 2 * self.nb_stocks + len(self.combs)
        # print("self.nb_base_fcts", self.nb_base_fcts)
        # print("nb_stocks",  self.nb_stocks)

    def base_fct(self, i, x):   # i -> coefficient and x -> X's enteries X[path, :]
        # print("i and x in base : ", i, x)
        bf=np.nan
        if (i == 0):
            bf = 1.0#np.ones_like(x[0]) # (constant)
            # print(bf)
        elif (i <= self.nb_stocks):
            bf = x#[i-1] # (x1, x2, ..., xn)
            # print(bf)
        elif (self.nb_stocks < i <= 2 * self.nb_stocks):
            k = i - self.nb_stocks - 1
            bf = x**2#[k] ** 2 # (x1^2, x2^2, ..., xn^2)
            # print(bf)
        elif (i > 2 * self.nb_stocks):
            k = i - 2*self.nb_stocks -1
            bf = x[self.combs[k][0]] * x[self.combs[k][1]] # (x1x2, ..., xn-1xn)
            # print(bf)
        return bf

class BasisFunctionsLaguerre:
    def __init__(self, nb_stocks, K=1):
        self.nb_stocks = nb_stocks
        self.nb_base_fcts = 1 + 3 * self.nb_stocks  
        '''Figure out'''
        self.K = K

    def base_fct(self, i, x):
        bf=np.nan
        x = x / self.K
        if (i == 0):
            bf = np.ones_like(x) # (constant)
        elif (i <= self.nb_stocks):
            bf = np.exp(-x/2)
        elif (self.nb_stocks < i <= 2 * self.nb_stocks):
            k = i - self.nb_stocks - 1
            bf = np.exp(-x/2)*(1-x)
        elif (i > 2 * self.nb_stocks):
            k = i - 2*self.nb_stocks -1
            bf = np.exp(-x/2)*(1-2*x+(x**2)/2)
        return bf


class BasisFunctionsLaguerreTime:
    """assumes that the last stock is the current time"""
    def __init__(self, nb_stocks, T, K=1):
        self.nb_stocks = nb_stocks
        self.nb_base_fcts = 1 + 3 * self.nb_stocks
        self.T = T
        self.K = K

    def base_fct(self, i, x):
        bf = np.nan
        x = x / self.K
        if (i == 0):
            bf = np.ones_like(x[0]) # (constant)
        elif (i < self.nb_stocks):
            bf = np.exp(-x[i-1]/2)
        elif i == self.nb_stocks:  # time polynomial
            bf = np.sin(-np.pi*x[i-1]/2*self.K + np.pi/2)
        elif (self.nb_stocks < i < 2 * self.nb_stocks):
            k = i - self.nb_stocks - 1
            bf = np.exp(-x[k]/2)*(1-x[k])
        elif i == 2 *self.nb_stocks:  # time polynomial
            k = i - self.nb_stocks - 1
            bf = np.log(1 + self.T * (1-x[k]*self.K))
        elif (2 * self.nb_stocks < i < 3*self.nb_stocks):
            k = i - 2*self.nb_stocks -1
            bf = np.exp(-x[k]/2)*(1-2*x[k]+(x[k]**2)/2)
        elif i == 3*self.nb_stocks:
            k = i - 2*self.nb_stocks -1
            bf = (x[k]*self.K)**2
        return bf