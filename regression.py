""" Calculate continuation values by regression.

It contains the different forms of regressions:
- Least squares regression using basis functions (for LSM)
- Least squares regression using randomized neural networks (for RLSM and RRLSM)

- Ridge regression using basis function or randomized neural networks.
- Least square using weighted Laguerre Polynomials.
"""

import numpy as np
import basis_functions
# import torch
# import sklearn.linear_model
COEFFICIENTS=[]

nb_paths = 16

class Regression:
  def __init__(self, payoff_fct=None):
    pass

class LeastSquares(Regression):
  def __init__(self, nb_stocks):
    self.nb_stocks = nb_stocks
    self.bf = basis_functions.BasisFunctionsLaguerre(self.nb_stocks)

  def calculate_regression(self, X, Y, in_the_money, in_the_money_all, date_no, ML, noise):
    """ Calculate continuation values by least squares regression."""
    # nb_paths = X.shape
    if ML:
      reg_vect_mat = np.empty((1, self.bf.nb_base_fcts))  # (8 X 3) matrix
      for path in range(1):
          print(path)
          for coeff in range(self.bf.nb_base_fcts): # [0,1,2,3,4]
            reg_vect_mat[path, coeff] = self.bf.base_fct(coeff, X)   # base function calling
      # print("COEFF : ", COEFFICIENTS)
      coefficients = COEFFICIENTS[len(COEFFICIENTS)-date_no]
      continuation_values = np.dot(reg_vect_mat[in_the_money_all[0]],
                                  coefficients)
      
      # print("coefficients :   ",coefficients)  # only first index "0" coefficients are important
      # print("in the money : ", in_the_money)
      # print("in the money all : ", in_the_money_all)
      # print("overall reg : ", reg_vect_mat)
      # print("reg in dot : ", reg_vect_mat[in_the_money_all])

      return continuation_values  

    elif noise:
      reg_vect_mat = np.empty((nb_paths, self.bf.nb_base_fcts))  # (8 X 3) matrix
      for path in range(nb_paths):
          # print(path)
          for coeff in range(self.bf.nb_base_fcts): # [0,1,2,3,4]
            reg_vect_mat[path, coeff] = self.bf.base_fct(coeff, X[path])   # base function calling
      # print("COEFF : ", COEFFICIENTS)
      coefficients = COEFFICIENTS[len(COEFFICIENTS)-date_no]
      continuation_values = np.dot(reg_vect_mat[in_the_money_all[0]],
                                  coefficients)
      print("Coefficients : ",COEFFICIENTS)
      return continuation_values                         

    else:
      reg_vect_mat = np.empty((nb_paths, self.bf.nb_base_fcts))  # (8 X 3) matrix
      for path in range(nb_paths):
        for coeff in range(self.bf.nb_base_fcts): # [0,1,2,3,4]
          reg_vect_mat[path, coeff] = self.bf.base_fct(coeff, X[path])   # base function calling
      
      #reg_vect_mat[in_the_money[0]] = 0
      #Y[in_the_money[0]] = 0
      coefficients = np.linalg.lstsq(
        reg_vect_mat[in_the_money[0]], Y[in_the_money[0]], rcond=None)
      continuation_values = np.dot(reg_vect_mat[in_the_money_all[0]],
                                  coefficients[0])
      # print("coefficients :   ",coefficients)  # only first index "0" coefficients are important
      COEFFICIENTS.append(coefficients[0])
      print("Coefficients : ",COEFFICIENTS)

      # print("Continuation in the regression file is : ", continuation_values)
      # print("continuation : ", continuation_values)
      return continuation_values












































# class LeastSquaresLaguerre(LeastSquares):
#   """ Calculate continuation values by least squares regression using
#   weighted Laguerre polynomials.
#   """
#   def __init__(self, nb_stocks):
#     self.nb_stocks = nb_stocks
#     self.bf = basis_functions.BasisFunctionsLaguerre(self.nb_stocks)


# class LeastSquaresRidge(Regression):
#   """ Calculate continuation values by Ridge regression."""

#   def __init__(self, nb_stocks, ridge_coeff=1.,):
#     self.nb_stocks = nb_stocks
#     self.bf = basis_functions.BasisFunctions(self.nb_stocks)
#     self.alpha = ridge_coeff


#   def calculate_regression(self, X, Y, in_the_money, in_the_money_all):
#     nb_paths, nb_stocks = X.shape
#     reg_vect_mat = np.empty((nb_paths, self.bf.nb_base_fcts))
#     for path in range(nb_paths):
#       for coeff in range(self.bf.nb_base_fcts):
#         reg_vect_mat[path, coeff] = self.bf.base_fct(coeff, X[path, :])

#     model = sklearn.linear_model.Ridge(alpha=self.alpha)
#     model.fit(X=reg_vect_mat[in_the_money[0]], y=Y[in_the_money[0]])
#     continuation_values = model.predict(reg_vect_mat[in_the_money_all[0]])
#     return continuation_values


# class ReservoirLeastSquares(Regression):
#   """  Calculate continuation values by least squares regression using
#   randomized neural networks.
#   """
#   def __init__(self, state_size, hidden_size=10, factors=(1.,),
#                activation=torch.nn.LeakyReLU(0.5)):
#     self.nb_base_fcts = hidden_size + 1
#     self.state_size = state_size
#     self.reservoir = randomized_neural_networks.Reservoir(
#       hidden_size, self.state_size, factors=factors, activation=activation)

#   def calculate_regression(self, X_unsorted, Y, in_the_money, in_the_money_all):
#     X = torch.from_numpy(X_unsorted)
#     X = X.type(torch.float32)
#     reg_input = np.concatenate(
#       [self.reservoir(X).detach().numpy(), np.ones((len(X), 1))], axis=1)
#     coefficients = np.linalg.lstsq(
#       reg_input[in_the_money[0]], Y[in_the_money[0]], rcond=None)
#     continuation_values = np.dot(reg_input[in_the_money_all[0]], coefficients[0])
#     return continuation_values


# class ReservoirLeastSquaresRidge(Regression):
#   """ Calculate continuation values by Ridge regression using randomized NN.
#   """
#   def __init__(self, state_size, hidden_size=10, factors=(1.,), ridge_coeff=1.,
#                activation=torch.nn.LeakyReLU(0.5)):
#     self.nb_base_fcts = hidden_size + 1
#     self.state_size = state_size
#     self.alpha = ridge_coeff
#     self.reservoir = randomized_neural_networks.Reservoir(
#       hidden_size, self.state_size, factors=factors, activation=activation)

#   def calculate_regression(self, X_unsorted, Y, in_the_money, in_the_money_all):
#     X = torch.from_numpy(X_unsorted)
#     X = X.type(torch.float32)
#     reg_input = np.concatenate(
#       [self.reservoir(X).detach().numpy(), np.ones((len(X), 1))], axis=1)

#     model = sklearn.linear_model.Ridge(alpha=self.alpha)
#     model.fit(X=reg_input[in_the_money[0]], y=Y[in_the_money[0]])
#     continuation_values = model.predict(reg_input[in_the_money_all[0]])
#     return continuation_values
