""" Computes the American option price by Least Square Monte Carlo (LSM).

It is the implementation of the Least Square Monte Carlo introduced in
(Valuing American Options by Simulation: A Simple Least-Squares Approach,
Longstaff and Schwartz, 2001).
"""

from audioop import maxpp
import numpy as np
import regression
import backward_induction_pricer
from stock_model import CustomStockModel
from payoff import MaxPut

class LeastSquaresPricer(backward_induction_pricer.AmericanOptionPricer):
  """ Computes the American option price by Least Square Monte Carlo (LSM).

  It uses a least square regression to compute the continuation value.
  The basis functions used are polynomial of order 2.
  """

  def __init__(self, model, payoff, nb_epochs=None, nb_batches=None,
               train_ITM_only=True):

    #regression class:  defines the regression used for the contination value.
    self.regression = regression.LeastSquares(model.nb_stocks)
    super().__init__(model, payoff, train_ITM_only=train_ITM_only)

  def calculate_continuation_value(self, values, immediate_exercise_value,
                   stock_paths_at_timestep, date_no, ML, noise):
    """See base class."""
    # self.split = int(len(stock_paths)/2) is the backward_induction pricers
    in_the_money = np.where(immediate_exercise_value > 0)
    in_the_money_all = np.where(immediate_exercise_value > 0)
    if not ML:
      return_values = np.zeros(stock_paths_at_timestep.shape[0])
    else:
      return_values = np.zeros(1)
    # print("stocks at time step t: ", stock_paths_at_timestep)
    return_values[in_the_money_all[0]] = self.regression.calculate_regression(
      stock_paths_at_timestep, values,
      in_the_money, in_the_money_all, date_no,ML, noise
    )
    # print("Stock paths at timestep = ", stock_paths_at_timestep)
    # print("immediate exercise value = ", immediate_exercise_value)
    # print("Return values = ", return_values)
    return return_values


nb_paths = 16
nb_dates = 3
model = CustomStockModel(nb_paths, nb_dates, ML=False, noise=False)
payoff = MaxPut(strike=1.10)

pricer = LeastSquaresPricer(model, payoff)
a = pricer.price()
print(a)


text = '''Predicting values for the Gaussain noise path'''
print(text)
# print(COEFFICIENTS)
model = CustomStockModel(nb_paths, nb_dates, ML=False, noise=True)
pricer = LeastSquaresPricer(model, payoff)
a = pricer.price()
print("Optimal Price Gaussian Noise : ",a)


# text = '''Predicting values for the 9th generated path'''
# print(text)
# # print(COEFFICIENTS)
# model = CustomStockModel(nb_paths, nb_dates, ML=True)
# pricer = LeastSquaresPricer(model, payoff)
# a = pricer.price()
# print("Optimal Price for 9th path : ",a)


























# class LeastSquarePricerLaguerre(LeastSquaresPricer):
#   """Least Square Monte Carlo using weighted Laguerre basis functions."""
#   def __init__(self, model, payoff, nb_epochs=None, nb_batches=None,
#                train_ITM_only=True):
#     super().__init__(model, payoff, train_ITM_only=train_ITM_only)
#     self.regression = regression.LeastSquaresLaguerre(
#       model.nb_stocks)


# class LeastSquarePricerRidge(LeastSquaresPricer):
#   """Least Square Monte Carlo using ridge regression."""
#   def __init__(self, model, payoff, nb_epochs=None, nb_batches=None,
#                train_ITM_only=True, ridge_coeff=1.,):
#     super().__init__(model, payoff, train_ITM_only=train_ITM_only)
#     self.regression = regression.LeastSquaresRidge(
#       model.nb_stocks, ridge_coeff=ridge_coeff)
