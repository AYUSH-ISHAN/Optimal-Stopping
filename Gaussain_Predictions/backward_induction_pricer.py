"""Base class that computes the American option price using backward recusrion.

All algorithms that are using a backward recusrion such as
LSM (Least squares Monte Carlo),
NLSM (Neural Least squares Monte Carlo),
RLSM (Randomized Least squares Monte Carlo)
and DOS (Deep Optimal Stopping) are inherited from this class.
"""

import numpy as np
import time


class AmericanOptionPricer:
  """Computes the price of an American Option using backward recusrion.
  """
  def __init__(self, model, payoff, use_rnn=False, train_ITM_only=True,
               use_path=False):

    #class model: The stochastic process model of the stock (e.g. Black Scholes).
    self.model = model

    #class payoff: The payoff function of the option (e.g. Max call).
    self.payoff = payoff

    #bool: randomized neural network is replaced by a randomized recurrent NN.
    self.use_rnn = use_rnn

    #bool: x_k is replaced by the entire path (x_0, .., x_k) as input of the NN.
    self.use_path = use_path

    #bool: only the paths that are In The Money (ITM) are used for the training.
    self.train_ITM_only = train_ITM_only

  def calculate_continuation_value(self):
    """Computes the continuation value of an american option at a given date.

    All algorithms that inherited from this class (AmericanOptionPricer) where
    the continuation value is approximated by basis functions (LSM),
    neural networks (NLSM), randomized neural networks (RLSM), or
    recurrent randomized neural networks (RRLSM) only differ by a this function.

    The number of paths determines the size of the arrays.

    Args:
      values (np array): the option price of the next date (t+1).
      immediate_exercise_value (np array): the payoff evaluated with the current
       stock price (date t).
      stock_paths_at_timestep (np array): The stock price at the current date t.

    Returns:
      np array: the option price at current date t if we continue until next
       date t+1.
    """
    raise NotImplementedError

  def stop(self, stock_values, immediate_exercise_values,
           discounted_next_values,date_no,ML,noise, h=None): # h is stock_paths_at_timestep
    """Returns a vector of {0, 1}s (one per path) for a given data, where:
        1 means stop, and
        0 means continue.

    The optimal stopping algorithm (DOS) where the optimal stopping is
    approximated by a neural network has a different function "stop".
    """
    if not ML:
      stopping_rule = np.zeros(len(stock_values))  
    else:
      #print(stock_values.shape)
      print(stock_values)
      stopping_rule = np.zeros(1) 
    
    continuation_values = self.calculate_continuation_value(
        discounted_next_values,
        immediate_exercise_values, stock_values, date_no,ML,noise)
    print("continuation : ", continuation_values)
    which = (immediate_exercise_values > continuation_values) & \
               (immediate_exercise_values > np.finfo(float).eps)  
              # np.finfo(float).eps --> machine limits for floating point types. (machines epsilon value)
    # print("which : ", which)
    stopping_rule[which] = 1
    return stopping_rule

  def price(self):
    """It computes the price of an American Option using a backward recusrion.
    """
    model = self.model
    t1 = time.time()
    stock_paths, ML, noise = self.model.generate_paths()
    
    if not ML:
      disc_factor = 0.94176  # @ 6 % riskless rate   (e^(-0.06))
      immediate_exercise_value = self.payoff.eval(stock_paths[:, -1])
      values = immediate_exercise_value
      for date in range(stock_paths.shape[1] - 2, 0, -1):   # backward recursion..(going backward)
        '''"-1" index already done before'''
        immediate_exercise_value = self.payoff.eval(stock_paths[:, date])
        
        h = None   # h is stock_paths_at_timestep (Not needed in our case)
        stopping_rule = self.stop(
            stock_paths[:, date], immediate_exercise_value,
            values*disc_factor,date,ML,noise, h=h)
        which = stopping_rule > 0.5
        values[which] = immediate_exercise_value[which]  #  taking only those which are in money
        values[~which] *= disc_factor  # ~which -> index counting starting from last. 
        print(f"Stpping Rule at time {date} : {stopping_rule}")
        print(f"values at time {date} : {values}")
        '''assigned "0" to the last element.'''
      payoff_0 = self.payoff.eval(stock_paths[:, 0])[0]  # finds payoff at time=0
      return max(payoff_0, np.mean(values) * disc_factor)  # previosuly it was values[self.split:]

    else:
      disc_factor = 1.06183655  # e^(+0.06)
      immediate_exercise_value = self.payoff.eval(stock_paths[0])
      values = immediate_exercise_value
      print(immediate_exercise_value, values)
      # print(stock_paths.shape[0])
      for dates in range(1, stock_paths.shape[0]-1):  
          print(dates)
          '''"-1" index already done before'''
          # print("hi")
          immediate_exercise_value = self.payoff.eval(stock_paths[dates])
          h = None   # h is stock_paths_at_timestep (Not needed in our case)
          stopping_rule = self.stop(
              stock_paths[dates], immediate_exercise_value,
              values*disc_factor,dates,ML, h=h)
          which = stopping_rule > 0.5
          # print("whichhhhhhhhhhhhhhhhh  : ", which)  # [True, False, True, False]
          # print("stoppppppppppppppp   : ", stopping_rule) # [1, 0, 1, 0]
          # print("valueeeeeeeeeee : ", values)  # array of values
          # print("valueeeeeeeeeee : ", values[which])  # only those values whose index is true
          # print("IMMEDIATE : ", immediate_exercise_value)
          if which:
            values = immediate_exercise_value
          else:
            values *= disc_factor
          
          '''assigned "0" to the last element.'''
      # print(self.payoff.eval(stock_paths[0]))
      payoff_0 = self.payoff.eval(stock_paths[0])  # finds payoff at time=0
      print("final values : ", values)
      return max(payoff_0, np.mean(values) * disc_factor)  # previosuly it was values[self.split:]

