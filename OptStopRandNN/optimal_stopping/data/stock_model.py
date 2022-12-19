""" Underlying model of the stochastic processes that are used:
- Black Scholes
- Heston
- Fractional Brownian motion
"""

import math
import numpy as np
import matplotlib.pyplot as plt
from fbm import FBM

import joblib


NB_JOBS_PATH_GEN = 1

class Model:
  def __init__(self, drift, volatility, spot, nb_stocks,  nb_paths, nb_dates,
         maturity, **keywords):
    self.drift = drift
    self.volatility = volatility
    self.spot = spot
    self.nb_stocks = nb_stocks
    self.nb_paths = 1#nb_paths
    self.nb_dates = nb_dates
    self.maturity = maturity
    self.dt = self.maturity / self.nb_dates
    self.df = math.exp(-drift * self.dt)

  def disc_factor(self, date_begin, date_end):
    time = (date_end - date_begin) * self.dt
    return math.exp(-self.drift * time)

  def drift_fct(self, x, t):
    raise NotImplemented()

  def diffusion_fct(self, x, t, v=0):
    raise NotImplemented()

  def generate_one_path(self):
      raise NotImplemented()

  def generate_paths(self, nb_paths=None):
    """Returns a nparray (nb_paths * nb_stocks * nb_dates) with prices."""
    nb_paths = nb_paths or self.nb_paths
    if NB_JOBS_PATH_GEN > 1:
        return np.array(
            joblib.Parallel(n_jobs=NB_JOBS_PATH_GEN, prefer="threads")(
                joblib.delayed(self.generate_one_path)()
                for i in range(nb_paths)))
    else:
        # path = np.array([self.generate_one_path() for i in range(nb_paths)])
        # print(path)
        return np.array([self.generate_one_path() for i in range(nb_paths)])

    # path = np.array([1., 1.09, 1.08, 1.34],
    #         [1., 1.16, 1.26, 1.54],
    #         [1., 1.22, 1.07, 1.03],
    #         [1., 0.93, 0.97, 0.92],
    #         [1., 1.11, 1.56, 1.52],
    #         [1., 0.76, 0.77, 0.90],
    #         [1., 0.92, 0.84, 1.01],
    #         [1., 0.88, 1.22, 1.34])
    #return path


class BlackScholes(Model):
  def __init__(self, drift, volatility, nb_paths, nb_stocks, nb_dates, spot,
         maturity, dividend=0, **keywords):
    super(BlackScholes, self).__init__(drift=drift - dividend, volatility=volatility,
             nb_stocks=nb_stocks, nb_paths=nb_paths, nb_dates=nb_dates,
             spot=spot, maturity=maturity)
    self.drift = drift   # included for dicounting as drift can be != drift

  def drift_fct(self, x, t):
    del t
    return self.drift * x

  def diffusion_fct(self, x, t, v=0):
    del t
    return self.volatility * x

  def generate_one_path(self):
    """Returns a nparray (nb_stocks * nb_dates) with prices."""
    path = np.empty((self.nb_stocks, self.nb_dates+1))
    path[:, 0] = self.spot
    for k in range(1, self.nb_dates+1):
      random_numbers = np.random.normal(0, 1, self.nb_stocks)
      dW = random_numbers*np.sqrt(self.dt)
      previous_spots = path[:, k - 1]
      diffusion = self.diffusion_fct(previous_spots, (k) * self.dt)
      path[:, k] = (
          previous_spots
          + self.drift_fct(previous_spots, (k) * self.dt) * self.dt
          + np.multiply(diffusion, dW))
    print("pattttttttttthhhh  : ", path/100)
    return path/100 #path


class FractionalBlackScholes(BlackScholes):
  def __init__(self, drift, volatility, hurst, nb_paths, nb_stocks, nb_dates, spot,
         maturity, dividend=0, **keywords):
    super(FractionalBlackScholes, self).__init__(drift, volatility, nb_paths, nb_stocks, nb_dates, spot, maturity, dividend, **keywords)
    self.drift = drift
    self.hurst = hurst
    self.fBM = FBM(n=nb_dates, hurst=self.hurst, length=maturity, method='hosking')


  def generate_one_path(self):
    """Returns a nparray (nb_stocks * nb_dates) with prices."""
    path = np.empty((self.nb_stocks, self.nb_dates+1))
    fracBM_noise = np.empty((self.nb_stocks, self.nb_dates))
    path[:, 0] = self.spot
    for stock in range(self.nb_stocks):
      fracBM_noise[stock, :] = self.fBM.fgn()
    for k in range(1, self.nb_dates+1):
      previous_spots = path[:, k - 1]
      diffusion = self.diffusion_fct(previous_spots, (k) * self.dt)
      path[:, k] = (
          previous_spots
          + self.drift_fct(previous_spots, (k) * self.dt) * self.dt
          + np.multiply(diffusion, fracBM_noise[:,k-1]))
    # print("path",path/100)
    return path/100 # path
