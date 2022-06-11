from LSM import LeastSquaresPricer
from stock_model import CustomStockModel
from payoff import MaxPut

nb_paths = 8
nb_dates = 3
model = CustomStockModel(nb_paths, nb_dates)
payoff = MaxPut(strike=1.10)

pricer = LeastSquaresPricer(model, payoff)
a = pricer.price()
print(a)
