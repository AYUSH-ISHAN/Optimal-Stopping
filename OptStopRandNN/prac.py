###  Use of flags 

from absl import flags
from absl import app
import numpy as np
FLAGS = flags.FLAGS

flags.DEFINE_list("nb_stocks", None, "List of number of Stocks")

def main(argv):
    a = [int(nb) for nb in FLAGS.nb_stocks or []]
    print(a[1])
    #a = 0.001
    # print(np.finfo(float).eps)
    # a = [1,2,3,4]
    # print(a[~1])

app.run(main)
