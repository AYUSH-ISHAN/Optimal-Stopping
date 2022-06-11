import numpy as np
longstaff = True#False
# noise = False#True

class CustomStockModel:
    def __init__(self, nb_paths, nb_dates, ML, noise):
        self.nb_path=nb_paths
        self.nb_dates=nb_dates
        self.nb_stocks=1
        self.ML = ML
        self.nois = noise

    def generate_paths(self):

        self.ones = np.ones((self.nb_path, 1), dtype=np.float32)
        self.arr = np.random.uniform(0,2,size=(self.nb_path, self.nb_dates))
        if longstaff and not self.nois:
            self.arr =np.array([[1.09, 1.08, 1.34],\
                                [1.16, 1.26, 1.54],\
                                [1.22, 1.07, 1.03],\
                                [0.93, 0.97, 0.92],\
                                [1.11, 1.56, 1.52],\
                                [0.76, 0.77, 0.90],\
                                [0.92, 0.84, 1.01],\
                                [0.88, 1.22, 1.34],\
                                [1.09972955, 1.20984266, 1.42786124],\
                                [1.30282465, 1.33379354, 1.54642148],\
                                [1.30811775, 1.18827482, 1.05821046],\
                                [1.00225371, 1.22542088, 1.06681238],
                                [1.17592812, 1.65280718, 1.72744582],\
                                [0.89931269, 0.85576233, 0.95650906],\
                                [0.9397596,  0.92159093, 1.03515879],\
                                [0.90082674, 1.26323588, 1.41475168]])
            # self.arr = np.array([[ 0.046685 ,   0.28383169 , 1.38935959],
            #                     [-0.32419235 , 0.65521504,  0.93998436],
            #                     [ 1.81223658 , 1.49837879,  0.71713394],
            #                     [ 1.1693588  , 1.8554999  , 2.06245975],
            #                     [ 0.03927709,  1.27342529, -0.22821531],
            #                     [ 1.26047525 , 1.66719946 , 0.53206636],
            #                     [ 1.65588861 , 0.31331137,  2.69502627],
            #                     [ 0.89529263 , 0.92070166 , 1.00316362]])
        
        if longstaff and self.nois:
            self.arr =np.array([[1.09, 1.08, 1.34],\
                                [1.16, 1.26, 1.54],\
                                [1.22, 1.07, 1.03],\
                                [0.93, 0.97, 0.92],\
                                [1.11, 1.56, 1.52],\
                                [0.76, 0.77, 0.90],\
                                [0.92, 0.84, 1.01],\
                                [0.88, 1.22, 1.34],\
                                [1.09972955, 1.20984266, 1.42786124],\
                                [1.30282465, 1.33379354, 1.54642148],\
                                [1.30811775, 1.18827482, 1.05821046],\
                                [1.00225371, 1.22542088, 1.06681238],\
                                [1.17592812, 1.65280718, 1.72744582],\
                                [0.89931269, 0.85576233, 0.95650906],\
                                [0.9397596,  0.92159093, 1.03515879],\
                                [0.90082674, 1.26323588, 1.41475168]])
            self.noise = np.random.normal(0,0.2,self.nb_dates*self.nb_path)
            self.noise = np.reshape(self.noise, (self.nb_path, self.nb_dates))
            self.arr += abs(self.noise)
            # print(self.arr)
        if self.ML:
            self.path = np.array([1.0, 0.98, 1.02, 1.34])
            return self.path, self.ML
        a =  np.concatenate((self.ones, self.arr), axis=1)
        print(a)
        return a, self.ML, self.nois

# a = CustomStockModel(8,3)
# print(a.generate_paths())