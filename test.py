import csv
from sklearn import preprocessing
import numpy as np

if __name__ == '__main__':
    tst = [
        [1,2,1],
        [2,4,1],
        [3,6,1]
    ]
    print(preprocessing.scale(np.array(tst,dtype=float).tolist()))