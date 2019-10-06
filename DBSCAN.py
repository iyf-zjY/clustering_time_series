from sklearn import cluster
import numpy as np
from matplotlib import pyplot as plt
import math
import csv
import Kmeans_cDTW_DBA

if __name__ == '__main__':
    DBSCAN = cluster.DBSCAN(eps=3.0, metric=lambda x,y:Kmeans_cDTW_DBA.cDTW(x, y))
    time_interval = '171225-180126'
    datadir = 'cluster_ans/for_train/z-scored_' + time_interval + '_wm.csv'
    Data = []
    #num_cluster = 8
    with open(datadir, 'r') as f:
        Data = list(csv.reader(f))
    Data = np.array(Data, dtype=float)
    y_label = DBSCAN.fit_predict(Data)
    print(y_label)
    idx = [i for i in range(1, Data.shape[1] + 1)]
    plt.figure()
    plt_d = []
    for i,t in enumerate(y_label):
        if t == 0:
            plt_d.append(Data[i])
        plt.plot(idx,Data[i])
    plt.show()
    #随便试了试 效果不太好