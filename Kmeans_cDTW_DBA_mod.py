import numpy as np
import csv
from matplotlib import pyplot as plt
import math

#距离度量：DTW
#中心向量计算：迭代DBA
#依然假设所有序列的长度都相等
#multiData版本
#mod版本：正在进行中，加入boost以及其他
time_interval = '190211-190410'
time_interval1 = '190212-190411'
time_interval2 = '190213-190412'
datadir = 'cluster_ans/for_train/'+'week9_zscored/wk9_zscored'
stock_idx_file = 'cluster_ans/for_train/'+'week9_zscored/stock_idx_'+time_interval+'.txt'
ans_dir = 'cluster_ans/train_ans/'
ans_prefix = '190513-'

def dis(i,j):
    return (i-j)**2

def argpath_min(a,b,c):
    if a<b:
        if c<a:
            return 2
        else:
            return 0
    else:
        if c<b:
            return 2
        else:
            return 1


def cDTW(x,y):
    #we assume the two series is equal length
    #dis = np.zeros((len(x),len(y)))
    #dis = [[(x[i]-t)**2 for t in y] for i in range(len(x))]
    acc = np.empty((len(x),len(y)))
    for i in range(len(x)):
        acc[i] = 1e20
    #惩罚歪的项 这是因为我们的序列都是等长的，原本的dtw是惩罚正的项 现在还没设置
    w = [1,1,1]
    #偏移最大值3
    M = 3

    acc[0][0] = dis(x[0],y[0])
    for i in range(1,len(x)):
        acc[0][i] = acc[0][i-1] + dis(x[0],y[i])
        acc[i][0] = acc[i-1][0] + dis(x[i],y[0])
    for i in range(1,acc.shape[0]):
        for j in range(max(1,i-M),min(acc.shape[1],i+M)):
            acc[i][j] = min(acc[i-1][j]*w[0],acc[i-1][j-1]*w[1],acc[i][j-1]*w[2]) + dis(x[i],y[j])
    #print(acc)
    return math.sqrt(acc[len(x)-1][len(y)-1])

def cDTW_s(x,y_s):
    Dis = []
    for y in y_s:
        Dis.append(cDTW(x,y))
    return Dis.index(min(Dis)),min(Dis)
    # print(acc)

#做一步迭代
def DBA_iteration(avg,seqs):
    #M存每个位置的所有序列进来的对应值，算平均用
    M = []
    #路径方向权重
    w = [1, 1, 1]
    for i in range(seqs.shape[1]):
        M.append([])
    cost_matrix = np.empty((len(avg),seqs.shape[1]))
    path_matrix = np.empty((len(avg),seqs.shape[1]))

    for k in range(seqs.shape[0]):
        seq = seqs[k]
        cost_matrix[0][0] = dis(avg[0],seq[0])
        #标记匹配路径头
        path_matrix[0][0] = -1
        #cos_matrix[x][y] x：avg, y:seq 其实应该也没影响
        for i in range(1, len(seq)):
            cost_matrix[0][i] = cost_matrix[0][i - 1] + dis(avg[0], seq[i])
            cost_matrix[i][0] = cost_matrix[i - 1][0] + dis(avg[i], seq[0])
            path_matrix[0][i] = 1
            path_matrix[i][0] = 2   #0513
        #代码成立仅当序列长度相等

        for i in range(1,len(avg)):
            for j in range(1,len(seq)):
                idx = argpath_min(cost_matrix[i-1][j-1],cost_matrix[i][j-1],
                                  cost_matrix[i-1][j])
                path_matrix[i][j] = idx
                ret = 0
                if idx == 0:
                    ret = cost_matrix[i-1][j-1]
                elif idx == 1:
                    ret = cost_matrix[i][j-1]
                else:
                    ret = cost_matrix[i-1][j]
                cost_matrix[i][j] = ret + dis(avg[i],seq[j])

        i = len(avg) - 1
        j = len(seq) - 1
        while True:
            M[i].append(seq[j])
            if path_matrix[i][j] == 0:
                i -= 1
                j -= 1
            elif path_matrix[i][j] == 1:
                j -= 1
            elif path_matrix[i][j] == 2:
                i -= 1
            else:
                break

    ret_avg = []
    for i in range(len(avg)):
        ret_avg.append(np.array(M[i]).mean())
    return ret_avg

def DBA_chooseOne(avg,seqs):
    Dis = []
    for seq in seqs:
        Dis.append(cDTW(avg,seq))
    return Dis.index(min(Dis))

def kmeans(X,K): #X:window_size * m(stock_num)*n(days) (type:numpy)     k:number of clusters  多个矩阵的形状是一样的
    m = X[0].shape[0]
    n = X[0].shape[1]
    iter = 0
    mem = list(np.random.rand(m))
    mem = [i*K for i in mem]
    mem = [int(math.floor(t)) for t in mem]
    cent = np.zeros((K,n),dtype = float)
    D = np.zeros((m, K), dtype=float)
    while iter<=100:
        print(iter)
        print(mem)
        prev_mem = [t for t in mem]
        for i in range(K):
            seqs =[]
            for t in range(m):
                if mem[t] == i:           #190513 K->i
                    seqs.append([tt[t] for tt in X])
            if len(seqs) == 0:
                continue
            #从每个stock的多个series中找一个距离最近滴
            seqs_ = [seqs[t][DBA_chooseOne(cent[i],seqs[t])] for t in range(len(seqs))]
            seqs_ = np.array(seqs_)
            cent[i] = DBA_iteration(cent[i],seqs_)

        for i in range(m):
            for j in range(K):
                _, D[i][j] = cDTW_s(cent[j],[t[i] for t in X])
        if iter == 0:
            D = D.tolist()
        mem = [t.index(min(t)) for t in D]
        if np.linalg.norm([(prev_mem[i]-mem[i]) for i in range(len(mem))]) == 0:
            break
        iter += 1
    return mem, cent

def read_and_run():
    Data = []
    Data_2 = []
    Data_3 = []
    num_cluster = 10
    data1 = datadir + time_interval + '.csv'
    data2 = datadir + time_interval1 + '.csv'
    data3 = datadir + time_interval2 + '.csv'
    with open(data1,'r') as f:
        Data = list(csv.reader(f))
    with open(data2,'r') as f:
        Data_2 = list(csv.reader(f))
    with open(data3,'r') as f:
        Data_3 = list(csv.reader(f))

    Data_1 = np.array(Data,dtype=float)
    Data_2 = np.array(Data_2,dtype=float)
    Data_3 = np.array(Data_3,dtype=float)
    Data = [Data_1,Data_2,Data_3]

    mem,cent = kmeans(Data,num_cluster)

    mem_file = 'mem_' + time_interval + '.csv'
    cent_file = 'cent_' + time_interval + '.csv'

    for t in range(num_cluster):
        data_plt = []
        plt.figure()
        idx = [i for i in range(1,Data_1.shape[1]+1)]
        for i in range(Data_1.shape[0]):
            if mem[i] == t:
                plt.plot(idx,Data_1[i])
        plt.show()

    #存
    stock_idx = []
    with open(stock_idx_file, 'r') as f:
        for line in f.readlines():
            stock_idx.append(line.replace('\n', ''))

    for i,tmp in enumerate(stock_idx):
        stock_idx[i] = [stock_idx[i],mem[i]]
    with open(ans_dir+ans_prefix+mem_file,'w',newline='') as f:
        writer = csv.writer(f)
        writer.writerows(stock_idx)
    with open(ans_dir+ans_prefix+cent_file,'w',newline='') as f:
        writer = csv.writer(f)
        for ct in range(cent.shape[0]):
            writer.writerow([ct] + list(cent[ct]))


if __name__ == '__main__':
    read_and_run()