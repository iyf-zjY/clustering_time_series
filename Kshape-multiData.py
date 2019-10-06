import numpy as np
import os
from sklearn import preprocessing
import csv
import math
from matplotlib import pyplot as plt
time_interval = '190211-190410'
time_interval1 = '190212-190411'
time_interval2 = '190213-190412'
datadir = 'cluster_ans/for_train/'+'week9_zscored/wk9_zscored'
data_raw = 'cluster_data/'+time_interval+'/close.csv'
stock_idx_file = 'cluster_ans/for_train/'+'week9_zscored/stock_idx_'+time_interval+'.txt'
ans_dir = 'cluster_ans/train_ans/'


def NCCc(w,m,x,y):  # y is aligned towards x
    k = w - m
    if k >= 0:
        t_sum = 0
        for i in range(m - k):
            t_sum += x[i + k] * y[i]
        if np.linalg.norm(x,ord=2)<=1e-20 or np.linalg.norm(y,ord=2)<=1e-20:
            t_sum = 0
        else:
            t_sum = t_sum/math.sqrt(np.linalg.norm(x,ord=2)*np.linalg.norm(y,ord=2))
        return t_sum
    else:
        t_sum = 0
        for i in range(m + k):
            t_sum += y[i-k] * x[i]
        if np.linalg.norm(x, ord=2) <= 1e-10 or np.linalg.norm(y, ord=2) <= 1e-10:
            t_sum = 0
        else:
            t_sum = t_sum / math.sqrt(np.linalg.norm(x, ord=2) * np.linalg.norm(y, ord=2))
        return t_sum

def SBD(x,y_s):
    m = len(x)
    NCCc_seq = [[] for i in range(len(y_s))]
    # 以前是range(1,2m) 现在是左右移动三天的话，应该是 +-3
    #for i in range(1, 2 * m):
    for i in range(max(1,m-3),min(2*m,m+3)):
        for j,y in enumerate(y_s):
            NCCc_seq[j].append(NCCc(i,m,x,y))
    #外层是不同的序列，内层是序列的不同对齐位置
    value_outer = [max(t) for t in NCCc_seq]
    index_outer = [t.index(max(t)) for t in NCCc_seq]

    value = max(value_outer)
    index = value_outer.index(max(value_outer))
    outer_idx = index
    index = index_outer[index] + max(1,m-3)
    dist = 1 - value
    shift = index - m
    y_ = []
    if shift >= 0:
        for t in range(shift):
            y_.append(0)
        for t in range(m-shift):
            y_.append(y_s[outer_idx][t])
    else:
        for t in range(-shift,m):
            y_.append(y_s[outer_idx][t])
        for t in range(-shift):
            y_.append(0)
    return dist,shift,y_

def kshape_centroid(mem,A,k,cent_vector):
    window_size = len(A)
    data_k=[]
    for i in range(len(mem)):
        if mem[i] == k:
            if math.fabs(np.sum(cent_vector))==0:
                #这种情况下 三个日期中随便选一个当中心 这里是第一天
                opt_v = [A[0][i][t] for t in range(A[0].shape[1])]
            else:
                tmp,tmp1,opt_v = SBD(cent_vector,[t[i] for t in A])
            data_k.append(opt_v)
    if len(data_k) == 0:
        return np.zeros(A[0].shape[1])
    data_k = np.array(data_k)
    #由于数据的预处理不再是z-scored,并且瑞丽熵求解不要求z-scored
    #如果处理z-scored的数据，可以直接复用
    data_k = np.array(preprocessing.scale(data_k.T.tolist())).T
    S = np.dot(data_k.T,data_k)
    Q = np.eye(data_k.shape[1]) - (1/data_k.shape[1]) * np.ones((data_k.shape[1],data_k.shape[1]))
    M = np.dot(Q,S)
    M = np.dot(M,Q)
    #M = S
    e_val_s,e_vec_s = np.linalg.eig(M)
    e_val_s = list(e_val_s)
    #print(e_val_s)
    #e_vec_s = list(e_vec_s)
    e_valm = max(e_val_s)
    e_vec = e_vec_s[:,e_val_s.index(max(e_val_s))]
    e_vec = [i.real for i in e_vec]
    dis_1 = [data_k[0][i]-e_vec[i] for i in range(data_k.shape[1])]
    dis_2 = [data_k[0][i]+e_vec[i] for i in range(data_k.shape[1])]
    if np.linalg.norm(dis_1) >= np.linalg.norm(dis_2):
        e_vec = [-t for t in e_vec]
    e_vec = preprocessing.scale(e_vec)
    return e_vec

def Kshape(X,K): #X:window_size * m(stock_num)*n(days) (type:numpy)     k:number of clusters  多个矩阵的形状是一样的
    m = X[0].shape[0]
    n = X[0].shape[1]
    iter = 0
    mem = list(np.random.rand(m))
    mem = [i*K for i in mem]
    mem = [int(math.floor(i)) for i in mem]   #index:which cluster the data should belong to
    cent = np.zeros((K,n),dtype=float)
    while iter <= 100:
        print(iter)
        print(mem)
        prev_mem = [t for t in mem]          #deep copy
        for i in range(K):
            cent[i] = kshape_centroid(mem,X,i,cent[i])
        D = np.zeros((m,K),dtype=float)
        for i in range(m):
            for j in range(K):
                d_,t1,t2 = SBD(cent[j],[t[i] for t in X])
                D[i][j] = d_
        # time complexity: m*K*2m*O(m)  if with FFT and IFFT
        D = D.tolist()
        mem = [t.index(min(t)) for t in D]
        if np.linalg.norm([(prev_mem[i]-mem[i]) for i in range(len(mem))]) == 0:
            break
        iter += 1
    return mem,cent

if __name__ == '__main__':

    Data = []
    Data_2 = []
    Data_3 = []
    num_cluster = 10
    data1 = datadir + time_interval+'.csv'
    data2 = datadir + time_interval1 + '.csv'
    data3 = datadir + time_interval2 + '.csv'
    with open(data1,'r') as f:
        Data = list(csv.reader(f))
    with open(data2,'r') as f:
        Data_2 = list(csv.reader(f))
    with open(data3,'r') as f:
        Data_3 = list(csv.reader(f))
    with open(data_raw,'r') as f:
        Data_raw = list(csv.reader(f))
    Data_raw = Data_raw[1:]
    Data_raw = [t[1:] for t in Data_raw]
    for i in range(len(Data_raw)):
        Data_raw[i] = np.array(Data_raw[i],dtype=float)
        #print(Data_raw[i])
        m = Data_raw[i].mean()
        Data_raw[i] = [t-m for t in Data_raw[i]]
    Data = np.array(Data,dtype=float)
    Data_2 = np.array(Data_2,dtype=float)
    Data_3 = np.array(Data_3,dtype=float)
    Data = [Data,Data_2,Data_3]
    Data_raw = np.array(Data_raw,dtype=float)
    plt.figure()
    idx = [i for i in range(1, Data[0].shape[1] + 1)]
    for i in range(Data[0].shape[0]):
        plt.plot(idx, Data[0][i])
    plt.show()
    os.system("pause")
    mem,cent = Kshape(Data,num_cluster)

    #plot and write ans to files
    mem_file = 'mem_'+time_interval+'.csv'
    cent_file = 'cent_'+time_interval+'.csv'

    for tt in range(num_cluster):
        data_1 = []
        plt.figure()
        #idx = [i for i in range(1,Data[0].shape[1]+1)]
        idx =[i for i in range(1,Data_raw.shape[1]+1)]
        for i in range(Data_raw.shape[0]):
            if mem[i] == tt:
                plt.plot(idx,Data_raw[i])
        plt.show()

    stock_idx = []
    with open(stock_idx_file,'r') as f:
        for line in f.readlines():
            stock_idx.append(line.replace('\n',''))
    for i,tmp in enumerate(stock_idx):
        stock_idx[i] = [stock_idx[i],mem[i]]

    with open(ans_dir+mem_file,'w',newline='') as f:
        writer = csv.writer(f)
        writer.writerows(stock_idx)

    with open(ans_dir+cent_file,'w',newline='') as f:
        writer = csv.writer(f)
        for ct in range(cent.shape[0]):
            writer.writerow([ct]+list(cent[ct]))





