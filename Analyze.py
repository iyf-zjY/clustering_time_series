import numpy as np
import csv
from matplotlib import pyplot as plt
import math
from xml.etree import ElementTree as ET
import os
import run_preprocess
from Kmeans_cDTW_DBA_mul import cDTW
from tqdm import tqdm
from Kmeans_cDTW_DBA_mul import DBA_iteration


#根据聚类的结果，对未来的进行分类
conf_file = 'AnalyzeConf.xml'
clustering_ans_dir = 'cluster_ans/train_ans/{begin}-{end}/'
predict_dir = 'cluster_data/{begin}-{end}/'
predict_attr = 'close'
mem_file = 'mem_{begin}-{end}.csv'
cent_file = 'cent_{begin}-{end}.csv'
up_and_down_file = 'up_and_down_{begin}-{end}.csv'
num_cluster = 0
mem_dict = {}
mem_list = []
mem_index_dict = {}
train_begin_date = ET.parse(conf_file).getroot()[0].text.strip()
train_end_date = ET.parse(conf_file).getroot()[1].text.strip()
predict_begin_date = ET.parse(conf_file).getroot()[2].text.strip()
predict_end_date = ET.parse(conf_file).getroot()[3].text.strip()
predict_prefix = ET.parse(conf_file).getroot()[4].text.strip()
predict_idx_prefix = ET.parse(conf_file).getroot()[5].text.strip()
predict_file = predict_prefix + '{begin}-{end}.csv'
predict_idx_file = predict_idx_prefix + '{begin}-{end}.txt'
#对于聚类结果的再筛选
cluster_dir = 'cluster_data/{begin}-{end}/'
def Euclidean_dis(seq1,seq2):
    if len(seq1)!=len(seq2):
        print("error format of sequences")
        os.system("pause")
    dis = 0
    for i in range(len(seq1)):
        dis += (seq1[i]-seq2[i])**2
    return dis

#本想着精炼结果，比如剔除一些类里边距离过远的？？还没弄完
def refine_ans():
    c_Data = []
    cluster_dir = 'cluster_ans\\for_train\week9_zscored\\'
    cluster_file = 'wk9_zscored190211-190410.csv'
    with open(cluster_dir+cluster_file,'r') as f:
        lines = csv.reader(f)
        for line in lines:
            c_Data.append([float(t) for t in line])
    cent = []
    with open(clustering_ans_dir + cent_file, 'r') as f:
        cents = csv.reader(f)
        for c in cents:
            cent.append([float(tt) for tt in c])
    mem_1 = []
    with open(clustering_ans_dir + mem_file, 'r') as f:
        mems = csv.reader(f)
        for line in mems:
            mem_1.append(int(line[1]))
    global num_cluster
    x_idx = [i for i in range(1, len(c_Data[0]) + 1)]
    for i in range(num_cluster):
        data_i = []
        for j in range(len(mem_1)):
            if mem_1[j] == i:
                data_i.append(c_Data[j])
        # Cn2
        pair_mindis = [[1], [1]]
        mindis = 1e20
        pair_maxdis = [[1], [1]]
        maxdis = 0
        for t1, seq1 in enumerate(data_i):
            for seq2 in data_i[t1 + 1:]:
                thisdis = cDTW(seq1, seq2)
                if thisdis < mindis:
                    mindis = thisdis
                    pair_mindis = [seq1, seq2]
                if thisdis > maxdis:
                    maxdis = thisdis
                    pair_maxdis = [seq1, seq2]
        plt.figure()
        plt.plot(x_idx, pair_mindis[0])
        plt.plot(x_idx, pair_mindis[1])
        plt.show()
        plt.plot(x_idx, pair_maxdis[0])
        plt.plot(x_idx, pair_maxdis[1])
        plt.show()

def draw_pic_on_predictData():
    idx = []
    with open(predict_dir+predict_idx_file,'r') as f:
        for line in f.readlines():
            idx.append(line[1:].strip())

    Data = {}  #预测的
    Data_c = {}  #训练集上的
    with open(predict_dir+predict_file,'r') as f:
        stocks = csv.reader(f)
        for i,stock in enumerate(stocks):
            Data[idx[i]] = [float(tmp) for tmp in stock]
    cluster_dir = 'cluster_ans\\for_train\week9_zscored\\'
    cluster_file = 'wk9_zscored190211-190410.csv'
    with open(cluster_dir + cluster_file, 'r') as f:
        lines = csv.reader(f)
        for i,line in enumerate(lines):
            Data_c[idx[i]] = [float(tmp) for tmp in line]
    global  num_cluster
    x_idx = [i for i in range(1,len(Data[idx[0]])+1)]
    x_idx_c = [i for i in range(1,len(Data_c[idx[0]])+1)]
    '''
    #先画一个总图
    plt.figure()
    for it in mem_dict.items():
        plt.plot(x_idx,Data[it[0]])
    plt.show()


    for i in range(num_cluster):
        plt.figure()
        for it in mem_dict.items():
            if it[1]==i:
                plt.plot(x_idx,Data[it[0]])
        plt.show()
    '''
    #画"对比图" 在同一类中
    have_plted = {}
    for stock in idx:
        try:
            a = have_plted[mem_dict[stock]]
        except:
            plt.figure()
            print(mem_dict[stock])
            have_plted[mem_dict[stock]] = 1
            for compare in idx[idx.index(stock)+1:]:
                if mem_dict[compare] == mem_dict[stock]:
                    plt.plot(x_idx, Data[stock])
                    plt.plot(x_idx,Data[compare])
            plt.show()
            for compare in idx[idx.index(stock)+1:]:
                if mem_dict[compare] == mem_dict[stock]:
                    plt.plot(x_idx_c,Data_c[stock])
                    plt.plot(x_idx_c,Data_c[compare])
            plt.show()

#函数名字xjb起的，就是找一个类里边距离最大和最小的
def numerical_compare():
    c_Data = []
    stock_idx = []
    cluster_dir = 'cluster_ans\\for_train\week9_zscored\\'
    cluster_file = 'wk9_zscored190211-190410.csv'
    with open(cluster_dir + cluster_file, 'r') as f:
        lines = csv.reader(f)
        for line in lines:
            c_Data.append([float(t) for t in line])
    with open(predict_dir+predict_idx_file,'r') as f:
        lines = f.readlines()
        for line in lines:
            stock_idx.append(line.strip())

    all_Data = [] #这是预测的
    with open(predict_dir+predict_file,'r') as f:
        lines = csv.reader(f)
        for stock in lines:
            all_Data.append([float(t) for t in stock])
    '''
    cent = []
    with open(clustering_ans_dir+cent_file,'r') as f:
        cents = csv.reader(f)
        for c in cents:
            cent.append([float(tt) for tt in c])
    '''
    mem_1 = []
    with open(clustering_ans_dir+mem_file,'r') as f:
        mems = csv.reader(f)
        for line in mems:
            mem_1.append(int(line[1]))

    global num_cluster
    x_idx = [i for i in range(1,len(all_Data[0])+1)]
    x_idx_c = [i for i in range(1,len(c_Data[0])+1)]
    all_dis = []
    all_dis_predict = []
    for i in tqdm(range(len(c_Data))):
        for j in range(i+1,len(c_Data)):
            all_dis.append(cDTW(c_Data[i],c_Data[j]))
            all_dis_predict.append(cDTW(all_Data[i],all_Data[j]))
   # plt.hist(all_dis,bins=50)
    for i in range(num_cluster):
        dis_i = []
        dis_i_predict = []
        cs_idx = []
        # 统计一下子：一对股票在训练数据的DTW距离与在预测数据上的DTW距离的关系
        print(i)
        data_i = []
        c_data_i = []
        plt.hist(all_dis, bins=50,color="#FF0000",alpha=.9)
        for j in range(len(mem_1)):
            if mem_1[j] == i:
                cs_idx.append(j)
                data_i.append(all_Data[j])
                c_data_i.append(c_Data[j])
        for t1 in range(len(c_data_i)):
            for t2 in range(t1+1,len(c_data_i)):
                dis_i.append(cDTW(c_data_i[t1],c_data_i[t2]))
                dis_i_predict.append(cDTW(data_i[t1],data_i[t2]))
        plt.hist(dis_i,bins=50,color="#C1F320",alpha=.5)
        plt.show()
        #预测
        plt.hist(all_dis_predict, bins=50, color="#FF0000", alpha=.9)
        plt.hist(dis_i_predict, bins=50, color="#C1F320",alpha=.5)
        plt.show()

        ''' 
        #Cn2
        pair_mindis = [[1],[1]]
        pair_mindis_idx_c = [0,0]
        mindis = 1e20
        pair_maxdis = [[1],[1]]
        pair_maxdis_idx_c = [0,0]
        maxdis = 0
        for t1,seq1 in enumerate(data_i):
            for t2,seq2 in enumerate(data_i[t1+1:]):
                thisdis = cDTW(seq1,seq2)
                if thisdis < mindis:
                    mindis = thisdis
                    pair_mindis = [seq1,seq2]
                    pair_mindis_idx_c = [cs_idx[t1],cs_idx[t2]]
                if thisdis > maxdis:
                    maxdis = thisdis
                    pair_maxdis = [seq1,seq2]
                    pair_maxdis_idx_c = [cs_idx[t1],cs_idx[t2]]
        plt.figure()
        #预测
        plt.plot(x_idx,pair_mindis[0])
        plt.plot(x_idx,pair_mindis[1])
        plt.show()
        #原始训练集
        plt.plot(x_idx_c,c_Data[pair_mindis_idx_c[0]])
        plt.plot(x_idx_c,c_Data[pair_mindis_idx_c[1]])
        plt.show()

        #预测
        plt.plot(x_idx,pair_maxdis[0])
        plt.plot(x_idx,pair_maxdis[1])
        plt.show()
        #原始训练集
        plt.plot(x_idx_c,c_Data[pair_maxdis_idx_c[0]])
        plt.plot(x_idx_c,c_Data[pair_maxdis_idx_c[1]])
        plt.show()
        '''

def distance_compare():
    c_Data= []
    cluster_dir = 'cluster_data\\180901-181201\\'
    cluster_file = 'z_scored_180901-181201.csv'
    predict_test = 'cluster_data\\181201-190201\\'
    predict_test_file = 'z_scored_181201-190201.csv'
    with open(cluster_dir + cluster_file, 'r') as f:
        lines = csv.reader(f)
        for line in lines:
            c_Data.append([float(t) for t in line])
    all_Data = []  # 这是预测的
   #with open(predict_dir + predict_file, 'r') as f:
    with open(predict_test+predict_test_file,'r') as f:
        lines = csv.reader(f)
        for stock in lines:
            all_Data.append([float(t) for t in stock])
    dis_1= []
    dis_2 = []
    min_1 = 1e20
    max_1 = 0
    for i in tqdm(range(len(all_Data))):
        for j in range(i+1,len(all_Data)):
            #dis_1.append(cDTW(c_Data[i],c_Data[j]))
            #dis_2.append(cDTW(all_Data[i],all_Data[j]))
            dis_1.append(cDTW(c_Data[i],c_Data[j]))
            dis_2.append(cDTW(all_Data[i],all_Data[j]))
            if(dis_1[-1]<min_1):
                min_1 = dis_1[-1]
            if(dis_1[-1]>max_1):
                max_1= dis_1[-1]

    #plt.scatter(dis_1,dis_2,s=5)
    #plt.show()
    dis_11 = []
    dis_21 = []
    #总的距离分布直方图，在未来区间上
    #plt.hist(dis_2,bins=80)
    #plt.show()
    #print("total mean: ",np.mean(dis_2))
    delta = (max_1-min_1)/8
    iter = min_1
    while(iter<max_1):
        dis_21 = []
        for i,t in enumerate(dis_1):
            if t<=iter+delta and t>iter:
                dis_21.append(dis_2[i])
        plt.hist(dis_21,bins=80)
        plt.show()
        print(iter,"-",iter+delta," mean: ",np.mean(dis_21))
        iter += delta

#以下用来计算DBI

def AvgDis(Data):
    data_len = len(Data)
    ret = 0
    for i,s1 in enumerate(Data):
        for j,s2 in enumerate(Data[i+1:]):
           ret += cDTW(s1,s2)
    ret = ret*2 / (data_len * (data_len - 1))
    return ret

def compute_DBI():
    if_predict = 1
    c_Data = []
    cluster_dir = 'cluster_ans\\for_train\\' + train_begin_date+'-'+train_end_date + '\\'
    cluster_file = 'z_scored_' + train_begin_date + '-' + train_end_date+'.csv'
    with open(cluster_dir + cluster_file, 'r') as f:
        lines = csv.reader(f)
        for line in lines:
            c_Data.append([float(t) for t in line])

    #all_Data 是预测区间的
    all_Data = []
    with open(predict_dir + predict_file, 'r') as f:
        lines = csv.reader(f)
        for stock in lines:
            all_Data.append([float(t) for t in stock])

    cent = []
    with open(clustering_ans_dir + cent_file, 'r') as f:
        cents = csv.reader(f)
        for c in cents:
            cent.append([float(tt) for tt in c[1:]])

    mem = []
    with open(clustering_ans_dir + mem_file, 'r') as f:
        mems = csv.reader(f)
        for line in mems:
            mem.append(int(line[1]))

    global num_cluster
    DBI = 0
    for i in tqdm(range(num_cluster)):
        this_cluster = []
        for stock,stock_type in enumerate(mem):
            if stock_type == i:
                this_cluster.append(all_Data[stock])   #c_data or all_data
        avg_i = AvgDis(this_cluster)
        cent_pi = 1
        if if_predict:
            cent_pi,_ = DBA_iteration(this_cluster[0], np.array(this_cluster))
        else:
            cent_pi = cent[i]
        #print(i,": ",avg_i)
        tmp_max = -1
        for j in range(num_cluster):
            if j == i:
                continue
            this_j_cluster = []
            for stock, stock_type in enumerate(mem):
                if stock_type == j:
                    this_j_cluster.append(all_Data[stock])
            avg_j = AvgDis(this_j_cluster)
            d_cent = cDTW(cent[i],cent[j])
            cent_pj = 1
            if if_predict:
                cent_pj,_ = DBA_iteration(this_j_cluster[0],np.array(this_j_cluster))
            else:
                cent_pj = cent[j]
            d_cent = cDTW(cent_pi,cent_pj)

            if tmp_max < (avg_i+avg_j) / d_cent:
                tmp_max = (avg_i+avg_j) / d_cent
        DBI += tmp_max
    DBI = DBI / num_cluster
    return DBI

def compute_DBI_obv():
    '''
    for clustering ans which has been refined
    :return: DBI
    '''
    mem_file_1 = 'cluster_ans/for_train/' + train_begin_date + '-'+ train_end_date + '/' + 'obvious_mem.csv'
    data_file_1 = 'cluster_ans/for_train/' + train_begin_date + '-'+ train_end_date + '/' + 'obvious_stock.csv'
    data_file_predict = 'cluster_data/' + predict_begin_date + '-' + predict_end_date + '/' \
    + 'z_scored_' +  predict_begin_date + '-' + predict_end_date+'.csv'

    all_Data = []
    with open(data_file_1, 'r') as f:
        lines = csv.reader(f)
        for stock in lines:
            all_Data.append([float(t) for t in stock])
    mem = []
    temp_dic = {}
    with open(mem_file_1, 'r') as f:
        mems = csv.reader(f)
        for line in mems:
            mem.append(int(line[1]))
            temp_dic[mem_index_dict[line[0]]] = 1
    pre_Data = []
    with open(data_file_predict,'r') as f:
        pres = csv.reader(f)
        for ii,line in enumerate(pres):
            try:
                a = temp_dic[ii]
                pre_Data.append([float(t) for t in line])
            except:
                continue
    assert(len(pre_Data)==len(all_Data))
    global num_cluster
    DBI = 0
    for i in tqdm(range(num_cluster)):
        this_cluster = []
        for stock,stock_type in enumerate(mem):
            if stock_type == i:
                this_cluster.append(pre_Data[stock])   #c_data or all_data
        avg_i = AvgDis(this_cluster)

        cent_pi,_ = DBA_iteration(this_cluster[0], np.array(this_cluster))

        #print(i,": ",avg_i)
        tmp_max = -1
        for j in range(num_cluster):
            if j == i:
                continue
            this_j_cluster = []
            for stock, stock_type in enumerate(mem):
                if stock_type == j:
                    this_j_cluster.append(pre_Data[stock])
            avg_j = AvgDis(this_j_cluster)
            cent_pj,_ = DBA_iteration(this_j_cluster[0],np.array(this_j_cluster))
            d_cent = cDTW(cent_pi,cent_pj)

            if tmp_max < (avg_i+avg_j) / d_cent:
                tmp_max = (avg_i+avg_j) / d_cent
        DBI += tmp_max
    DBI = DBI / num_cluster
    return DBI

def extract_shape_simmilar(obv_param,tol_param):
    '''

    :param obv_param: 涨和跌的差异显著性参数，0-100
    :param tol_param: 与主趋势差异容忍参数 0-1
    :return:
    '''
    Data = []
    stock_idx = []
    c_dir = 'cluster_ans/for_train/{begin}-{end}/z_scored_{begin}-{end}.csv'.format(begin=train_begin_date,end=train_end_date)
    idx_dir = 'cluster_ans/for_train/{begin}-{end}/stock_idx_{begin}-{end}.txt'.format(begin=train_begin_date,end=train_end_date)
    obvious_ans_dir = 'cluster_ans/for_train/{begin}-{end}/'.format(begin=train_begin_date,end=train_end_date)
    with open(idx_dir,'r') as f:
        for line in f.readlines():
            stock_idx.append(line.replace('\n','').replace('s',''))

    with open(c_dir,'r') as f:
        lines = csv.reader(f)
        for line in lines:
            Data.append([float(t) for t in line])
    assert len(Data) == len(mem_list)
    assert len(stock_idx) == len(Data)

    time_interval = '190101-190601'
    data_ud_dir = 'cluster_ans/for_train/' + time_interval + '/up_and_down_' + time_interval + '.csv'
    Data_ud = []
    with open(data_ud_dir, 'r') as f:
        Data_ud = list(csv.reader(f))
    Data_ud = np.array(Data_ud, dtype=float)
    Data_ud = np.array(Data_ud,dtype=int)
    remain_num_stock = 0

    for i in range(num_cluster):
        print("cluster: ",i)
        data_i = []
        data_ud_i = []
        idx_i = []
        for j in range(len(mem_list)):
            if mem_list[j] == i:
                data_i.append(Data[j])
                data_ud_i.append(Data_ud[j])
                idx_i.append(stock_idx[j])
        count_sum = np.zeros((len(Data_ud[0]),2))   # 1涨 0 跌
        for t in data_ud_i:
            for day,tt in enumerate(t):
                if tt == 1:
                    count_sum[day][1] += 1
                else:
                    count_sum[day][0] += 1
        '''
        for j in range(count_sum.shape[0]):
            total_count = count_sum[j][0]+count_sum[j][1]
            assert total_count == len(data_ud_i)
            print(j)
            print("up: {UP}%".format(UP=100*count_sum[j][1]/total_count) )
            print("down: {DOWN}%".format(DOWN=100*count_sum[j][0]/total_count))
        '''
        majority_direction = []
        count_remarkable = 0
        for j in range(count_sum.shape[0]):
            total_count = count_sum[j][0] + count_sum[j][1]
            assert total_count == len(data_ud_i)
            up = 100*count_sum[j][1]/total_count
            down = 100*count_sum[j][0]/total_count
            if math.fabs(up - down) > obv_param: # 1 up -1 down 0 no main direction
                majority_direction.append(1 if up > down else -1)
                count_remarkable += 1
            else:
                majority_direction.append(0)
        print("以下比例的天数，趋势保持一致： ",count_remarkable / count_sum.shape[0])  #多少比例的天数，趋势是显著趋向于一边的

        #将那些完全符合主趋势or差异天数在一个阈值内的stock从这个堆儿里边挑出来
        obvious_stock = []
        obvious_stock_no = []
        for seq,stock_ud in enumerate(data_ud_i):
            diff_day = 0
            for day in range(count_sum.shape[0]):
                if majority_direction[day] != 0:
                    if stock_ud[day] != majority_direction[day]:
                        diff_day += 1
            if diff_day / count_sum.shape[0] <= tol_param:
                remain_num_stock += 1
                obvious_stock.append(data_i[seq])
                obvious_stock_no.append(idx_i[seq])
        with open(obvious_ans_dir+'obvious_ans'+'.txt','a+',newline='') as f:
            for sno in obvious_stock_no:
                f.write('s'+str(sno)+'\n')
        with open(obvious_ans_dir + 'obvious_stock'+'.csv','a+',newline='') as f:
            w = csv.writer(f)
            w.writerows(obvious_stock)
        with open(obvious_ans_dir + 'obvious_mem'+'.csv','a+',newline='') as f:
            w  =csv.writer(f)
            for tt in obvious_stock_no:
                w.writerow(['s'+str(tt),i])
        count_sim = len(obvious_stock)
        print("有{}%的stock是显著与主趋势一致的".format(100*count_sim / len(data_ud_i)))

        #paint
        plt.figure()
        idx = [s for s in range(1,len(Data[0])+1)]
        for stock in data_i:
            plt.plot(idx,stock)
        plt.savefig(obvious_ans_dir+'init_ans_'+str(i)+'.png')
        plt.show()
        #os.system("pause")
        plt.figure()

        for stock in obvious_stock:
            plt.plot(idx,stock)
        if(len(obvious_stock)>0):
            plt.savefig(obvious_ans_dir+'refine_ans_'+str(i)+'.png')
        plt.show()
        #os.system("pause")
    print("保留下了：",100*remain_num_stock / len(Data),"%的stocks")

def show_stocks(Data):
    idx = [s for s in range(1, len(Data[0]) + 1)]
    plt.figure()
    for stock in Data:
        plt.plot(idx,stock)
    plt.show()

def run():
    global mem_file,cent_file,mem_dict,\
        predict_dir,predict_file,predict_idx_file,up_and_down_file,clustering_ans_dir
    clustering_ans_dir = clustering_ans_dir.format(begin=train_begin_date,end=train_end_date)
    mem_file = mem_file.format(begin=train_begin_date,end=train_end_date)
    cent_file = cent_file.format(begin=train_begin_date,end=train_end_date)
    up_and_down_file = up_and_down_file.format(begin=train_begin_date,end=train_end_date)
    predict_dir = predict_dir.format(begin=predict_begin_date,end=predict_end_date)
    predict_file = predict_file.format(begin=predict_begin_date,end=predict_end_date)
    predict_idx_file = predict_idx_file.format(begin=predict_begin_date,end=predict_end_date)

    #拿到聚类结果
    global num_cluster
    cluster_tag = {}
    with open(clustering_ans_dir+mem_file,'r') as f:
        lines = csv.reader(f)
        for ii,line in enumerate(lines):
            mem_dict[line[0][1:]] = int(line[1])
            mem_list.append(int(line[1]))
            mem_index_dict[line[0]] = ii
            try:
                a = cluster_tag[line[1]]
            except:
                cluster_tag[line[1]] = 1
                num_cluster += 1
    #print(compute_DBI_obv())
    #refine_ans()
    #numerical_compare()
    #draw_pic_on_predictData()
    #print(compute_DBI())
    #distance_compare()
    extract_shape_simmilar(30,0.08)
    '''
    dd = []
    with open('cluster_data/190601-190720/z_scored_190601-190720.csv','r') as f:
        stocks = csv.reader(f)
        for stock in stocks:
            dd.append([float(t) for t in stock])
    show_stocks(dd)
    '''

if __name__ == '__main__':
    run()