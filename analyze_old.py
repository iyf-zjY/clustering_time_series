import numpy as np
import os
import csv

#暂时这个代码应该没用

data_dir = 'cluster_ans/train_ans/'
former = '170816-170915'
later = '171225-180126'
def pick_stock(cent_f,cent_l):
    stock_list_f = []
    stock_list_l = []
    with open(data_dir+'mem_'+former+'_wm.csv','r') as f:
        tmp = list(csv.reader(f))
        for s in tmp:
            if int(s[1]) == cent_f:
                stock_list_f.append(s[0])

    with open(data_dir+'mem_'+later+'_wm.csv','r') as f:
        tmp = list(csv.reader(f))
        for s in tmp:
            if int(s[1]) == cent_l:
                stock_list_l.append(s[0])
    return stock_list_f,stock_list_l

if __name__ == '__main__':
    cent_former_f = data_dir+'cent_'+former+'_wm.csv'
    cent_later_f = data_dir + 'cent_'+later+'_wm.csv'
    cent_former = []
    cent_later = []
    with open(cent_former_f,'r') as f:
        cent_former = list(csv.reader(f))
    with open(cent_later_f,'r') as f:
        cent_later = list(csv.reader(f))
    cent_former = [t[1:]+[0] for t in cent_former]
    cent_later = [t[1:] for t in cent_later]
    min_dis = 1e20
    min_f = 0
    min_l = 0
    for i,tmp1 in enumerate(cent_former):
        for j,tmp2 in enumerate(cent_later):
            dis = np.linalg.norm([float(tmp1[i])-float(tmp2[i]) for i in range(len(tmp1))])
            if  dis < min_dis:
                min_dis = dis
                min_f = i
                min_l = j

    s_f,s_l = pick_stock(min_f,min_l)
    with open(data_dir+'f_list_'+former+'.csv','w',newline='') as f:
        writer = csv.writer(f)
        for t in s_f:
            writer.writerow([t])
    with open(data_dir+'l_list_'+later+'.csv','w',newline='') as f:
        writer = csv.writer(f)
        for t in s_l:
            writer.writerow([t])

