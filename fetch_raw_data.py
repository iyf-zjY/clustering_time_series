#encoding=utf-8
import pymysql
import numpy as np
import math
import os
import csv
import pandas as pd
import datetime
import copy

DB_conf = {
    'host': '10.10.8.11',
    'user': 'root',
    'password': 'xiang2qiong',
    'database': 'stock_after_price2',
    'port': 9806,
    'charset': 'utf8mb4'
}
DB_conf_test = {
    'host':'localhost',
    'user':'root',
    'password':'123456',
    'database':'lab_data',
    'port':3306,
    'charset':'utf8mb4'
}
global db_after_price
attr_name = ['date','open',
             'close','high','low','volume']
data_dir = 'cluster_data/'
all_data = []
stock_prefix = 'afterprice_'
csv_header = ['stock_no']

class DB:
    def __init__(self,DB_conf):
        self.database = pymysql.connect(**DB_conf)
        self.cursor = self.database.cursor()
    def get_all_table_names(self):
        sql = 'show tables'
        self.cursor.execute(sql)
        ret = self.cursor.fetchall()
        return ret

#这个函数fetch了指定区间的时间序列，并且对丢失数据进行了补全（对齐前一天的）
def get_all_raw_data(tables,begin_date,end_date):
    sql = 'select * from {} where DATE(date) BETWEEN "{}" and "{}"'
    tmp_stock_data = []
    table_name_index = []
    max_len = 0
    max_index = -1
    for i,table in enumerate(tables):
        print(table)
        try:
            db_after_price.cursor.execute(sql.format(stock_prefix+table,begin_date,end_date))
            tmp_df = list(db_after_price.cursor.fetchall())
            #把str转换成date
            for tk,tt in enumerate(tmp_df):
                tt = list(tt)
                tt[0] = str(tt[0])
                tt[0] = datetime.date(int(tt[0][:4]), int(tt[0][5:7]), int(tt[0][8:]))
                tmp_df[tk] = tt
            #if len(list(tmp_df)) != expected_count:
            #    print('data_lost')
            #tmp_df = pd.DataFrame(list(tmp_df),columns=attr_name)
            tmp_stock_data.append(tmp_df)
            table_name_index.append(table)
            if max_len < len(tmp_df):
                max_len = len(tmp_df)
                max_index = i
        except:
            continue
        '''
        for i,tmp_name in enumerate(attr_name):
            if i == 0:
                continue
            #记得读的时候把s replace掉
            stockNo = ['s'+str(table.replace(stock_prefix,''))]
            all_data[i-1].append(stockNo+tmp_df[tmp_name].tolist())
        '''
    date_index = []
    #find all date that should in the time series
    for tt in tmp_stock_data[max_index]:
        date_index.append(tt[0])
        csv_header.append(str(tt[0]))

    #pad missing date data
    d_t = datetime.timedelta(1)
    for i,stock_ in enumerate(tmp_stock_data):
        if len(stock_) == 0:
            continue
        else:
            stock_t = pd.DataFrame(stock_, columns=attr_name)
            if len(stock_) < max_len:
                this_date_index = [tt[0] for tt in stock_]
                for standard_date in date_index:
                    if standard_date not in this_date_index:
                        #前一天有
                        if standard_date - d_t in this_date_index:
                            standard_date_data = copy.deepcopy(stock_t[stock_t['date']==standard_date - d_t])
                            standard_date_data.iloc[0:1]['date'] = standard_date
                            stock_t = stock_t.append(standard_date_data,ignore_index=True)
                        #前一天没有
                        else:
                            Find = 0
                            search_date = copy.deepcopy(standard_date - d_t)
                            #大于最开始的时间段，向前找
                            while(search_date >= date_index[0]):
                                if search_date in this_date_index:
                                    Find = 1
                                    break
                                else:
                                    search_date -= d_t
                            if not Find:
                                standard_date_data = pd.Series({
                                    'date':standard_date,
                                    'open':0,'close':0, 'high':0, 'low':0, 'volume':0
                                })
                                stock_t = stock_t.append(standard_date_data,ignore_index=True)
                            else:
                                standard_date_data = copy.deepcopy(stock_t[stock_t['date'] == search_date])
                                standard_date_data.iloc[0:1]['date'] = standard_date
                                stock_t = stock_t.append(standard_date_data,ignore_index=True)
                stock_t = stock_t.sort_values(by='date')
            for t, tmp_name in enumerate(attr_name):
                if t == 0:
                    continue
                # s为了不让csv把股票代码自作聪明省略0
                stockNo = ['s' + str(table_name_index[i])]
                all_data[t - 1].append(stockNo + stock_t[tmp_name].tolist())


def write_to_csv():
    this_dir = os.path.join(os.path.curdir,data_dir)
    if not os.path.exists(this_dir):
        os.mkdir(this_dir)
    for i,tmp_name in enumerate(attr_name):
        if i==0:
            continue
        with open(data_dir+tmp_name+'.csv','w',newline='') as f:
            writer = csv.writer(f)
            writer.writerow(csv_header)
            #writer.writerow(csv_header)
            writer.writerows(all_data[i-1])

if __name__ == '__main__':
    #the two intervals under this : only for test
    #17-08-16 17-09-16
    #17-12-25 18-01-26
    begin_date = '2019-04-13'
    end_date = '2019-05-10'

    db_after_price = DB(DB_conf_test)
    #tables = db_after_price.get_all_table_names()
    tables = []
    with open(data_dir+'stock_list/'+'local_test.txt','r') as f:
        for tmp in f.readlines():
            tables.append(tmp.replace('\n',''))
    #不算日期维度
    for i in range(len(attr_name)-1):
        all_data.append([])
    get_all_raw_data(tables,begin_date,end_date)
    #print(all_data)
    write_to_csv()
