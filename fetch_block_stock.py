# -- coding: utf-8 --
import csv
import os
import pandas as pd
import requests
from bs4 import  BeautifulSoup

def get_stock():
    with open('fdc.csv','r') as f:
        r = list(csv.reader(f))
        df = pd.DataFrame(r)
        stock_list = df.iloc[:,0].tolist()
        stock_list = [r.zfill(6) for r in stock_list]
        return stock_list
if __name__ == '__main__':
    '''
   stock_list = get_stock()
   with open('fdc_stock.txt','w') as f:
       for s in stock_list:
           f.write(s+'\n')
    '''
    stock_list = []
    with open('fdc_stock.txt','r') as f:
        for tmp in f.readlines():
            stock_list.append(tmp.replace('\n',''))
    print(stock_list)
