from sklearn.linear_model import LassoCV
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data.dataset import Dataset,Subset
from random import choice
import pickle
import bz2
import dill
import pandas as pd
import statsmodels.formula.api as smf
model_dir = r'D:\\data\\pd_frame\\'

industry_factors = ['AERODEF', 'AIRLINES', 'ALUMSTEL', 'APPAREL', 'AUTO',
       'BANKS','BEVTOB', 'BIOLIFE', 'BLDGPROD','CHEM', 'CNSTENG', 'CNSTMACH', 'CNSTMATL', 'COMMEQP', 'COMPELEC',
       'COMSVCS', 'CONGLOM', 'CONTAINR', 'DISTRIB',
       'DIVFIN', 'DIVYILD', 'ELECEQP', 'ELECUTIL', 'FOODPROD', 'FOODRET', 'GASUTIL',
       'HLTHEQP', 'HLTHSVCS', 'HOMEBLDG', 'HOUSEDUR','INDMACH', 'INSURNCE', 'INTERNET',
        'LEISPROD', 'LEISSVCS', 'LIFEINS', 'MEDIA', 'MGDHLTH','MULTUTIL',
       'OILGSCON', 'OILGSDRL', 'OILGSEQP', 'OILGSEXP', 'PAPER', 'PHARMA',
       'PRECMTLS','PSNLPROD','REALEST',
       'RESTAUR', 'ROADRAIL','SEMICOND', 'SEMIEQP','SOFTWARE', 'SPLTYRET', 'SPTYCHEM', 'SPTYSTOR',
       'TELECOM', 'TRADECO', 'TRANSPRT', 'WIRELESS']

style_factors = ['BETA','SIZE','MOMENTUM','VALUE']

def sort_cols(test):
    return (test.reindex(sorted(test.columns), axis=1))

class factor_set(Dataset):
    def __init__(self,year = 2003,random = False,channel = 1,period = 3):
        if random:
            year = choice(list(range(2003,2009)))
        super().__init__()
        self.years = [year+i for i in range(period)]
        self.path_str = model_dir
        self.frames = {}
        for year in self.years:
            fil = model_dir + "pandas-frames." + str(year) + ".pickle.bz2"
            self.frames.update(pickle.load(bz2.open(fil, "rb")))
        self.label_list = []
        self.frames_list = []
        self.id_amt_int = 0
        #index intersection
        try:
            id = pd.read_csv('id.csv',index_col = 0)['id'].tolist()
        except:
            id = set(self.frames.popitem()[1]['ID'].to_list())
            for x in self.frames:
                frame = self.frames[x]
                frame = frame.where(frame['IssuerMarketCap'] > 1e9).dropna(axis=0,how = 'all')
                # get liquad universe
                id = id.intersection(set(frame['ID'].tolist()))
            id = list(id)
            pd.DataFrame(id,columns=['id']).to_csv('id.csv')

        self.id_amt_int = len(id)
        for x in self.frames:
            frame = self.frames[x].set_index('ID')
            self.label_list.append(frame.loc[id, ['Ret']])
            frame = sort_cols(frame.loc[id,['STREVRSL', 'LTREVRSL', 'EARNQLTY','EARNYILD', 'MGMTQLTY', 'PROFIT', 'SEASON', 'SENTMT']]).bfill().ffill()
            self.frames_list.append(frame)
            del frame

    def __getitem__(self, index):
        return self.label_list[index], self.frames_list[index]

    def __len__(self):
        return len(self.label_list)

factor_lasso = factor_set(2007,period=3)
with open(r'D:\data\pd_frame\factors_set_lasso_2007-2009.pkl', 'wb') as f:
    dill.dump(factor_lasso,f)
with open(r'D:\data\pd_frame\factors_set_lasso_2007-2009.pkl', 'rb') as f:
    factor = dill.load(f)
X = pd.concat(factor.frames_list,axis = 0)
y = pd.concat(factor.label_list,axis = 0)
reg = LassoCV(cv = 10,random_state=0).fit(X.values,y.values.reshape(-1))
with open(r'C:\Users\zhuyu\PycharmProjects\ml4fin\LassoCV10.pt','wb') as f:
    dill.dump(reg,f)
train_pred = reg.predict(X)
train_pred_df = pd.DataFrame(train_pred,index = X.index,columns=["LassoFactor"])
date = []
for i in factor.frames.keys():
    date += [pd.to_datetime(i,format = '%Y%m%d')]*1478
train_pred_df.reset_index(inplace=True)
train_pred_df.index = date
train_pred_df.to_csv('Lasso_factor_train.csv')

with open(r'D:\data\pd_frame\factors_set_lasso_2010.pkl', 'rb') as f:
    factor = dill.load(f)
X_test = pd.concat(factor.frames_list,axis = 0)
test_pred = reg.predict(X_test.values)

test_pred_df = pd.DataFrame(test_pred,index = X_test.index,columns=["LassoFactor"])
date = []
for i in factor.frames.keys():
    date += [pd.to_datetime(i,format = '%Y%m%d')]*1478
test_pred_df.reset_index(inplace=True)
test_pred_df.index = date
test_pred_df.to_csv('Lasso_factor_test.csv')
pass

