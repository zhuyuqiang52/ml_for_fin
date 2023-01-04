import pickle
import bz2
import dill
import pandas as pd
import numpy as np
dates_df = pd.read_csv('dates.csv')
dates = dates_df.iloc[:,1].tolist()
id = pd.read_csv('id.csv',index_col = 0)['id'].tolist()
with open(r'D:\data\pd_frame\factors_val_2007_pred.pkl', 'rb') as f:
    y2007 = dill.load(f)
    y2007 = y2007.detach().numpy()
with open(r'D:\data\pd_frame\factors_val_2008_pred.pkl', 'rb') as f:
    y2008 = dill.load(f)
    y2008 = y2008.detach().numpy()
with open(r'D:\data\pd_frame\factors_val_2009_pred.pkl', 'rb') as f:
    y2009 = dill.load(f)
    y2009 = y2009.detach().numpy()
factor = np.concatenate([y2007,y2008,y2009],axis=0)
factor_df = pd.DataFrame(data= factor,index = pd.to_datetime(dates,format = '%Y%m%d'),columns=id)
factor_stack_df = factor_df.stack()
factor_stack_df = factor_stack_df.reset_index()
factor_stack_df.columns = ['date','ID','CNNFactor']
factor_stack_df.set_index('date',inplace=True)
factor_stack_df.to_csv('CNNFactor_train.csv')

with open(r'D:\data\pd_frame\factors_set_2010.pkl', 'rb') as f:
    y2010 = dill.load(f)
dates_2010 = list(y2010.frames.keys())[4:]
dates_2010 = pd.to_datetime(dates_2010,format = '%Y%m%d')

with open(r'D:\data\pd_frame\factors_set_2010_pred.pkl', 'rb') as f:
    y2010_pred = dill.load(f)
y2010_pred = pd.DataFrame(y2010_pred,index=dates_2010,columns=id)
y2010_pred_stack = y2010_pred.stack().reset_index()
y2010_pred_stack.columns = ['date','ID','CNNFactor']
y2010_pred_stack.set_index('date',inplace=True)
y2010_pred_stack.to_csv('CNNFactor_test.csv')