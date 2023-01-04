import pickle
import bz2
import dill
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
model_dir = r'D:\\data\\pd_frame\\'

#get next trade day return
#ret_df = pd.DataFrame()
'''ret_list =[]
date = []
for year in range(2007,2011):
    frames = {}
    fil = model_dir + "pandas-frames." + str(year) + ".pickle.bz2"
    frames.update(pickle.load(bz2.open(fil, "rb")))

    for key,val in frames.items():
        sub = val.loc[:,['ID','Ret']]
        sub_pivot = sub.pivot_table(values='Ret',columns= 'ID')
        sub_pivot.index = [key]
        ret_list.append(sub_pivot)
        date.append(pd.to_datetime(key))
        
with open(r'D:\data\pd_frame\ret_2007-2010.pkl','wb') as f:
    dill.dump([date,ret_list],f)
        
'''
with open(r'D:\data\pd_frame\ret_2007-2010.pkl','rb') as f:
    [date_list,ret_list] = dill.load(f)
id = pd.read_csv('id.csv',index_col = 0)['id'].tolist()
val_list = []
for val in ret_list:
    val = val.reindex(columns = id)
    val_list.append(val)
ret_df = pd.concat(val_list,axis = 0)
ret_sub_df =ret_df.iloc[:-1,:]
ret_sub_df.index = pd.to_datetime(ret_df.index[1:].tolist())

#load_weight
weight_df =pd.read_csv('weight_20070109.csv')
weight_sub_df = weight_df.loc[:,['ID','weight']]
ret_sub_df = ret_sub_df.reindex(columns = weight_sub_df['ID'].tolist())
ret_array = ret_sub_df.values
weight_array = weight_sub_df.loc[:,'weight'].values
pos_weight = np.sum(weight_array[weight_array>0])
neg_weight = np.sum(weight_array[weight_array<0])
weight_array[weight_array>0] = weight_array[weight_array>0]/pos_weight
weight_array[weight_array<0] = weight_array[weight_array<0]/neg_weight
pfl_ret_array = np.matmul(ret_array,weight_array)
pfl_cul_ret_array = np.multiply.accumulate(1+pfl_ret_array)-1
plt.plot(pfl_cul_ret_array)
plt.show()
pass

