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
def sort_cols(test):
    return (test.reindex(sorted(test.columns), axis=1))

model_dir = r'D:\\data\\pd_frame\\'


# factor names
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

def get_formula(alpha):
    L = ["0", alpha]
    L.extend(style_factors)
    L.extend(industry_factors)
    return "Ret ~ " + " + ".join(L)


def wins(x, a, b):
    return (np.where(x <= a, a, np.where(x >= b, b, x)))


def clean_nas(df):
    numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()

    for numeric_column in numeric_columns:
        df[numeric_column] = np.nan_to_num(df[numeric_column])

    return df
def standlizard(factor_df):
    factor_df.dropna(axis=0, how='all', inplace=True)
    factor_name_str = factor_df.columns.tolist()[-1]
    factor_pivot_df = factor_df.pivot_table(index = 'date',columns = 'ID',values = factor_name_str)
    factor_demean_pivot_df = factor_pivot_df.sub(factor_pivot_df.mean(axis = 1),axis =0)
    factor_demean_pivot_df = factor_demean_pivot_df.divide(factor_pivot_df.std(axis=1),axis =0)
    factor_stack_df = factor_demean_pivot_df.stack().reset_index()
    factor_stack_df.columns = factor_df.columns.tolist()
    factor_stack_df['date'] = pd.to_datetime(factor_stack_df['date'])
    return factor_stack_df

def estimate_factor_return(factor_name_list:str,factor_df:pd.DataFrame,output = 'ret'):
    factor_df = clean_nas(factor_df)
    liquid_universe_df = factor_df.where(factor_df['IssuerMarketCap']>1e9).dropna(axis=0)
    res_list = []

    # winsoried for ret
    ret_series = liquid_universe_df.loc[:, 'Ret']
    up_q_float = ret_series.quantile(0.95)
    down_q_float = ret_series.quantile(0.05)
    liquid_universe_df.loc[:, 'Ret'] = wins(ret_series, down_q_float, up_q_float).tolist()
    for factor_name in factor_name_list:
        #winsorized
            #factor
        factor_series = liquid_universe_df.loc[:,factor_name]
        up_q_float = factor_series.quantile(0.95)
        down_q_float = factor_series.quantile(0.05)
        liquid_universe_df.loc[:, factor_name] = wins(factor_series,down_q_float,up_q_float).tolist()

        #get reg formula
        formula_str = get_formula(factor_name)
        #OLS regression
        model = smf.ols(formula = formula_str,data = liquid_universe_df)
        res = model.fit()
        res_list.append(res)
    if output=='ret':
        ret_list = [res_list[i].params[factor_name_list[i]] for i in range(len(res_list))]
        return ret_list
    return res_list

def factor_ret(factor_name_list:list,frames:dict,new_factor_df):
    ret_list = []
    date_list = []
    id = list(set(new_factor_df['ID'].tolist()))
    idx = 0
    for key,factor_df in frames.items():
        idx += 1
        if idx<=4:
            continue
        factor_df.dropna(axis = 0,how = 'all',inplace=True)
        factor_df.set_index('ID',inplace=True)
        factor_df = factor_df.reindex(id)
        date = pd.to_datetime(key)
        new_factor_sub_df = new_factor_df.where(new_factor_df['date'] == date).dropna(axis=0)
        new_factor_sub_df.set_index('ID',inplace=True)
        new_factor_sub_df = new_factor_sub_df.loc[:,factor_name_list]
        if new_factor_sub_df.shape[0]==0:
            continue
        factor_df = pd.merge(factor_df,new_factor_sub_df,left_index=True,right_index=True)
        factor_df.dropna(axis =0)
        res = estimate_factor_return(factor_name_list,factor_df,output='ret')
        ret_list.append(res)
        date_list.append(key)

    ret_df = pd.DataFrame(index=date_list,data=ret_list,columns = factor_name_list)
    return ret_df


def cum_ror(factor_df,factor_name_str,beg_y_int,end_y_int,title_str):
    factor_ret_df = pd.DataFrame()
    factor_df['date'] = pd.to_datetime(factor_df['date'])
    factor_list = [factor_name_str]
    for year in list(range(beg_y_int,end_y_int+1)):
        frames = {}
        fil = model_dir + "pandas-frames." + str(year) + ".pickle.bz2"
        frames.update(pickle.load(bz2.open(fil, "rb")))

        for x in frames:
            frames[x] = sort_cols(frames[x])
        ret_df = factor_ret(factor_list,frames,factor_df)
        if factor_ret_df.empty:
            factor_ret_df = ret_df
        else:
            factor_ret_df = pd.concat([factor_ret_df,ret_df],axis=0)
        del frames

    factor_ret_df.index = pd.to_datetime(factor_ret_df.index,format = '%Y%m%d')
    factor_ret_df.cumsum().plot(figsize=(10,5))
    plt.legend()
    plt.title(title_str)
    plt.show()


'''factor_train_df = pd.read_csv('XGBoostFactor_train.csv')
factor_train_df = standlizard(factor_train_df)
cum_ror(factor_train_df,'XGBoost_Factor',2009,2009,'XGboost Factor culmulative return in train set')
factor_test_df = pd.read_csv('XGBoostFactor_test.csv')
factor_test_df = standlizard(factor_test_df)
cum_ror(factor_test_df,'XGBoost_Factor',2010,2010,'XGboost Factor culmulative return in test set')'''

#Load return
with open(r'D:\data\pd_frame\ret_2007-2010.pkl','rb') as f:
    [date_list,ret_list] = dill.load(f)
date_list = pd.to_datetime(date_list)

weight_df = pd.read_csv(r'C:\Users\zhuyu\PycharmProjects\ml4fin\weights\Lasso_factor_test_weight.csv', index_col=0)


def bkt(weight_df, holding_period,title_str):
    weight_pivot_df = weight_df.pivot(index='date', columns='ID', values='last_weight')
    initial_dt = pd.to_datetime(weight_pivot_df.index[0])

    weight_pivot_df.fillna(0, inplace=True)
    weight_pivot_df.index = pd.to_datetime(weight_pivot_df.index)
    sub_idx_list = list(range(0, weight_pivot_df.shape[0], holding_period))
    weight_pivot_sub_df = weight_pivot_df.iloc[sub_idx_list, :]
    id = weight_pivot_df.columns.tolist()
    ret_sub_list = []
    dt_loc_int = np.where(np.array(date_list) == initial_dt)[0][0]
    for ret in ret_list[dt_loc_int:]:
        ret_sub_list.append(ret.reindex(columns=id))
    ret_df = pd.concat(ret_sub_list, axis=0)
    ret_df.index = pd.to_datetime(date_list[dt_loc_int:])
    weight_pivot_sub_df = weight_pivot_sub_df.reindex(ret_df.index)
    weight_pivot_sub_df = weight_pivot_sub_df.ffill().bfill()

    ret_df.fillna(0, inplace=True)
    pfl_ret_mx = np.matmul(ret_df.values, weight_pivot_sub_df.values.T)
    pfl_ret_array = np.diag(pfl_ret_mx)
    pfl_cumsum_array = np.cumsum(pfl_ret_array)
    plt.plot(pfl_cumsum_array)
    plt.title(title_str)
    plt.show()
    return pd.DataFrame(data=pfl_ret_array,index = ret_df.index,columns = ['daily_ror'])
file_str = 'CNNFactor_train_top.csv'
a = pd.read_csv(r'C:\Users\zhuyu\PycharmProjects\ml4fin\weights\\'+file_str,index_col=0)
a['date'] = pd.to_datetime(a['date'],format='%Y/%m/%d')
a.to_csv(r'C:\Users\zhuyu\PycharmProjects\ml4fin\weights\\'+file_str)

a  = bkt(weight_df, 20,'cum ror')
pass