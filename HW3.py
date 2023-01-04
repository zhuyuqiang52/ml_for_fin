import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import pickle
import bz2
import dill
# load data
model_dir = r'D:\\data\\pd_frame\\'


def sort_cols(test):
    return (test.reindex(sorted(test.columns), axis=1))




'''covariance = {}
for year in [2003, 2004, 2005, 2006]:
    fil = model_dir + "covariance." + str(year) + ".pickle.bz2"
    covariance.update(pickle.load(bz2.open(fil, "rb")))'''

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

#problem 1
def estimate_factor_return(factor_name_list:str,factor_df:pd.DataFrame,output = 'ret'):
    factor_df = clean_nas(factor_df)
    liquid_universe_df = factor_df.where(factor_df['IssuerMarketCap']>1e9).dropna(axis=0)
    res_list = []

    # winsoried for ret
    ret_series = liquid_universe_df.loc[:, 'Ret']
    up_q_float = ret_series.quantile(0.95)
    down_q_float = ret_series.quantile(0.05)
    liquid_universe_df.loc[:, 'Ret'] = wins(ret_series, down_q_float, up_q_float)
    for factor_name in factor_name_list:
        #winsorized
            #factor
        factor_series = liquid_universe_df.loc[:,factor_name]
        up_q_float = factor_series.quantile(0.95)
        down_q_float = factor_series.quantile(0.05)
        liquid_universe_df.loc[:, [factor_name]] = wins(factor_series,down_q_float,up_q_float)

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

#problem 2
def factor_ret(factor_name_list:list,frames:dict,new_factor_df):
    ret_list = []
    date_list = []
    id = list(set(new_factor_df['ID'].tolist()))
    idx = 0
    for key,factor_df in frames.items():
        idx += 1
        if idx<=4:
            continue
        factor_df.set_index('ID',inplace=True)
        factor_df = factor_df.reindex(id)
        date = pd.to_datetime(key)
        new_factor_sub_df = new_factor_df.where(new_factor_df['date'] == date).dropna(axis=0)
        new_factor_sub_df.set_index('ID',inplace=True)
        new_factor_sub_df = new_factor_sub_df.loc[:,['CNNFactor']]
        factor_df = pd.merge(factor_df,new_factor_sub_df,left_index=True,right_index=True)
        res = estimate_factor_return(factor_name_list,factor_df,output='ret')
        ret_list.append(res)
        date_list.append(key)

    ret_df = pd.DataFrame(index=date_list,data=ret_list,columns = factor_name_list)
    return ret_df

#problem 3

factor_list = ['CNNFactor']

factor_ret_df = pd.DataFrame()
pred = pd.read_csv('CNNFactor_train.csv')
pred['date'] = pd.to_datetime(pred['date'])

for year in list(range(2007,2010)):
    frames = {}
    fil = model_dir + "pandas-frames." + str(year) + ".pickle.bz2"
    frames.update(pickle.load(bz2.open(fil, "rb")))

    for x in frames:
        frames[x] = sort_cols(frames[x])

    ret_df = factor_ret(factor_list,frames,pred)
    if factor_ret_df.empty:
        factor_ret_df = ret_df
    else:
        factor_ret_df = pd.concat([factor_ret_df,ret_df],axis=0)
    del frames
factor_ret_df.to_csv('factor_ror2.csv')
factor_ret_df = pd.read_csv('factor_ror2.csv',index_col=0)
factor_ret_df.index = pd.to_datetime(factor_ret_df.index,format = '%Y%m%d')
factor_ret_df.cumsum().plot(figsize=(10,5))
plt.legend()
plt.show()
pass