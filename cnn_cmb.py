import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader,TensorDataset
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda
import torch.nn.functional as F
from torch.utils.data.dataset import Dataset,Subset
from random import choice
import pickle
import bz2
import dill
import pandas as pd

#file path
model_dir = r'D:\\data\\pd_frame\\'
#CNN
# one channel
batch_size = 64

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

def sort_cols(test):
    return (test.reindex(sorted(test.columns), axis=1))

def dataset_split(test_per_float,obj_dataset):
    size_int = len(obj_dataset)
    test_size_int = int(test_per_float*size_int)
    train_size_int = size_int-test_size_int
    train_dataset,test_dataset = torch.utils.data.random_split(obj_dataset,[train_size_int,test_size_int])
    return train_dataset,test_dataset

class inception(nn.Module):
    def __init__(self,in_channel_int):
        super(inception, self).__init__()
        self.avg_pool_1x1 = nn.Sequential(
            nn.AvgPool2d(3,padding=1,stride=1),
            nn.Conv2d(in_channel_int,24,kernel_size=(1,1))
        )
        self.conv_1x1 = nn.Sequential(
            nn.Conv2d(in_channel_int,16,kernel_size=(1,1))
        )
        self.conv_1x1_5x5 = nn.Sequential(
            nn.Conv2d(in_channel_int,16,kernel_size=(1,1)),
            nn.Conv2d(16,24,kernel_size=(5,5),padding=2)
        )
        self.conv_1x1_3x3_3x3 = nn.Sequential(
            nn.Conv2d(in_channel_int, 16, kernel_size=(1, 1)),
            nn.Conv2d(16, 24, kernel_size=(3, 3),padding=1),
            nn.Conv2d(24,24,kernel_size=(3,3),padding=1)
        )
    def forward(self, x):
        x_avg_pool_1x1 = self.avg_pool_1x1(x)
        x_conv_1x1 = self.conv_1x1(x)
        x_conv_1x1_5x5 = self.conv_1x1_5x5(x)
        x_conv_1x1_3x3_3x3 = self.conv_1x1_3x3_3x3(x)
        x = torch.cat([x_avg_pool_1x1,x_conv_1x1,x_conv_1x1_5x5,x_conv_1x1_3x3_3x3],dim=1)
        return x

class resid_net(nn.Module):
    def __init__(self,in_channel_int,out_channel_int):
        super(resid_net,self).__init__()
        self.conv_layer = nn.Sequential(
            nn.Conv2d(in_channel_int,out_channel_int,kernel_size=(3,3),padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channel_int, out_channel_int, kernel_size=(3, 3), padding=1),
            nn.ReLU()
        )
    def forward(self,x):
        conv = self.conv_layer(x)
        return F.relu(conv+x)

class CNN_module(nn.Module):
    def __init__(self,in_channel,out_size):
        super(CNN_module, self).__init__()
        self.sequential_cnn = nn.Sequential(
            nn.Conv2d(in_channel,64,kernel_size=(3,3)),
            nn.ReLU(),
            nn.MaxPool2d(2),
            inception(64),
            nn.Dropout(0.2),
            resid_net(88,88),
            nn.BatchNorm2d(88),
            nn.Conv2d(88,88,kernel_size=(3,3)),
            nn.ReLU(),
            nn.MaxPool2d(2),
            inception(88),
            nn.Dropout(0.2),
            resid_net(88,88),
            #nn.Conv2d(88,88,kernel_size=(3,3)),
            nn.Flatten(),
            nn.Linear(32384,out_size)
        )

    def forward(self,x):
        pred = self.sequential_cnn(x)
        return pred

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
            frame = sort_cols(frame.loc[id,['STREVRSL', 'LTREVRSL', 'EARNQLTY','EARNYILD', 'MGMTQLTY', 'PROFIT', 'SEASON', 'SENTMT']
                                                             +style_factors]).bfill().ffill()
            self.frames_list.append(torch.tensor(frame.values))
            del frame
        self.label_tensors_list = []
        self.stacked_list =[]
        #stack the data to get data with channel
        for x_i in range(channel-1,len(self.frames)):
            self.label_tensors_list.append(torch.LongTensor(self.label_list[x_i].values))
            self.stacked_list.append(torch.stack(self.frames_list[x_i+1-channel:x_i+1],dim = 0))

    def __getitem__(self, index):
        return self.label_tensors_list[index], self.stacked_list[index]

    def __len__(self):
        return len(self.label_tensors_list)



def train_loop(train_datloader,model,loss_func,optimizer):
    print('--------------Train Begin--------------')
    model.train()
    size = len(train_datloader.dataset)
    idx = 0
    for index_int,(y,X) in enumerate(train_datloader):
        print(f'batch {index_int}')
        optimizer.zero_grad()
        pred = model(X.float())
        loss = loss_func(pred,y.float().squeeze())
        loss.backward()
        optimizer.step()
        if index_int%1==0:
            loss_float,current = loss.item(),idx+len(X)
            print(f"loss: {loss_float:>7f}  [{current:>5d}/{size:>5d}]")
            idx = current
            with torch.no_grad():
                global train_loss_list
                train_loss_list.append(loss_float)

def test_loop(test_datloader,model,loss_func):
    print('--------------Test Begin--------------')
    model.eval()
    size = len(test_datloader.dataset)
    idx = 0
    loss_list = []
    for index_int,(y,X) in enumerate(test_datloader):
        with torch.no_grad():
            pred = model(X.float())
            loss = loss_func(pred,y.float().squeeze())
            if index_int%1==0:
                loss_float,current = loss.item(),idx+len(X)
                print(f"loss: {loss_float:>7f}  [{current:>5d}/{size:>5d}]")
                idx = current
                loss_list.append(loss_float)
    with torch.no_grad():
        global test_loss_list
        test_loss_list += loss_list
        print(f'Average Test Loss: {np.mean(loss_list)}')

def train_loop_kfold(train_datloader,val_loader,model,loss_func,optimizer):
    print('--------------Train Begin--------------')
    model.train()
    size = len(train_datloader.dataset)
    idx = 0
    for index_int,(y,X) in enumerate(train_datloader):
        print(f'batch {index_int}')
        optimizer.zero_grad()
        pred = model(X.float())
        loss = loss_func(pred,y.float().squeeze())
        loss.backward()
        optimizer.step()
        if index_int%1==0:
            loss_float,current = loss.item(),idx+len(X)
            print(f"loss: {loss_float:>7f}  [{current:>5d}/{size:>5d}]")
            idx = current
    print("Validation Begin:\n")
    model.eval()
    loss_list = []
    for index_int,(y,X) in enumerate(val_loader):
        with torch.no_grad():
            pred = model(X.float())
            loss = loss_func(pred,y.float().squeeze())
            loss_list.append(loss.item())
    with torch.no_grad():
        print(f'Validation Loss:{np.mean(loss_list)}')
        return np.mean(loss_list)

def ksplit(dataset,k):
    dataset_len_int = len(dataset)
    sub_len_int = int(dataset_len_int/(k-1))
    idx_list = list(range(dataset_len_int))
    Kflod_list = []
    for i in range(k):
        Kflod_list.append(idx_list[sub_len_int*i:sub_len_int*(i+1)])
    return Kflod_list

def train_proc(KFold):
    channel_int = 5
    out_size = factors.id_amt_int
    epochs = 50


    # loss func
    loss_func = nn.MSELoss()

    if KFold:
        #k-FOLD Validation part
        K = 5
        kFold_list = ksplit(train_dataset,K)
        # hyper-params
        lr_list = [1e-2, 1e-3, 1e-4]
        KFold_epoch_int = 1
        loss_list = []
        for k in range(K):
            idx_list = list(range(len(train_dataset)))
            train_sub_dataset = Subset(train_dataset,idx_list[idx_list not in kFold_list[k]]).dataset
            valid_dataset = Subset(train_dataset,kFold_list[k]).dataset
            train_loader = DataLoader(train_sub_dataset,batch_size=64)
            valid_loader =DataLoader(valid_dataset,batch_size=64)
            sub_loss_list =[]
            for lr in lr_list:
                model = CNN_module(channel_int, out_size)
                # optimizer
                optimizer = torch.optim.Adam(model.parameters(), lr=lr)
                loss = 0
                for e in range(KFold_epoch_int):
                    loss += train_loop_kfold(train_loader,valid_loader,model,loss_func,optimizer)
                loss /= KFold_epoch_int
                sub_loss_list.append(loss)
            loss_list.append(sub_loss_list)
        loss_array = np.array(loss_list)
        loss_mean_array = np.mean(loss_array,axis = 0)
        min_loc_int = np.argmin(loss_mean_array)
        learning_rate = lr_list[min_loc_int]
    else:
        learning_rate =1e-4

    # optimizer
    model = CNN_module(channel_int, out_size)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    test_loader = DataLoader(test_dataset,batch_size=batch_size,shuffle=True,drop_last=False)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=False)
    for t in range(epochs):
        print(f'epochs {t + 1}\n---------------------')
        train_loop(train_loader,model, loss_func, optimizer)
        test_loop(test_loader,model,loss_func)
    print('Done')
    torch.save(model.state_dict(),f'CNN_complete_Adam_50.pt')

if __name__ == '__main__':
    train = True
    if train:
        try:
            with open(r'D:\data\pd_frame\factors_set_2007-2009.pkl', 'rb') as f:
                factors = dill.load(f)
        except:
            factors = factor_set(2007, random=False, channel=5, period=3)
            # dump dataset
            with open(r'D:\data\pd_frame\factors_set_2007-2009.pkl', 'wb') as f:
                dill.dump(factors, f)
        train_dataset, test_dataset = dataset_split(0.1, factors)
        train_loss_list = []
        val_loss_list = []
        test_loss_list = []
        train_proc(False)
        loss_df = pd.DataFrame(train_loss_list)
        df_len = len(train_loss_list)
        loss_df.loc[:,'val_loss'] = val_loss_list+[0]*(df_len-len(val_loss_list))
        loss_df.loc[:,'test_loss'] = test_loss_list+[0]*(df_len-len(test_loss_list))
        loss_df.to_csv('CNN_complete_Adam_50_loss.csv')
    else:
        model = CNN_module(5,1478)
        model.load_state_dict(torch.load(f'CNN_complete_Adam_50.pt'))
        with open(r'D:\data\pd_frame\factors_set_2010.pkl', 'rb') as f:
            factors = dill.load(f)

        factor_sub = Subset(factors,list(range(504,752)))
        del factors
        factor_loader = DataLoader(factor_sub,batch_size=len(factor_sub.indices))
        for idx,(y,X) in enumerate(factor_loader):
            pred = model(X.float())
        pass


