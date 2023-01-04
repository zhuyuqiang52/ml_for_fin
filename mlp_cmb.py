import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader,TensorDataset
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda
import torch.nn.functional as F
from torch.utils.data.dataset import Dataset
from random import choice
import pickle
import bz2
import dill
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


class mlp_module(nn.Module):
    def __init__(self,in_size,out_size):
        super(mlp_module, self).__init__()
        self.sequential_cnn = nn.Sequential(
            nn.Linear(in_size,120),
            nn.Dropout(0.2),
            nn.ReLU(),
            nn.BatchNorm2d(1),
            nn.Linear(120, 88),
            nn.Dropout(0.2),
            nn.ReLU(),
            nn.Linear(88, 10),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(14780,out_size)
        )

    def forward(self,x):
        pred = self.sequential_cnn(x)
        return pred

class factor_set(Dataset):
    def __init__(self,year = 2003,random = False,channel = 1):
        if random:
            year = choice(list(range(2003,2009)))
        super().__init__()
        self.years = [year,year+1,year+2,year+3]
        self.path_str = model_dir
        self.frames = {}
        for year in self.years:
            fil = model_dir + "pandas-frames." + str(year) + ".pickle.bz2"
            self.frames.update(pickle.load(bz2.open(fil, "rb")))
        self.label_list = []
        self.frames_list = []
        self.id_amt_int = 0
        #index intersection
        id = set(self.frames.popitem()[1]['ID'].to_list())
        for x in self.frames:
            frame = self.frames[x]
            frame = frame.where(frame['IssuerMarketCap'] > 1e9).dropna(axis=0,how = 'all')
            # get liquad universe
            id = id.intersection(set(frame['ID'].tolist()))
        id = list(id)
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


try:
    with open(r'D:\data\pd_frame\factors_set_2007-2010_mlp.pkl', 'rb') as f:
        factors = dill.load(f)
except:
    factors = factor_set(2007,random=False, channel=1)
    #dump dataset
    with open(r'D:\data\pd_frame\factors_set_2007-2010_mlp.pkl','wb') as f:
        dill.dump(factors,f)

train_dataset,test_dataset = dataset_split(0.2,factors)

def train_loop(train_datloader,val_loader,model,loss_func,optimizer):
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
    print(f'Validation Loss:{np.mean(loss_list)}')

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
    print(f'Average Test Loss: {np.mean(loss_list)}')

def main():
    in_size = 12
    out_size = factors.id_amt_int
    model = mlp_module(in_size,out_size)
    # hyper-params
    learning_rate = 1e-1

    epochs = 50
    # loss func
    loss_func = nn.MSELoss()
    # optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, drop_last=False)
    for t in range(epochs):
        print(f'epochs {t + 1}\n---------------------')
        train_sub_dataset, val_dataset = dataset_split(0.1, train_dataset)
        train_loader = DataLoader(train_sub_dataset, batch_size=batch_size, shuffle=True, drop_last=False)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, drop_last=False)
        train_loop(train_loader, val_loader, model, loss_func, optimizer)
        test_loop(test_loader, model, loss_func)
    print('Done')
    torch.save(model,'MLP_SGD_EPOCH_50')

if __name__ == '__main__':
    main()