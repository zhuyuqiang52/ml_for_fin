import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets
import pandas as pd

data = pd.read_table('marks.txt',sep=',',header=None).values
y_array = data[:,2].reshape(-1,1)
X_array = data[:,0:2].reshape(-1,2)
X_array = np.concatenate([X_array,np.ones(X_array.shape[0]).reshape(X_array.shape[0],1)],axis=1)
params_array = np.zeros(X_array.shape[1]).reshape(-1,1)

def sigmoid(y_array):
    sigmoid_array = 1/(1+np.exp(-y_array))-1e-4
    sigmoid_array[sigmoid_array<0.0] = 1e-4
    return sigmoid_array

def cross_entropy(params_array,y_true_array,X_array):
    y_pred_array = np.matmul(X_array, params_array.reshape(-1, 1))
    y_pred_prob_array = sigmoid(y_pred_array)
    error_float = -np.sum(y_true_array*np.log(y_pred_prob_array)+(1-y_true_array)*np.log(1-y_pred_prob_array))
    return error_float

def contour_loss(x1, x2):
    error_array = np.zeros(shape=(x1.shape[0],x2.shape[0]))
    for i in range(x1.shape[0]):
        for j in range(x2.shape[0]):
            w0 = x1[0,i]
            w1 = x2[j,0]
            param_array = np.array((w1,w0,-24)).reshape(-1,1)
            error_array[i,j] = cross_entropy(param_array,y_array,X_array)
    return error_array

def gradient_ce(params_array,y_array,X_array,method_str='GD'):
    y_pred_array = np.matmul(X_array, params_array.reshape(-1, 1))
    y_pred_prob_array = sigmoid(y_pred_array)
    grad_array = X_array*(y_pred_prob_array-y_array)
    if method_str=='SGD':
        len_int = len(grad_array)
        SGD_size_int = int(len_int*0.3)
        random_idx_array = np.random.randint(0,len_int-1,SGD_size_int)
        grad_output_array = grad_array[random_idx_array]
        grad_output_array = np.sum(grad_output_array, axis=0)
    elif method_str=='GD':
        grad_output_array = np.mean(grad_array,axis=0)
    return grad_output_array.reshape(-1,1)

def hessian(params_array,X_array):
    y_pred_array = np.matmul(X_array, params_array.reshape(-1, 1))
    y_pred_prob_array = sigmoid(y_pred_array)
    S_w_array = np.diag(y_pred_prob_array.flatten())
    H_array = np.matmul(np.matmul(X_array.T,S_w_array),X_array)
    return H_array

class Newton_Method:
    def __init__(self,params_array,y_array,X_array,obj_func,grad_func,Hessian_func,n_interation_int = 50000,lr_float = None,method = 'GD'):
        self.params_array = params_array
        self.y_array = y_array
        self.X_array = X_array
        self.obj_func = obj_func
        self.gradient = grad_func
        self.hessian = Hessian_func
        self.n_interation = n_interation_int
        self.lr_float = lr_float
        self.method_str = method
        self.params_list = []
        self.loss_list = []
    def Backtrack_LineSearch(self,alpha_float = None,beta_float = None):
        t_float = 1
        if alpha_float is None:
            alpha_float = np.random.uniform(0,0.5,1)[0]
        if beta_float is None:
            beta_float = np.random.uniform(0,1,1)[0]
        grad_array = self.gradient(self.params_array,self.y_array,self.X_array,self.method_str)
        error_bef_float = self.obj_func(self.params_array,self.y_array,self.X_array)
        error_aft_float = self.obj_func(self.params_array-t_float*grad_array,self.y_array,self.X_array)
        while error_aft_float > error_bef_float-alpha_float*t_float*np.matmul(grad_array.T,grad_array)[0][0]:
            t_float = beta_float*t_float
            error_aft_float = self.obj_func(self.params_array - t_float * grad_array, self.y_array, self.X_array)
        if t_float<1e-3:
            t_float = 1e-3
        return t_float

    def fit(self):
        for i in range(self.n_interation):
            grad_array = self.gradient(self.params_array,self.y_array,self.X_array)
            hessian_array = hessian(self.params_array,X_array)
            delta_params_array = np.matmul(np.linalg.pinv(hessian_array),grad_array)
            if self.lr_float is None:
                lr_float = self.Backtrack_LineSearch()
            else:
                lr_float = self.lr_float
            self.params_array = self.params_array-lr_float*delta_params_array
            self.params_list.append(self.params_array)
            error_float = cross_entropy(self.params_array,y_array,X_array)
            self.loss_list.append(error_float)
            print(f'error {i}:{error_float}')
        print(f'params: {self.params_array}')

    def plot_params(self):
        params_history_array = np.concatenate(self.params_list,axis=1)
        params_size_int = params_history_array.shape[1]
        plt.plot(self.loss_list, label='loss')
        plt.legend()
        plt.show()
        plt.figure(figsize=(10,5))
        w0_max_float = np.max(params_history_array[1,:])
        w0_min_float = np.min(params_history_array[1,:])
        w1_max_float = np.max(params_history_array[0,:])
        w1_min_float = np.min(params_history_array[0,:])
        w0_range_array = np.linspace(w0_min_float,2*w0_max_float-w0_min_float,150)
        w1_range_array = np.linspace(w1_min_float,2*w1_max_float-w1_min_float,150)
        W0,W1 = np.meshgrid(w0_range_array,w1_range_array)
        z = contour_loss(W0,W1)
        contours = plt.contour(W0,W1,z,40)
        plt.clabel(contours,inline=True,fontsize=10)
        plt.plot(params_history_array[1,:],params_history_array[0,:])
        sample_array = np.linspace(params_size_int-1,50,dtype=np.int64)
        plt.plot(params_history_array[1,sample_array],params_history_array[0,sample_array],'*',label = 'contour_loss',alpha=0.5)
        plt.scatter(params_history_array[1,-1],params_history_array[0,-1],marker='o',s = 50)
        plt.legend()
        plt.show()
        pass
model = Newton_Method(params_array,y_array,X_array,cross_entropy,gradient_ce,hessian,n_interation_int=20000,lr_float = 1,method='GD')
model.fit()
model.plot_params()
y_pred_array = np.matmul(X_array,model.params_array)
y_prob_array = sigmoid(y_pred_array)
y_class_array = (y_prob_array>=0.5).astype(int)
accuracy_float = np.sum((y_array-y_class_array)==0)/len(y_array)*100
print(f'accuracy: {accuracy_float}%')
pass