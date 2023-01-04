import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets
#gen data
data = datasets.load_boston()
data_array = data.data
target_array = data.target
target_array = target_array.reshape(-1,1)
params_array = np.ones(data_array.shape[1]).reshape(-1,1)

def mse(params_array,y_true_array,X_array):
    y_pred_array = np.matmul(X_array,params_array.reshape(-1,1))
    error_array = (y_true_array-y_pred_array).reshape(-1,1)
    mse_float = np.matmul(error_array.T,error_array)[0][0]/len(error_array)
    return mse_float

def sigmoid(y_array):
    return 1/(1+np.exp(-y_array))

def corss_entropy(params_array,y_true_array,X_array):
    y_pred_array = np.matmul(X_array, params_array.reshape(-1, 1))
    y_pred_prob_array = sigmoid(y_pred_array)
    error_float = np.sum(-y_true_array*np.log(y_pred_prob_array))
    return error_float

def gradient_ce(params_array,y_array,X_array):
    y_pred_array = np.matmul(X_array, params_array.reshape(-1, 1))
    y_pred_prob_array = sigmoid(y_pred_array)
    return np.matmul(X_array.T,y_pred_prob_array-y_array)

def gradient_mse(params_array,y_array,X_array,method_str):
    y_pred_array = np.matmul(X_array,params_array.reshape(-1,1))
    error_array = y_array-y_pred_array
    grad_array = -error_array * X_array
    len_int = grad_array.shape[0]
    if method_str=='SGD':
        SGD_size_int = int(len_int*0.1)
        random_idx_array = np.random.randint(0,len_int-1,SGD_size_int)
        grad_output_array = grad_array[random_idx_array]
        grad_output_array = np.mean(grad_output_array, axis=0)
    elif method_str=='GD':
        grad_output_array = np.mean(grad_array,axis=0)

    grad_output_array = grad_output_array.reshape(-1,1)
    return grad_output_array

class gradient_descent():
    def __init__(self,gradient_func,obj_func,params_array,y_array,X_array,n_iteration=100000,method = 'GD'):
        self.gradient = gradient_func
        self.obj_func = obj_func
        self.params_array = params_array
        self.y_array = y_array
        self.X_array = X_array
        self.GD_method = method
        self.n_iteration = n_iteration
    def Backtrack_LineSearch(self,alpha_float = None,beta_float = None):
        t_float = 1
        if alpha_float is None:
            alpha_float = np.random.uniform(0,0.5,1)[0]
        if beta_float is None:
            beta_float = np.random.uniform(0,1,1)[0]
        grad_array = self.gradient(self.params_array,self.y_array,self.X_array,self.GD_method)
        error_bef_float = self.obj_func(self.params_array,self.y_array,self.X_array)
        error_aft_float = self.obj_func(self.params_array-t_float*grad_array,self.y_array,self.X_array)
        while error_aft_float > error_bef_float-alpha_float*t_float*np.matmul(grad_array.T,grad_array)[0][0]:
            t_float = beta_float*t_float
            error_aft_float = self.obj_func(self.params_array - t_float * grad_array, self.y_array, self.X_array)
        return t_float

    def fit(self):
        error_list = []
        step_int = 0
        while len(error_list) <=10000 or step_int <= self.n_iteration:
            grad_array = self.gradient(self.params_array, self.y_array, self.X_array,self.GD_method)
            #backtrack line search
            lr_float = self.Backtrack_LineSearch(0.2,0.5)
            # update params
            delta_params_array = lr_float * grad_array
            self.params_array = np.add(self.params_array,-delta_params_array)
            error_float = self.obj_func(self.params_array,self.y_array,self.X_array)
            print(f'error: {error_float}')
            error_list.append(error_float)
            #iteration break
            step_int+=1
        print(f'Error: {error_float}\n Params: {self.params_array}')

model = gradient_descent(gradient_mse,mse,params_array,target_array,data_array,method='SGD')
model.fit()
#plotting
y_pred_array = np.matmul(model.X_array,model.params_array)
plt.plot(y_pred_array,label = 'pred')
plt.plot(target_array,label = 'ture')
plt.legend()
plt.show()
pass