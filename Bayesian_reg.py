import numpy as np
import matplotlib.pyplot as plt
class Gauss:
    def __init__(self,Mu_array=None,S_array=None):
        # default Mu
        if Mu_array is None:
            Mu_array = np.zeros(1).reshape(-1, 1)
        # default S
        if S_array is None:
            S_array = np.diag(1).reshape(-1,1)
        self.dim = Mu_array.shape[0]
        self.Mu_array = Mu_array
        if S_array.shape[0]!=self.dim:
            try:
                raise Exception('dim not compatiable')
            except Exception as e:
                print(e)

        self.S_array = S_array
        self.S_inv_array = np.linalg.inv(S_array)

    def pdf(self,X_array:np.array)->float: # accept arrays with 2 dim shape
        dim_int = X_array.shape[0]
        pdf_float = 1/np.sqrt(np.power(2*np.pi,dim_int),self.S_inv_array)*\
              np.exp(-0.5*np.matmul(np.matmul((X_array-self.Mu_array).T,self.S_inv_array),X_array-self.Mu_array))
        return pdf_float

    def sample(self,n_sample):
        sample_array = np.random.multivariate_normal(self.Mu_array[0],self.S_array,n_sample)
        return sample_array

class Bayesian_Regression:
    def __init__(self,Mu_array=None,S_array=None,x_sigma=1.0):

        # default Mu
        if Mu_array is None:
            Mu_array = np.zeros(1).reshape(-1, 1)
        # default S
        if S_array is None:
            S_array = np.diag(1).reshape(-1,1)

        self.dim = Mu_array.shape[0]
        self.prior = Gauss(Mu_array,S_array)
        self.likelihood = 0.0
        self.posterior = 0.0
        self.w_array = Mu_array
        self.x_sigma = x_sigma

    def likelihood_build(self,X_array,w_array,sigma_float = 1.0):# X_array N*K,w_array K*1,
        # suppose residual iid with known variance sigma
        self.x_sigma = sigma_float
        mean_array = np.matmul(X_array,w_array)
        dim = mean_array.shape[0]
        S_array = sigma_float*np.diag(np.ones(dim))
        self.likelihood = Gauss(mean_array,S_array)

    def posterior_build(self,X_array,y_array):
        S_inv_array = 1/(self.x_sigma**2)*np.matmul(X_array.T,X_array)+self.prior.S_inv_array
        S_array = np.linalg.inv(S_inv_array)
        Mu_array = 1/(self.x_sigma**2)*np.matmul(np.matmul(S_array,X_array.T),y_array)
        self.posterior = Gauss(Mu_array,S_array)
        self.prior = Gauss(Mu_array,S_array) #update prior

    def train(self,X_array,y_array):
        self.posterior_build(X_array,y_array)

    def predict(self,X_array):
        y_sample_list = []
        for row,x_array in enumerate(X_array):
            x_array = x_array.reshape(-1,1)
            Mu_array = np.matmul(x_array.T,self.posterior.Mu_array)
            S_array = np.matmul(np.matmul(x_array.T,self.posterior.S_array),x_array)
            pred = Gauss(Mu_array,S_array)
            y_sample_list.append(pred.sample(1))
        return np.array(y_sample_list).reshape(-1,1)

def gen_dataset(n_sample_int = 1,features_int = 1 ,weight_array=None):
    x_array = np.ones(n_sample_int).reshape(-1,1)
    for i in range(features_int):#add intercept
        x_array = np.concatenate((x_array,np.arange(n_sample_int).reshape(-1,1)),axis=1)
    if weight_array is None:
        weight_array = np.random.rand(features_int+1,1)
    y_array = np.matmul(x_array,weight_array)+np.random.normal(0,1,n_sample_int).reshape(-1,1)
    return y_array,x_array,weight_array

def main():
    features_int = 10
    n_sample_int = 10000
    y_train,x_train,weight_array = gen_dataset(n_sample_int,features_int)
    #parameter
    mu_array = np.zeros(features_int+1).reshape(-1,1)
    S_array = np.diag(np.ones(features_int+1))
    model = Bayesian_Regression(mu_array,S_array)
    model.train(x_train,y_train)
    y_test,x_test,weight_array = gen_dataset(20,features_int,weight_array)
    y_pred = model.predict(x_test)
    plt.plot(y_test,label = 'test data')
    plt.plot(y_pred,label = 'pred data')
    plt.legend()
    plt.show()

if __name__=='__main__':
    main()


