import numpy as np
import pandas as pd
from sklearn import datasets
import sklearn


class Data:
    def __init__(self):
        self.X_tr = None
        self.X_te = None
        self.X_va = None

        self.y_tr = None
        self.y_te = None
        self.y_va = None

    def random_data_with_f(self, f, D, N=1000, n=1000, v=1000):
        N = int(np.round(N))
        np.random.seed(12345)

        self.X_tr = np.random.uniform(0, 1, size=(N, D))
        self.y_tr = f(self.X_tr).reshape(-1, 1)
        self.X_tr -= 0.5

        self.X_te = np.random.uniform(0, 1, size=(n, D))
        self.y_te = f(self.X_te).reshape(-1, 1)
        self.X_te -= 0.5

        self.X_va = np.random.uniform(0, 1, size=(v, D))
        self.y_va = f(self.X_va).reshape(-1, 1)
        self.X_va -= 0.5

    def load_mnist(self):
        data = datasets.load_digits()
        X = data['data']
        X = (X - np.mean(X, 0)) / (np.std(X, 0) + 0.00001)
        y = data['target']
        y = pd.get_dummies(y).values

        ind = np.arange(0, y.shape[0], 1)
        np.random.shuffle(ind)

        tr = int(np.ceil(len(ind) * 0.8))
        te = int(np.ceil(len(ind) * 0.9))

        self.X_tr = X[np.where(ind[:tr])[0], :]
        self.X_te = X[np.where(ind[tr:te])[0], :]
        self.X_va = X[np.where(ind[te:])[0], :]

        self.y_tr = y[np.where(ind[:tr])[0], :]
        self.y_te = y[np.where(ind[tr:te])[0], :]
        self.y_va = y[np.where(ind[te:])[0], :]

    def load_crypto(self, LAG=10):
        df = pd.read_csv('dat/coinbase.csv').dropna()
        df['USD'] = (df['USD']-df['USD'].mean())/df['USD'].std()
        y=df[['USD']].values


        X = []
        for i in range(LAG+1,0-1,-1):
            if i > 0:
                X.append(y[LAG+1 - i:-i])
            else:
                X.append(y[LAG+1 - i:])



        X = np.concatenate(X, 1)
        y = X[:,-1].reshape(-1,1)
        X = X[:,:-1]
        #normalization
        # m=np.repeat(np.mean(X, 0).reshape(1,-1),X.shape[0],0)
        # s=np.repeat(np.mean(X, 0).reshape(1,-1),X.shape[0],0)
        # X= (X-m)/s
        X = X.reshape((X.shape[0], X.shape[1], 1)) # add the input dimension !

        ind = np.arange(0, y.shape[0], 1)
        tr = int(np.ceil(len(ind) * 0.8))
        te = int(np.ceil(len(ind) * 0.9))

        self.X_tr = X[np.where(ind[:tr])[0], :,:]
        self.X_te = X[np.where(ind[tr:te])[0], :,:]
        self.X_va = X[np.where(ind[te:])[0], :,:]

        self.y_tr = y[np.where(ind[:tr])[0], :]
        self.y_te = y[np.where(ind[tr:te])[0], :]
        self.y_va = y[np.where(ind[te:])[0], :]



self = Data()
