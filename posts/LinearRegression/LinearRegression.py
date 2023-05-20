import numpy as np
import random
from math import e as e
import sklearn.metrics
from sklearn.datasets import make_blobs

class LinearRegression:

    def __init__(self) -> None:
        """
        Input:
        w: vector of weights
        history: history of accuracy scores

        This function declares the instance variables w and history
        """

        self.w = []
        self.score_history = []

    def pad(self,X):
        return np.append(X, np.ones((X.shape[0], 1)), 1)

    def fit_analytic(self, X, y, alpha = .1, max_epochs = 1000):
        """
        Input
        X: matrix of predictor variables
        y: vector of predictor variables
        iter: number of iterations. 1000 is the default number
        """
        X = self.pad(X)
        #self.w = np.array([random.uniform(-1,1) for i in range(len(X[0]))])

        self.w = np.linalg.inv((np.transpose(X)@X))@np.transpose(X)@y

    def fit_gradient(self, X, y, alpha = .0000001, max_epochs = 1000):
        """
        Input
        X: matrix of predictor variables
        y: vector of predictor variables
        iter: number of iterations. 1000 is the default number
        """
        X = self.pad(X)
        self.w = np.array([random.uniform(-1,1) for i in range(len(X[0]))])
        
        i = 0
        SCORE = 0
        P = np.transpose(X)@X
        q = np.transpose(X)@y

        while np.isclose(self.score(X, y, pad=False), SCORE) == False and i < max_epochs:
            #print(self.w.shape)
            #print(X.shape)
            SCORE = self.score(X, y, pad = False)

            self.score_history.append(self.score(X, y, pad= False))

            self.w -= alpha * 2*(P@self.w - q)
           #print(self.w.shape)
            i+=1

    
    def predict(self, X, padded = True):
        if padded: X = self.pad(X)
        return X@self.w

    def score(self, X, y, pad = True):
        """
        y: vector of predictor variables
        y_pred: predicted y values
        """

        y_ = np.sum(y)/len(y)
        y_pred = self.predict(X, padded = pad)
        return 1 - (np.sum((y_pred - y)**2))/(np.sum((y_ - y)**2))
    