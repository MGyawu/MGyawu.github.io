import numpy as np
import pandas as pd
import seaborn as sns
import random
import sklearn.metrics
from matplotlib import pyplot as plt

from sklearn.datasets import make_blobs

class Perceptron:

    def __init__(self) -> None:
        """
        Input:
        w: vector of weights
        history: history of accuracy scores

        This function declares the instance variables w and history
        """

        self.w = []
        self.history = []

    def fit(self, X, y, iter = 1000):
        """
        Input
        X: matrix of predictor variables
        y: vector of predictor variables
        iter: number of iterations. 1000 is the default number
        """

        self.w = np.array([random.uniform(-100,100) for i in range(len(X[0]))])
        #self.w = np.append(self.w, 0)

        y_hat = self.predict(X)
        i = 0
        while i < iter and self.score(X,y) != 1.0:
            self.history.append(self.score(X,y))
            rand_index = random.randint(0,len(X)-1)
            self.w = self.w + np.sign(np.dot(self.w, X[rand_index]))*y_hat[rand_index]*X[rand_index]
            y_hat = self.predict(X)
            i+=1
        if self.score(X,y) == 1.0: self.history.append(self.score(X,y))

    def predict(self, X):
        """
        Input:
        X: matrix of predictor variables

        Output:
        y_hat: A vector of predicted labels based on X
        """
        
        y_hat = np.sign(np.dot(X,self.w))
        y_hat[y_hat == -1] = 0
        return y_hat

    def score(self, X, y):
        """
        Input:
        X: matrix of predictor variables
        y: vector of predictor variables

        Output:
        The accuracy of our prediction compared to the vector of labels
        """

        y_hat = self.predict(X)
        #return np.sum(np.equal(y,y_hat))/len(y)
        return sklearn.metrics.accuracy_score(y,y_hat)