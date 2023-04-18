import numpy as np
import random
from math import e as e
import sklearn.metrics
from sklearn.datasets import make_blobs

class LogisticRegression:

    def __init__(self) -> None:
        """
        Input:
        w: vector of weights
        history: history of accuracy scores

        This function declares the instance variables w and history
        """

        self.w = []
        self.loss_history = []
        self.score_history = []

    #def Empirical_Risk():

    def pad(self,X):
        return np.append(X, np.ones((X.shape[0], 1)), 1)

    def fit(self, X, y, alpha = .1, max_epochs = 1000):
        """
        Input
        X: matrix of predictor variables
        y: vector of predictor variables
        iter: number of iterations. 1000 is the default number
        """

        X = self.pad(X)

        self.w = np.array([random.uniform(-1,1) for i in range(len(X[0]))])
        #self.w = np.append(self.w, 0)

        #y_hat = self.predict(X)
        LOSS = float("inf")
        i = 0
        while np.isclose(self.loss(X,y), LOSS) == False and i < max_epochs:
            
            self.loss_history.append(self.loss(X,y))
            LOSS = self.loss(X,y)
            self.score_history.append(self.score(X,y))
            
            self.w = self.w - alpha*((1/len(X))*self.gradient(X,y))
            i+=1

    def fit_stochastic(self, X, y, batch_size = 10 ,alpha = .1, max_epochs = 1000):
        """
        Input
        X: matrix of predictor variables
        y: vector of predictor variables
        iter: number of iterations. 1000 is the default number
        """

        X = self.pad(X)

        self.w = np.array([random.uniform(-1,1) for i in range(len(X[0]))])
        n = X.shape[0]
        LOSS = float("inf")
        i = 0

        while np.isclose(self.loss(X,y), LOSS) == False and i < max_epochs:
        #for i in range(max_epochs):
            self.loss_history.append(self.loss(X,y))
            LOSS = self.loss(X,y)
            self.score_history.append(self.score(X,y))

            order = np.arange(n)
            np.random.shuffle(order)

            for batch in np.array_split(order, n // batch_size + 1):
                x_batch = X[batch,:]
                y_batch = y[batch]
                grad = self.gradient( x_batch, y_batch)
                if len(batch) == batch_size: self.w = self.w - alpha*((1/batch_size)*np.sum(grad))
                else: self.w = self.w - alpha*((1/len(batch))*grad)
            i+=1 

    def gradient(self,X,y):
        #return (1/len(X)) * np.sum((self.omega(np.dot(self.w,X)) - y) * X)
        INSIDES = []
        for i in range(len(X)):
            INSIDES.append((self.omega(np.dot(self.w,X[i])) - y[i]) *X[i])
        
        return (1/len(X)) * np.sum(INSIDES, axis = 0)

    def predict(self, X):
        """
        Input:
        X: matrix of predictor variables

        Output:
        y_hat: A vector of predicted labels based on X
        """
        
        #y_hat = np.sign(np.dot(X,self.w))
        #y_hat[y_hat == -1] = 0
        #return y_hat
        return np.dot(X,self.w)

    def score(self, X, y):
        """
        Input:
        X: matrix of predictor variables
        y: vector of predictor variables

        Output:
        The accuracy of our prediction compared to the vector of labels
        """

        y_hat = self.predict(X)
        y_hat[y_hat > 0] = 1
        y_hat[y_hat < 0] = 0
        #return np.sum(np.equal(y,y_hat))/len(y)
        return sklearn.metrics.accuracy_score(y,y_hat)
    
    def omega (self, y_hat):
        return 1/(1+(e**(-y_hat)))
        #return 1 / (1 + np.exp(-y_hat))

    def loss(self, X, y):
        """
        Input:
        X: matrix of predictor variables
        y: vector of predictor variables

        Output:
        Returns oveall loss of current weights on X and y
        """
        y_hat = self.predict(X)
        O = self.omega(y_hat)
        return np.mean((-1*y)*np.log(O) - (1 - y)*np.log(1 - O))

    