import pandas as pd
import numpy as np


class inputs:
    def __init__(self,X,y,X_test):
        self.X = np.array(X)
        self.y = np.array(y)
        self.X_test = np.array(X_test)

class LogisticRegression:
    def __init__(self, learning_rate=0.01, limit=10000):
        self.__learning_rate = learning_rate
        self.__limit = limit
        self.__weights = None
        self.__bias = None
        
        
## it is worth mentioning, that i used a modified version of gradient descent in which i start by randomly trying to converge. This idea came from stochastic gradient descent
    @staticmethod 
    def sigmoid(z):
        return 1/(1+np.exp(-z))
    @staticmethod
    def f_wb(w,b,x):
            return LogisticRegression.sigmoid (np.dot(x,w)+b)
    @staticmethod
    def cost(w,b,x):
        m =x.shape[0] #mute variable for squaring
        prediction = LogisticRegression.f_wb(w,b,x)
        cost = (-1/m)*(np.dot(inputs.y,np.log(prediction))+np.dot((1-inputs.y),np.log(1-prediction)))

        return cost[0]


    @staticmethod
    def gradient_descent(w,b,alpha,num_iterations):
    #minimizing the cost function
        m = x.shape[0]
        for _ in range(num_iterations):
            predictions = LogisticRegression.f_wb(w, b)
                
            dw = (1/m) * np.dot(inputs.X.T, (predictions - inputs.y))
            db = (1/m) * np.sum(predictions - inputs.y)
                
            w = w - alpha * dw
            b = b - alpha * db
            
        return w, b
            
    def fit(self, X, y):
        self.inputs = inputs(X, y, None)  # Create inputs instance
        first_lim = self.__limit/2
        w = np.zeros(self.inputs.X.shape[1])  # Use shape[1] for feature count
        b = 0
        while self.cost(w,b) > first_lim:
            w = np.random.choice(self.inputs.X.flatten(), size=self.inputs.X.shape[1])
            b = np.random.choice(self.inputs.X.flatten())
        self.__weights, self.__bias = self.gradient_descent(w,b,self.__learning_rate,self.__limit)

    def predict(self, X_test):
        return self.f_wb(self.__weights, self.__bias, X_test)
    
