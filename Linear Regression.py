import pandas as pd
import numpy as np


class inputs:
    def __init__(self,X,y,X_test):
        self.X = np.array(X)
        self.y = np.array(y)
        self.X_test = np.array(X_test)

class LinearRegression:
    def __init__(self, learning_rate=0.01, limit=10000):
        self.__learning_rate = learning_rate
        self.__limit = limit
        self.__weights = None
        self.__bias = None
        
        
## it is worth mentioning, that i used a modified version of gradient descent in which i start by randomly trying to converge. This idea came from stochastic gradient descent
    @staticmethod
    def f_wb(w,b,x):
            s = 0
            x = np.array(x)
            for i in x.T:
                f_wb = np.dot(i, w) + b
                s += f_wb  # creating a linear function
            return s
    @staticmethod
    def cost(w,b):
        m =0 #mute variable for squaring
        sum = 0

        for i in range(inputs.X.shape[1]):
            m = (f_wb(w,b) - inputs.y[i])**2
        sum += m

        cost = sum/(2*(inputs.X.shape[1]))
        return cost[0]


    @staticmethod
    def gradient_descent(w,b,alpha,limit):
    #minimizing the cost function

  

        dw_sum = 0
        db_sum = 0
        while LinearRegression.cost(w,b) > limit:
            for i in range(inputs.X.shape[1]):
                dw_sum += (LinearRegression.f_wb(w,b,inputs.X[i]) - inputs.y[i])*inputs.X[i]
                db_sum += (LinearRegression.f_wb(w,b,inputs.X[i]) - inputs.y[i])
        dw = dw_sum/inputs.X.shape[1]
        db = db_sum/inputs.X.shape[1]

        #update parameters
        for i in range(inputs.X.shape[0]):
            w[i] = w[i] - alpha*dw[0]
            b = b - alpha*db[0]

        return w,b
    
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
    
