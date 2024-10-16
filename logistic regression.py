import pandas as pd
import numpy as np


class inputs:

    def __init__(self,X:np.array,y:np.array,X_test:np.array):
        self.X = np.array(X)
        self.y = np.array(y)
        self.X_test = np.array(X_test)




class LogisticRegression:
    def __init__(self, learning_rate:float =0.01 , limit:float =10000, num_iterations:int=1000):
        self.learning_rate = learning_rate
        self.limit = limit
        self.weights = None
        self.bias = None
        self.num_iterations = num_iterations
        
        
## it is worth mentioning, that i used a modified version of gradient descent in which i start by randomly trying to converge. This idea came from stochastic gradient descent
    @staticmethod 
    def sigmoid(z:np.float64):
        return 1/(1+np.exp(-z))
    @staticmethod
    def f_wb(w:np.array,b:float,x:np.array):
            return LogisticRegression.sigmoid (np.dot(x,w)+b)
    @staticmethod
    def cost(w:np.array,b:float,x:np.array):
        m =x.shape[0] #mute variable for squaring
        prediction = LogisticRegression.f_wb(w,b,x)
        cost = (-1/m)*(np.dot(inputs.y,np.log(prediction))+np.dot((1-inputs.y),np.log(1-prediction)))

        return cost[0]


    @staticmethod
    def gradient_descent(w:np.array, b:float, x:np.array, y:np.array, alpha:float, limit:float)->tuple[np.array, float]:
        m = x.shape[0]
        iterations = 0
        prev_cost = float('inf')
        
        while True:
            # Logistic regression prediction
            predictions = LogisticRegression.f_wb(w, b, x)
            
            # Compute gradients for logistic regression
            dw = (1/m) * np.dot(x.T, (predictions - y))
            db = (1/m) * np.sum(predictions - y)
            
            # Update parameters
            w = w - alpha * dw
            b = b - alpha * db
            
            # Compute current cost for logistic regression
            current_cost = LogisticRegression.cost(w, b, x)
            
            # Check convergence
            if abs(prev_cost - current_cost) < limit or iterations >= 10000:
                break
            
            prev_cost = current_cost
            iterations += 1
        
        return w, b
    def fit(self, X, y):
        self.inputs = inputs(X, y, None)  # Create inputs instance
        first_lim = self.limit/2
        w = np.zeros(self.inputs.X.shape[1])  # Use shape[1] for feature count
        b = 0
        while self.cost(w,b) > first_lim:
            w = np.random.choice(self.inputs.X.flatten(), size=self.inputs.X.shape[1])
            b = np.random.choice(self.inputs.X.flatten())
        self.__weights, self.__bias = self.gradient_descent(w,b,self.learning_rate,self.limit)

    def predict(self, X_test):


        return self.f_wb(self.__weights, self.__bias, X_test)

    
