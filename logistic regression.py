import numpy as np

# Assuming x and y are defined as before
x = np.array([1,2,3,4,5,6,7,8,9,10])
x = x.reshape(5,2)
y = np.array([0,1,0,1,1])  # Changed to binary outcomes
y = y.reshape(5,1)

def sigmoid(z):
    return 1 / (1 + np.exp(-z))
  #the function is the same as in linear regression, just logistic
def f_wb(w, b):
    return sigmoid(np.dot(x, w) + b)
 #modified the cost to have a log loss instead of a MSE (standard)
def cost(w, b):
    m = x.shape[0]
    predictions = f_wb(w, b)
    cost = (-1/m) * np.sum(y * np.log(predictions) + (1 - y) * np.log(1 - predictions))
    return cost
# computed the gradient by hand to get logistic regression gradients
def gradient_descent(w, b, num_iterations=1000, alpha=0.01):
    m = x.shape[0]
    for _ in range(num_iterations):
        predictions = f_wb(w, b)
        
        dw = (1/m) * np.dot(x.T, (predictions - y))
        db = (1/m) * np.sum(predictions - y)
        
        w = w - alpha * dw
        b = b - alpha * db
    
    return w, b

# Initialize parameters
w = np.zeros((x.shape[1], 1))
b = 0

# Run gradient descent
w, b = gradient_descent(w, b)

# Make predictions
y_pred = f_wb(w, b)
y_pred_class = (y_pred > 0.5).astype(int)  # Convert probabilities to class predictions

print("Final parameters:")
print("w =", w)
print("b =", b)
print("Predictions:", y_pred_class.T[0])
