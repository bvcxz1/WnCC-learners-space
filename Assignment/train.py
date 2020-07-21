# Import libraries

from math import exp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sn

# Read data
data = np.genfromtxt("data.csv", delimiter=',')
data = data[1:,:]
print(data)

# Model the data
def mod():
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Key: Dark=0 Light=1')
    plt.scatter(data[:,0],data[:,1],c=data[:,2])
    plt.show()

# Make a prediction with coefficients
def predict(row, coefficients):
    yhat = coefficients[0]
    for i in range(len(row)-1):
        yhat += coefficients[i + 1] * row[i]
    return 1.0 / (1.0 + exp(-yhat))    
    
# Estimate logistic regression coefficients using stochastic gradient descent
def coefficients_sgd(train, l_rate, n_epoch):
    coef = [0.0 for i in range(len(train[0]))]
    for epoch in range(n_epoch):
        sum_error = 0
        for row in train:
            yhat = predict(row, coef)
            error = row[-1] - yhat
            sum_error += error**2
            coef[0] = coef[0] + l_rate * error * yhat * (1.0 - yhat)
            for i in range(len(row)-1):
                coef[i + 1] = coef[i + 1] + l_rate * error * yhat * (1.0 - yhat) * row[i]
        print('>epoch=%d, lrate=%.3f, error=%.3f' % (epoch, l_rate, sum_error))
    return coef
 
# Randomly initialising coefficients
coefficients=[10.5,0,0]
    
# Calculate coefficients
dataset = data.tolist()
l_rate = 0.3
n_epoch = 100
coef = coefficients_sgd(dataset, l_rate, n_epoch)
print(coef)