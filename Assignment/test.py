from math import exp
import numpy as np
import pandas as pd

import train

coef = train.coef

# Make a prediction with coefficients
def predict(row, coefficients):
    yhat = coefficients[0]
    for i in range(len(row)-1):
        yhat += coefficients[i + 1] * row[i]
    return 1.0 / (1.0 + exp(-yhat))
 
def expand(data):
    # Expanding 2 dimensional data into a 13 dimensional quantity
    dataset = np.zeros((len(data),14))
    # 4
    dataset[:,0] = data[:,0]**4
    dataset[:,1] = (data[:,0]**3)*data[:,1]
    dataset[:,2] = (data[:,0]**2)*(data[:,1]**2)
    dataset[:,3] = data[:,0]*(data[:,1]**3)
    dataset[:,4] = data[:,1]**4
    # 3
    dataset[:,5] = data[:,0]**3
    dataset[:,6] = (data[:,0]**2)*(data[:,1])
    dataset[:,7] = (data[:,0])*(data[:,1]**2)
    dataset[:,8] = data[:,1]**3
    # 2
    dataset[:,9]  = data[:,0]**2
    dataset[:,10] = data[:,0]*data[:,1]
    dataset[:,11] = data[:,1]**2
    # 1
    dataset[:,12] = data[:,0]
    dataset[:,13] = data[:,1]
    
    return dataset
    
# test predictions

df = pd.read_json('input.json')
data = df[df.columns[0:2]]
dataset = np.array(data)
input_ = expand(dataset)
test = input_.tolist()

i=0
for row in test:
    yhat = predict(row, coef)
    if row[-1]==0 or row[-1]==1:
        print("Expected=%.3f, Predicted=%.3f [%d]" % (row[-1], yhat, round(yhat)))
        #if row[-1] == round(yhat):
            #per+=1
    else:
        print("%d) Predicted=%d" % (i+1, round(yhat)))
        i+=1
