from math import exp
import numpy as np

import train

coef = train.coef

# Make a prediction with coefficients
def predict(row, coefficients):
    yhat = coefficients[0]
    for i in range(len(row)-1):
        yhat += coefficients[i + 1] * row[i]
    return 1.0 / (1.0 + exp(-yhat))
 
# test predictions

input = np.genfromtxt("data.csv", delimiter=',')
input = input[1:,:]
input = input.tolist()
input = input[::100]

per = 0

for row in input:
    yhat = predict(row, coef)
    if row[-1]==0 or row[-1]==1:
        print("Expected=%.3f, Predicted=%.3f [%d]" % (row[-1], yhat, round(yhat)))
        if row[-1] == round(yhat):
            per+=1
    else:
        print("Predicted=%.3f [%d]" % (yhat, round(yhat)))
        
print(per/len(input))