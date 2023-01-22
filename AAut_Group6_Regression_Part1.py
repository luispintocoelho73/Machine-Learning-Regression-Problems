# -*- coding: utf-8 -*-
"""
Created on Tue Oct 12 11:50:10 2021

@author: AndreSilva 90015 + LuisCoelho 90127 (Group 6)
"""

# linear and ridge regression models were also tested, 
# but the lasso regresion model was found to have the best performance

# this is the code that generated the zipped prediction vector

#imports
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error as MSE
from sklearn.metrics import balanced_accuracy_score as BACC
from sklearn.linear_model import Lasso


def scores(y_real,y_pred,mode):
    ###y_real - ground truth vector 
    ###y_pred - vector of predictions, must have the same shape as y_real
    ###mode   - if evaluating regression ('r') or classification ('c')
    
    if y_real.shape != y_pred.shape:
       print('confirm that both of your inputs have the same shape')
    else:
        if mode == 'r':
            mse = MSE(y_real,y_pred)
            #print('The Mean Square Error is', mse)
            return mse
        
        elif mode == 's':
            sse = 0
            for i in range(10):
                sse_aux = (y_real[i]-y_pred[i])*(y_real[i]-y_pred[i])
                sse += sse_aux
            #print('The Sum of Square Errors is', sse)
            return sse
            
        
        elif mode == 'c':
            bacc = BACC(y_real,y_pred)
            print('The Balanced Accuracy is', bacc)
            return bacc
        
        else:
            print('You must define the mode input.')

#def 

# variable declaration
alpha_iteration = 0.0001
MSE_min = 1000
SSE_min = 0
alpha_min = 10000
i = 0
# iterate over different values of alpha to assess which is the preferable value
# alpha values from 0.0001 to 10 with step size of 0.0005
for i in range(20000):
    
    # data from the train files is loaded into these variables
    x_allfolds = np.load('Ypred_Regression_Part1.npy')
    #y_allfolds = np.load('Ytrain_Regression_Part1.npy')
    
    # generates the folds with the same random state (7) to allow for comparison between models
    # (9 folds are used for training and 1 fold is used for validation)
    kf = KFold(n_splits=10, random_state=7, shuffle=True)
    
    # variable and list declaration
    model = [ 0 for aux in range(10)] 
    y_pred = [0 for aux in range(10)] 
    MSE_iteration = [0 for aux in range(10)] 
    SSE_iteration = [0 for aux in range (10)] 
    MSE_total = 0 
    SSE_total = 0 
    MSE_avg = 0 
    SSE_avg = 0 
    aux = 0 
    
    # split ouput data into folds
    kf.split(y_allfolds)
    
    # split the input data into folds and iterate over each fold
    for train_index, validation_index in kf.split(x_allfolds):
        
        # train variables correspond to the train index and 
        # validation variables to the validation index
        x_train, x_validation = x_allfolds[train_index], x_allfolds[validation_index]
        y_train, y_validation = y_allfolds[train_index], y_allfolds[validation_index]
        
        # the lasso regression mode is defined with a given alpha
        lassoregression = Lasso(alpha=alpha_iteration)  
        
        # fit lasso model with the training folds' data using coordinate descent
        model[aux] = lassoregression.fit(x_train,y_train)
        
        # obtaining the predicted ouputs based on the validation folds
        y_pred[aux] = model[aux].predict(x_validation)
        
        # SSE and MSE are computed by comparing the y_validation set with the predicted values
        MSE_iteration[aux] = scores(y_validation, y_pred[aux].reshape(10,1), "r")
        SSE_iteration[aux] = scores(y_validation, y_pred[aux].reshape(10,1), "s")
        
        MSE_total += MSE_iteration[aux]
        SSE_total += SSE_iteration[aux]
        aux += 1
        
    # computing the performance of a lasso model using multiple rounds of cross-validation
    MSE_avg = float(MSE_total)/10
    SSE_avg = float(SSE_total)/10
    #print("The average MSE of this model, obtained through cross-validation, is: ", MSE_avg)
    #print("The average SSE of this model, obtained thorugh cross-validation, is: ", SSE_avg)
    
    # if an alpha parameter that produces a lower MSE value has been found, it is registered
    if MSE_avg < MSE_min:
        MSE_min= MSE_avg
        alpha_min = alpha_iteration
        SSE_min = SSE_avg
        
    # updating variables i and alpha
    alpha_iteration += 0.0005
    i += 1
    print(i)
    
print("For alpha ", alpha_min) # alpha_min = 0.0026
print("we got a MSE of ", MSE_min) # MSE_min = 0.01675
print("And a SSE of ", SSE_min) # SSE_min = 0.1675


# data from the train files is loaded into these variables
X = np.load('Xtrain_Regression_Part1.npy')
y = np.load('Ytrain_Regression_Part1.npy')

# for alpha=0.0026 we obtained the prediction of our best model
final_lassoregression = Lasso(alpha=alpha_min)
final_model = final_lassoregression.fit(X,y)

# data from the test fail is loaded
X_test = np.load('Xtest_Regression_Part1.npy')

# vector with the prediction of our 'best' model is defined and computed
y_prediction = final_model.predict(X_test)


