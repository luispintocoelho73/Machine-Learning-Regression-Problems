# -*- coding: utf-8 -*-
"""
Created on Sat Oct 23 13:58:12 2021

@author: Group 6- André Silva 90015 and Luis Coelho 90127
"""
# imports
import numpy as np
from sklearn.model_selection import KFold
from sklearn.svm import SVR
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import Lasso, HuberRegressor
import scipy.stats as stats
from numpy import mean, absolute

# importing the train files with outliers
X = np.load('Xtrain_Regression_Part2.npy')
y = np.load('Ytrain_Regression_Part2.npy')

# 
# creation of 100 (the number of sample in the training set) folds for cross validation
cv = KFold(n_splits=len(y.ravel()))
# a huber regressor was used since it is more resistant to outliers
reg = HuberRegressor().fit(X,y.ravel())
# the method cross_val_score was used in order to compute the MSE of each sample 
scores = cross_val_score(reg, X, y.ravel(), scoring='neg_mean_squared_error', cv=cv, n_jobs=-1)
y_pred_with_outliers = scores
#print(scores)
# then the zscores of each MSE value were computer, in order to identify outliers
z = np.abs(stats.zscore(y_pred_with_outliers))
#print(z)
# the indexes of the non-outlier samples were saved in an array 
# (samples whose zscore was less than 3)
non_outlier_indexes = [ i for i , zscore_value in enumerate(z) if zscore_value < 3]
# the outliers from the X matrix and y vector are removed
X_without_outliers = X[non_outlier_indexes]
y_without_outliers = y[non_outlier_indexes]

alpha_min = 0
MSE_min = 100000
alpha_i = 0.00001
# since the testing set was said to not have any outliers, the performances of the Lasso, Ridge, 
#Linear and Huber were compared
# The Lasso regression presented better results

# a "for" statement was used in order to find the ideal alpha value for our model
for i in range(1000):
    reg = Lasso(alpha=alpha_i).fit(X_without_outliers,y_without_outliers.ravel())
    # 10 folds were used in the cross validation process
    cv = KFold(n_splits=10, random_state=7, shuffle=True)
    # the mse values of each fold were computed
    scores = cross_val_score(reg, X_without_outliers, y_without_outliers.ravel(), scoring='neg_mean_squared_error', cv=cv, n_jobs=-1)
    # the average mse value for each alpha was computed
    mse = mean(absolute(scores))
    alpha_i += 0.00001
    # the best average mse value was registered
    if mse < MSE_min:
        MSE_min= mse
        alpha_min = alpha_i

print("For alpha ", alpha_min) # alpha_min = 0.00059
print("we got a MSE of ", MSE_min) # MSE_min = 0.0164309



# for alpha=0.0006 we obtained the prediction of our best model
final_lassoregression = Lasso(alpha=alpha_min)
final_model = final_lassoregression.fit(X_without_outliers,y_without_outliers)

# data from the test file is loaded
X_test = np.load('Xtest_Regression_Part2.npy')

# vector with the prediction of our 'best' model is defined and computed
y_prediction = final_model.predict(X_test)
