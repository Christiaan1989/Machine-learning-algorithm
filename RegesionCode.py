#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 24 20:19:27 2019

@author: christiaanbecker
"""

import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd  
import seaborn as sns 
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.datasets import load_boston
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from eli5.sklearn import PermutationImportance

# function for creating a feature importance dataframe
def imp_df(column_names, importances):
    df = pd.DataFrame({'feature': column_names,
                       'feature_importance': importances}) \
           .sort_values('feature_importance', ascending = False) \
           .reset_index(drop = True)
    return df

# plotting a feature importance dataframe (horizontal barchart)
def var_imp_plot(imp_df, title):
    imp_df.columns = ['feature', 'feature_importance']
    sns.barplot(x = 'feature_importance', y = 'feature', data = imp_df, orient = 'h', color = 'royalblue') \
       .set_title(title, fontsize = 20)


#Load data 
boston_dataset = load_boston()
boston = pd.DataFrame(boston_dataset.data, columns=boston_dataset.feature_names)
boston.head()
boston['MEDV'] = boston_dataset.target


#########################################preprocesing###################################################################################### 
#Check missing vlues
print (boston.isnull().sum())
sns.heatmap(boston.isnull(),yticklabels=False,cbar=False,cmap='viridis')
plt.show()

#Correlation matrix
correlation_matrix = boston.corr().round(2)
sns.heatmap(data=correlation_matrix, annot= True, fmt='.0g')
plt.show()

#X and Y data
X = boston[["INDUS", "RM", "TAX", "RAD", "PTRATIO", "LSTAT", "AGE", "NOX" ]]
Y = boston[['MEDV']]

#Dummy coding
'''from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder = LabelEncoder()
X[:, 3] = labelencoder.fit_transform(X[:, 3])
onehotencoder = OneHotEncoder(categorical_features = [3])
X = onehotencoder.fit_transform(X).toarray()

# Avoiding the Dummy Variable Trap
X = X[:, 1:]'''

# Splitting the dataset into the Training set and Test set
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2)


##########################################Multiple regresion###################################################################################### 
# Fitting Multiple Linear Regression to the Training set
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Visualising the MR results
plt.scatter(X.index, Y, color = 'red')
plt.plot(X.index, regressor.predict(X), color = 'blue')
plt.title('Add Title')
plt.xlabel('Add Title')
plt.ylabel('Add Title')
plt.show()

# model evaluation for training set
y_train_predict = regressor.predict(X_train)
rmse = (np.sqrt(mean_squared_error(y_train, y_train_predict)))
r2 = r2_score(y_train, y_train_predict)

print("The MR model performance for training set")
print("--------------------------------------")
print('RMSE is {}'.format(rmse))
print('R2 score is {}'.format(r2))
print("\n")

# model evaluation for testing set
y_test_predict = regressor.predict(X_test)
rmse = (np.sqrt(mean_squared_error(y_test, y_test_predict)))
r2 = r2_score(y_test, y_test_predict)

print("The MR model performance for testing set")
print("--------------------------------------")
print('RMSE is {}'.format(rmse))
print('R2 score is {}'.format(r2))

#Feature importance plot
Betas2 = list(regressor.coef_)
Betas = Betas2[0]   
Col = list(X_train.columns) 
sns.barplot(x= Betas, y= Col)
plt.xlabel('Feature importance')
plt.ylabel('Features')
plt.title("Visualizing Betas")
plt.legend()
plt.show()

perm = PermutationImportance(regressor, cv = None, refit = False, n_iter = 50).fit(X_train, y_train)
perm_imp_eli5 = imp_df(X_train.columns, perm.feature_importances_)
var_imp_plot(perm_imp_eli5, 'Permutation feature importance (eli5)')
plt.show()

##########################################Support Vector Regression###################################################################################### 
#Feature Scaling
sc_X = StandardScaler()
sc_y = StandardScaler()
XT = sc_X.fit_transform(X)
YT = sc_y.fit_transform(Y)

# Splitting the dataset into the Training set and Test set
X_trainSVR, X_testSVR, y_trainSVR, y_testSVR = train_test_split(XT, YT, test_size = 0.2)

# Fitting SVR to the dataset
from sklearn.svm import SVR
regressorSVR = SVR(kernel = 'rbf')
regressorSVR.fit(X_trainSVR, y_trainSVR)

# Predicting a new result
y_predSVR = regressorSVR.predict(XT)
y_predSVR = sc_y.inverse_transform(y_predSVR)

# Visualising the SVR results
plt.scatter(np.arange(XT.shape[0]), Y, color = 'red')
plt.plot(np.arange(XT.shape[0]), y_predSVR, color = 'blue')
plt.title('Add Title')
plt.xlabel('Add Title')
plt.ylabel('Add Title')
plt.show()


# model evaluation for training set
y_train_predictSVR = regressorSVR.predict(X_trainSVR)
rmseSVR = (np.sqrt(mean_squared_error(y_trainSVR, y_train_predictSVR)))
r2SVR = r2_score(y_trainSVR, y_train_predictSVR)

print("The SVR model performance for training set")
print("--------------------------------------")
print('RMSE is {}'.format(rmseSVR))
print('R2 score is {}'.format(r2SVR))
print("\n")

# model evaluation for testing set
y_test_predictSVR = regressorSVR.predict(X_testSVR)
rmseSVR = (np.sqrt(mean_squared_error(y_testSVR, y_test_predictSVR)))
r2SVR = r2_score(y_testSVR, y_test_predictSVR)

print("The SVR model performance for testing set")
print("--------------------------------------")
print('RMSE is {}'.format(rmseSVR))
print('R2 score is {}'.format(r2SVR))

perm = PermutationImportance(regressorSVR, cv = None, refit = False, n_iter = 50).fit(X_trainSVR, y_trainSVR)
perm_imp_eli5 = imp_df(X_train.columns, perm.feature_importances_)
var_imp_plot(perm_imp_eli5, 'Permutation feature importance (eli5)')
plt.show()








