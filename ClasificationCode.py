#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 25 14:00:51 2019

@author: christiaanbecker
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from matplotlib.colors import ListedColormap
from sklearn import datasets
import eli5
from eli5.sklearn import PermutationImportance


###################################################################logisic regresion############################################################
# Importing the dataset
train = pd.read_csv('titanic_train.csv')

def impute_age(cols):
    Age = cols[0]
    Pclass = cols[1]
    
    if pd.isnull(Age):

        if Pclass == 1:
            return 37

        elif Pclass == 2:
            return 29

        else:
            return 24

    else:
        return Age


train['Age'] = train[['Age','Pclass']].apply(impute_age,axis=1)
train.drop(['Cabin', 'PassengerId','Embarked','Name','Ticket'],axis=1,inplace=True)

X1 = train[['Pclass', 'Age', 'Parch', 'Fare', 'SibSp', 'Sex' ]]
y = train[['Survived']]

#Dummy coding
labelencoderSex = LabelEncoder()
X1['Sex'] = labelencoderSex.fit_transform(X1['Sex'])
Class = pd.get_dummies(X1.Pclass, prefix='Class')
#Class.drop(['Class_1'],axis=1,inplace=True)
X1.drop(['Pclass'],axis=1,inplace=True)
X = pd.concat([X1, Class], axis=1)

# Splitting the dataset into the Training set and Test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

# Feature Scaling
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Fitting Logistic Regression to the Training set
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print (cm)

#Accuracy Report
print(classification_report(y_test, y_pred))


perm = PermutationImportance(classifier, cv = None, refit = False, n_iter = 50).fit(X_train, y_train)
eli5.show_weights(perm, feature_names = X.columns.tolist(), top = 100)

#Feature importance plot
Betas2 = list(classifier.coef_)
Betas = Betas2[0]   
Col = list(X.columns) 
sns.barplot(x= Betas, y= Col)
plt.xlabel('Feature importance')
plt.ylabel('Features')
plt.title("Visualizing Betas")
plt.legend()
plt.show()


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

perm = PermutationImportance(classifier, cv = None, refit = False, n_iter = 50).fit(X_train, y_train)
perm_imp_eli5 = imp_df(X.columns, perm.feature_importances_)
var_imp_plot(perm_imp_eli5, 'Permutation feature importance (eli5)')
plt.show()






###################################################################Random forest clasification############################################################


# Importing the dataset
train = pd.read_csv('titanic_train.csv')

def impute_age(cols):
    Age = cols[0]
    Pclass = cols[1]
    
    if pd.isnull(Age):

        if Pclass == 1:
            return 37

        elif Pclass == 2:
            return 29

        else:
            return 24

    else:
        return Age


train['Age'] = train[['Age','Pclass']].apply(impute_age,axis=1)
train.drop(['Cabin', 'PassengerId','Embarked','Name','Ticket'],axis=1,inplace=True)

X1 = train[['Pclass', 'Age', 'Parch', 'Fare', 'SibSp', 'Sex' ]]
y = train[['Survived']]

#Dummy coding
labelencoderSex = LabelEncoder()
X1['Sex'] = labelencoderSex.fit_transform(X1['Sex'])
Class = pd.get_dummies(X1.Pclass, prefix='Class')
#Class.drop(['Class_1'],axis=1,inplace=True)
X1.drop(['Pclass'],axis=1,inplace=True)
X = pd.concat([X1, Class], axis=1)

# Splitting the dataset into the Training set and Test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

# Fitting Random Forest Classification to the Training set
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print(classification_report(y_test, y_pred))

perm = PermutationImportance(classifier, cv = None, refit = False, n_iter = 50).fit(X_train, y_train)
perm_imp_eli5 = imp_df(X.columns, perm.feature_importances_)
var_imp_plot(perm_imp_eli5, 'Permutation feature importance (eli5)')
plt.show()







