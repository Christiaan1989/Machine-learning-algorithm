# -*- coding: utf-8 -*-
"""
Created on Mon Dec 16 15:52:34 2019

@author: beckerchr
"""

from os import chdir
import pandas as pd
from surprise import Dataset
from surprise import Reader
import seaborn as sns
import matplotlib.pyplot as plt
from surprise.model_selection import cross_validate
from surprise import NormalPredictor
from surprise import KNNBasic
from surprise import KNNWithMeans
from surprise import KNNWithZScore
from surprise import KNNBaseline
from surprise import SVD
from surprise import BaselineOnly
from surprise import SVDpp
from surprise import NMF
from surprise import SlopeOne
from surprise import CoClustering
from surprise.accuracy import rmse
from surprise import accuracy
from surprise.model_selection import train_test_split
from surprise.model_selection import GridSearchCV

chdir(r"C:\Users\beckerchr\Desktop\AI Projects")   

####################################Movie_database###############################################

data1 = pd.read_csv('BX-Book-Ratings.csv', sep=';', encoding='iso-8859-1')

####################################Chomme_database#############################################

#dataFil = pd.read_excel('Chomme.xlsx')
#reader = Reader(rating_scale=(1, 10))

####################################Data_exploration#############################################

RatingCount = data1['Book-Rating'].value_counts().sort_values().plot(kind = 'barh')
plt.show()

BooksRatedCount = data1.groupby('ISBN')['Book-Rating'].count().reset_index().sort_values('Book-Rating', ascending=False)
sns.distplot(BooksRatedCount["Book-Rating"], hist=False, rug=True)
plt.show()

UsersRatedCount = data1.groupby('User-ID')['Book-Rating'].count().reset_index().sort_values('Book-Rating', ascending=False)
sns.distplot(UsersRatedCount["Book-Rating"], hist=False, rug=True)
plt.show()

####################################Data_Cleaning###################################################

min_book_ratings = 50
filter_books = data1['ISBN'].value_counts() > min_book_ratings
filter_books = filter_books[filter_books].index.tolist()

min_user_ratings = 50
filter_users = data1['User-ID'].value_counts() > min_user_ratings
filter_users = filter_users[filter_users].index.tolist()

df_new = data1[(data1['ISBN'].isin(filter_books)) & (data1['User-ID'].isin(filter_users))]
print('The original data frame shape:\t{}'.format(data1.shape))
print('The new data frame shape:\t{}'.format(df_new.shape))

df_new2 = df_new[df_new["Book-Rating"] != 0]

####################################Model_Selection###################################################

reader = Reader(rating_scale=(1, 10))
data = Dataset.load_from_df(df_new2[['User-ID', 'ISBN', 'Book-Rating']], reader)

Algorithms = ['SVD', 'SVDpp', 'SlopeOne', 'NMF',' NormalPredictor', 'KNNBaseline',  'KNNBasic', 'KNNWithMeans', 'KNNWithZScore', 'BaselineOnly', 'CoClustering']

RMSEScore = []
# Iterate over all algorithms
for algorithm in [SVD(), SVDpp(), SlopeOne(), NMF(), NormalPredictor(), KNNBaseline(), KNNBasic(), KNNWithMeans(), KNNWithZScore(), BaselineOnly(), CoClustering()]:
    # Perform cross validation
    results = cross_validate(algorithm, data, measures=['RMSE'], cv=3, verbose=False)
    RM = results['test_rmse'].mean()
    RMSEScore.append(RM)
    
ResDF = pd.DataFrame(list(zip(Algorithms, RMSEScore)), columns=['Algorithm','RMSE Score']).sort_values('RMSE Score', ascending=True)
    
####################################Parameter_Selection###################################################

param_grid = {'n_epochs': [20, 25], 
                  'lr_all': [0.007, 0.009, 0.01],
                  'reg_all': [0.4, 0.6]}

gs = GridSearchCV(SVDpp, param_grid, measures=['rmse', 'mae'], cv=5, n_jobs=5)
gs.fit(data)

print(gs.best_score["rmse"])
print(gs.best_params["rmse"])

####################################Model_Fit_Predict###################################################

algo = gs.best_estimator['rmse']
algo.fit(data.build_full_trainset())

uid = str(250709)
iid = str(671042262)

pred = algo.predict(uid, iid, r_ui=5, verbose=True)


















