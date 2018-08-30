#!/usr/bin/env python

# This program uses KNN to determine the type of signal that is being inputted based
# on the signal's bandwidth and shape
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import numpy as np

#n value
#n = 3
iterations = 100

#loads data from .csv into a datafile and shuffles/randomizes the order
df = pd.read_csv("handmeasured_data.csv", header = 0)
#df = pd.read_csv("computercollected_data.csv", header = 0)  
df = df.sample(frac=1).reset_index(drop=True) 

#df = df.drop(['frequency'],1)

k_max = 8
percent_training = .7 # % of data used for training

# Trains and test using different k values up to max
for k_value in range (1, k_max):
	score = 0
	for i in range (0, iterations):
		#training set
		#https://stackoverflow.com/questions/24147278/how-do-i-create-test-and-train-samples-from-one-dataframe-with-pandas
		mask = np.random.rand(len(df)) < percent_training   #chooses which data to use for training and which for testing.
		train = df[mask]
		#test set
		test = df[~mask]

		y = train['class'] #answers that the program should give
		x = train.drop(['class'],1) #features that are used to find the answer

		#creates KNN
		knn = KNeighborsClassifier(algorithm='auto', leaf_size=30, metric="euclidean", metric_params=None, n_jobs=1, n_neighbors=k_value, p=2, weights='uniform')

		knn.fit(x,y) #KNN training

		test_no_class = test.drop(['class'],1) #drop "class fields" from csv file to create test data

		a = knn.predict(test_no_class) # predicts class using KNN
		score += sum(a == test['class'])/float(len(test) * iterations) #compare results w/ answer
	print k_value 
	print score
