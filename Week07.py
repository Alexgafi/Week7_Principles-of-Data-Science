# -*- coding: utf-8 -*-
"""
Created on Tue Nov 14 18:53:37 2023

@author: adfw980
"""

#Load the Data in a DataFrame

f = open('C:/Users/adfw980/Downloads/censusCrimeClean (2).csv')

import csv

import pandas as pd

df1 = pd.DataFrame(csv.reader(f))

#Skip the first column

df1 = df1.iloc[:, 1:]



#CHange the name of the columns and skip the first row in 

column_names = df1.iloc[0].to_numpy()

df1.columns = column_names

#Skip the first row as it is not numerical

df1 = df1.iloc[1:]

# Transform the DataFrame into Variables

variables = df1.values

#Fit the PCA

from sklearn.decomposition import PCA

pca = PCA(n_components=2)
pca.fit(df1)

#We will observe the explained variance ratio of the PCA

explained_variance_ratio = pca.explained_variance_ratio_ 

print (explained_variance_ratio)

#[0.67387831 0.08863102]

#We will standardise the features and test the explained_variance_ratio again

from sklearn.preprocessing import StandardScaler

df2 = StandardScaler().fit_transform(df1)

explained_variance_ratio2 = pca.explained_variance_ratio_ 

print (explained_variance_ratio2)

# [0.25267231 0.16667711] The explained variance ratio decreased for the 1st component and 
#increased for the second one. 

#We transform the samples into the principal components and plot them on a scatterplot

df1_transformed = pca.fit_transform(df1)

import matplotlib.pyplot as plt

x = df1_transformed.columns[0]
y = df1_transformed.columns[1]

plt.scatter(x, y)

components = pca.components_

print(components)

#We will put the values in a dataframe, transpose rows with columns 
#Get the absolute of the values
#Sort the values in the dataframe

components = pd.DataFrame(components)

components.columns = df1.columns

components = components.transpose()

components = components.abs()

components = components.sort_values(by= components[0])

#Exercise 2


