"""
This is the Data Preprocessing before fitting it to the polynomial regression model for predicting
S/R.
"""
# Data Preprocessing for the dataset before applying any model to it.

# Importing the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Importing the dataset
dataset = pd.read_csv('latest_new.csv')

# Changing the numerical date to textual form
dataset['Date'] = pd.to_datetime(dataset['Date'])

# creating new column with new values from specific date
dataset['DayOfMonth'] = dataset['Date'].dt.day_name()
dataset['MonthOfYear'] = dataset['Date'].dt.month_name()
dataset['Year'] = dataset['Date'].dt.year

# taking independent variables and dependent variables
x = dataset.iloc[:, [4,5,6]].values
y = dataset.iloc[:, [1]].values

# Encoding categorical data such as DayOfMonth, MonthOfYear, Year
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_x = LabelEncoder()
x[:, 0] = labelencoder_x.fit_transform(x[:, 0])
x[:, 1] = labelencoder_x.fit_transform(x[:, 1])
x[:, 2] = labelencoder_x.fit_transform(x[:, 2])

# one hot encoding the DayOfMonth
onehotencoder = OneHotEncoder(categorical_features = [0])
x = onehotencoder.fit_transform(x).toarray()

# one hot encoding the MonthOfYear
onehotencoder = OneHotEncoder(categorical_features = [6])
x = onehotencoder.fit_transform(x).toarray()

# one hot encoding the Year
onehotencoder = OneHotEncoder(categorical_features = [18])
x = onehotencoder.fit_transform(x).toarray()

# Avoiding Dummy Variable Trap
x = x[:, [1,2,3,4,5,7,8,9,10,11,12,13,14,15,16,17,19,20,21,22]]

# Splitting the dataset into training set and test set
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)