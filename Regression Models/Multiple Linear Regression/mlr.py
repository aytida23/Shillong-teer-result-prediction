"""
This is the Multiple Linear Regression implementation on the dataset for predicting F/R the result.
Unfortunately the prediction with backward elimination method is not upto the mark. Infact the 
dataset is 
"""

# Multiple Linear Regression

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

# Fitting Multiple Linear Regression on the dataset
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train, y_train)

# Predicting the Test set results
y_pred = regressor.predict(x_test)

# Building the optimal model using Backward Elimination
"""import statsmodels.formula.api as sm
x = np.append(arr = np.ones((1138, 1)).astype(int), values = x, axis = 1)
x_opt = x[:, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21]]
regressor_OLS = sm.OLS(endog = y, exog = x_opt).fit()
regressor_OLS.summary()

x_opt = x[:, [0, 1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21]]
regressor_OLS = sm.OLS(endog = y, exog = x_opt).fit()
regressor_OLS.summary()

x_opt = x[:, [0, 1, 2, 3, 4, 5, 6, 8, 9, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21]]
regressor_OLS = sm.OLS(endog = y, exog = x_opt).fit()
regressor_OLS.summary()

x_opt = x[:, [0, 1, 2, 3, 4, 5, 6, 8, 9, 11, 12, 13, 14, 15, 16, 17, 19, 20, 21]]
regressor_OLS = sm.OLS(endog = y, exog = x_opt).fit()
regressor_OLS.summary()

x_opt = x[:, [0, 1, 2, 3, 4, 5, 6, 9, 11, 12, 13, 14, 15, 16, 17, 19, 20, 21]]
regressor_OLS = sm.OLS(endog = y, exog = x_opt).fit()
regressor_OLS.summary()

x_opt = x[:, [0, 1, 2, 3, 4, 5, 6, 9, 12, 13, 15, 16, 17, 19, 20, 21]]
regressor_OLS = sm.OLS(endog = y, exog = x_opt).fit()
regressor_OLS.summary()

x_opt = x[:, [0, 1, 2, 3, 4, 5, 6, 9, 12, 13, 15, 16, 17, 19, 20, 21]]
regressor_OLS = sm.OLS(endog = y, exog = x_opt).fit()
regressor_OLS.summary()

x_opt = x[:, [0, 1, 2, 3, 4, 6, 9, 12, 13, 15, 16, 17, 19, 20, 21]]
regressor_OLS = sm.OLS(endog = y, exog = x_opt).fit()
regressor_OLS.summary()

x_opt = x[:, [0, 1, 2, 4, 6, 9, 12, 13, 15, 16, 17, 19, 20, 21]]
regressor_OLS = sm.OLS(endog = y, exog = x_opt).fit()
regressor_OLS.summary()

x_opt = x[:, [0, 1, 2, 4, 6, 9, 12, 13, 15, 16, 17, 20, 21]]
regressor_OLS = sm.OLS(endog = y, exog = x_opt).fit()
regressor_OLS.summary()

x_opt = x[:, [0, 1, 2, 4, 6, 9, 12, 13, 15, 16, 17, 20, 21]]
regressor_OLS = sm.OLS(endog = y, exog = x_opt).fit()
regressor_OLS.summary()

x_opt = x[:, [0, 1, 2, 4, 6, 9, 12, 13, 15, 16, 17, 21]]
regressor_OLS = sm.OLS(endog = y, exog = x_opt).fit()
regressor_OLS.summary()

x_opt = x[:, [0, 1, 2, 4, 9, 12, 13, 15, 16, 17, 21]]
regressor_OLS = sm.OLS(endog = y, exog = x_opt).fit()
regressor_OLS.summary()

x_opt = x[:, [0, 1, 2, 4, 9, 12, 13, 15, 16, 21]]
regressor_OLS = sm.OLS(endog = y, exog = x_opt).fit()
regressor_OLS.summary()

x_opt = x[:, [0, 1, 2, 4, 9, 12, 13, 16, 21]]
regressor_OLS = sm.OLS(endog = y, exog = x_opt).fit()
regressor_OLS.summary()

x_opt = x[:, [0, 1, 2, 4, 9, 12, 13, 16]]
regressor_OLS = sm.OLS(endog = y, exog = x_opt).fit()
regressor_OLS.summary()

x_opt = x[:, [0, 1, 2, 4, 9, 12, 13]]
regressor_OLS = sm.OLS(endog = y, exog = x_opt).fit()
regressor_OLS.summary()

x_opt = x[:, [0, 1, 2, 4, 9, 12]]
regressor_OLS = sm.OLS(endog = y, exog = x_opt).fit()
regressor_OLS.summary()

x_opt = x[:, [0, 1, 2, 4, 9]]
regressor_OLS = sm.OLS(endog = y, exog = x_opt).fit()
regressor_OLS.summary()

x_opt = x[:, [0, 1, 2, 4]]
regressor_OLS = sm.OLS(endog = y, exog = x_opt).fit()
regressor_OLS.summary()

x_opt = x[:, [0, 1, 4]]
regressor_OLS = sm.OLS(endog = y, exog = x_opt).fit()
regressor_OLS.summary()

# Splitting the final obtained table of optimal IV dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
x_opt_train, x_opt_test = train_test_split(x_opt, test_size = 0.2, random_state = 0)

# Fitting Multiple Linear Regression to the Optimal Training set
from sklearn.linear_model import LinearRegression
regressor_opt = LinearRegression()
regressor_opt.fit(x_opt_train, y_train)

# Predicting the optimal test set results that will have the strong impact on predicting profit
y_pred_opt = regressor_opt.predict(x_opt_test)"""