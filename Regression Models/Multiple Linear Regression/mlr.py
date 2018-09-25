"""
This is the Multiple Linear Regression implementation on the dataset for predicting S/R i.e the 
result. Unfortunately the prediction with backward elimination method is not doing actually what 
we want. Infact the dataset is not suitable for prediction via multiple linear regression model to predict
better result of S/R with the following independent variables in the dataset.
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
x = dataset.iloc[:, [1,4,5,6]].values
y = dataset.iloc[:, [2]].values

# Encoding categorical data such as DayOfMonth, MonthOfYear, Year
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_x = LabelEncoder()
x[:, 1] = labelencoder_x.fit_transform(x[:, 1])
x[:, 2] = labelencoder_x.fit_transform(x[:, 2])
x[:, 3] = labelencoder_x.fit_transform(x[:, 3])

# one hot encoding the DayOfMonth
onehotencoder = OneHotEncoder(categorical_features = [1])
x = onehotencoder.fit_transform(x).toarray()

# one hot encoding the MonthOfYear
onehotencoder = OneHotEncoder(categorical_features = [7])
x = onehotencoder.fit_transform(x).toarray()

# one hot encoding the Year
onehotencoder = OneHotEncoder(categorical_features = [19])
x = onehotencoder.fit_transform(x).toarray()

# Avoiding Dummy Variable Trap
x = x[:, [1,2,3,4,5,7,8,9,10,11,12,13,14,15,16,17,19,20,21,22,23]]

# Splitting the dataset into training set and test set
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)

# Fitting Multiple Linear Regression on the dataset
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train, y_train)

# Predicting the Test set results
y_pred = regressor.predict(x_test)

#visualising the test set without backward elimination method
plt.scatter(x_test[:,[20]], y_test, color='red')
plt.plot(x_test[:,[20]], regressor.predict(x_test), color='green')
plt.title('F/R vs S/R (Without Backward Elimination Method)')
plt.xlable('F/R')
plt.ylabel('S/R')
plt.show()

# Building the optimal model using Backward Elimination
import statsmodels.formula.api as sm
x = np.append(arr = np.ones((1138, 1)).astype(int), values = x, axis = 1)
x_opt = x[:, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21]]
regressor_OLS = sm.OLS(endog = y, exog = x_opt).fit()
regressor_OLS.summary()

# removin' index 14 predictor from the original x matrix because of highest p value for teamin' up x_opt with optimal predictors
x_opt = x[:, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 15, 16, 17, 18, 19, 20, 21]]
regressor_OLS = sm.OLS(endog = y, exog = x_opt).fit()
regressor_OLS.summary()

# removin' index 11 predictor from the original x matrix because of highest p value for teamin' up x_opt with optimal predictors
x_opt = x[:, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 13, 15, 16, 17, 18, 19, 20, 21]]
regressor_OLS = sm.OLS(endog = y, exog = x_opt).fit()
regressor_OLS.summary()

# removin' index 6 predictor from the original x matrix because of highest p value for teamin' up x_opt with optimal predictors
x_opt = x[:, [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 12, 13, 15, 16, 17, 18, 19, 20, 21]]
regressor_OLS = sm.OLS(endog = y, exog = x_opt).fit()
regressor_OLS.summary()

# removin' index 3 predictor from the original x matrix because of highest p value for teamin' up x_opt with optimal predictors
x_opt = x[:, [0, 1, 2, 4, 5, 7, 8, 9, 10, 12, 13, 15, 16, 17, 18, 19, 20, 21]]
regressor_OLS = sm.OLS(endog = y, exog = x_opt).fit()
regressor_OLS.summary()

# removin' index 12 predictor from the original x matrix because of highest p value for teamin' up x_opt with optimal predictors
x_opt = x[:, [0, 1, 2, 4, 5, 7, 8, 9, 10, 13, 15, 16, 17, 18, 19, 20, 21]]
regressor_OLS = sm.OLS(endog = y, exog = x_opt).fit()
regressor_OLS.summary()

# removin' index 15 predictor from the original x matrix because of highest p value for teamin' up x_opt with optimal predictors
x_opt = x[:, [0, 1, 2, 4, 5, 7, 8, 9, 10, 13, 16, 17, 18, 19, 20, 21]]
regressor_OLS = sm.OLS(endog = y, exog = x_opt).fit()
regressor_OLS.summary()

# removin' index 13 predictor from the original x matrix because of highest p value for teamin' up x_opt with optimal predictors
x_opt = x[:, [0, 1, 2, 4, 5, 7, 8, 9, 10, 16, 17, 18, 19, 20, 21]]
regressor_OLS = sm.OLS(endog = y, exog = x_opt).fit()
regressor_OLS.summary()

# removin' index 9 predictor from the original x matrix because of highest p value for teamin' up x_opt with optimal predictors
x_opt = x[:, [0, 1, 2, 4, 5, 7, 8, 10, 16, 17, 18, 19, 20, 21]]
regressor_OLS = sm.OLS(endog = y, exog = x_opt).fit()
regressor_OLS.summary()

# removin' index 1 predictor from the original x matrix because of highest p value for teamin' up x_opt with optimal predictors
x_opt = x[:, [0, 2, 4, 5, 7, 8, 10, 16, 17, 18, 19, 20, 21]]
regressor_OLS = sm.OLS(endog = y, exog = x_opt).fit()
regressor_OLS.summary()

# removin' index 20 predictor from the original x matrix because of highest p value for teamin' up x_opt with optimal predictors
x_opt = x[:, [0, 2, 4, 5, 7, 8, 10, 16, 17, 18, 19, 21]]
regressor_OLS = sm.OLS(endog = y, exog = x_opt).fit()
regressor_OLS.summary()

# removin' index 16 predictor from the original x matrix because of highest p value for teamin' up x_opt with optimal predictors
x_opt = x[:, [0, 2, 4, 5, 7, 8, 10, 17, 18, 19, 21]]
regressor_OLS = sm.OLS(endog = y, exog = x_opt).fit()
regressor_OLS.summary()

# removin' index 17 predictor from the original x matrix because of highest p value for teamin' up x_opt with optimal predictors
x_opt = x[:, [0, 2, 4, 5, 7, 8, 10, 18, 19, 21]]
regressor_OLS = sm.OLS(endog = y, exog = x_opt).fit()
regressor_OLS.summary()

# removin' index 19 predictor from the original x matrix because of highest p value for teamin' up x_opt with optimal predictors
x_opt = x[:, [0, 2, 4, 5, 7, 8, 10, 18, 21]]
regressor_OLS = sm.OLS(endog = y, exog = x_opt).fit()
regressor_OLS.summary()

# removin' index 10 predictor from the original x matrix because of highest p value for teamin' up x_opt with optimal predictors
x_opt = x[:, [0, 2, 4, 5, 7, 8, 18, 21]]
regressor_OLS = sm.OLS(endog = y, exog = x_opt).fit()
regressor_OLS.summary()

# removin' index 8 predictor from the original x matrix because of highest p value for teamin' up x_opt with optimal predictors
x_opt = x[:, [0, 2, 4, 5, 7, 18, 21]]
regressor_OLS = sm.OLS(endog = y, exog = x_opt).fit()
regressor_OLS.summary()

# removin' index 2 predictor from the original x matrix because of highest p value for teamin' up x_opt with optimal predictors
x_opt = x[:, [0, 4, 5, 7, 18, 21]]
regressor_OLS = sm.OLS(endog = y, exog = x_opt).fit()
regressor_OLS.summary()

# removin' index 18 predictor from the original x matrix because of highest p value for teamin' up x_opt with optimal predictors
x_opt = x[:, [0, 4, 5, 7, 21]]
regressor_OLS = sm.OLS(endog = y, exog = x_opt).fit()
regressor_OLS.summary()

# removin' index 5 predictor from the original x matrix because of highest p value for teamin' up x_opt with optimal predictors
x_opt = x[:, [0, 4, 7, 21]]
regressor_OLS = sm.OLS(endog = y, exog = x_opt).fit()
regressor_OLS.summary()

# removin' index 7 predictor from the original x matrix because of highest p value for teamin' up x_opt with optimal predictors
x_opt = x[:, [0, 4, 21]]
regressor_OLS = sm.OLS(endog = y, exog = x_opt).fit()
regressor_OLS.summary()

# removin' index 4 predictor from the original x matrix because of highest p value for teamin' up x_opt with optimal predictors
# Here we have got our optimal team of predictor by using backward elimination method
x_opt = x[:, [0, 21]]
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
y_pred_opt = regressor_opt.predict(x_opt_test)

#visualising the test set with backward elimination method
plt.scatter(x_opt_test[:,[1]], y_test, color='red')
plt.plot(x_opt_test[:,[1]], y_pred_opt, color='green')
plt.title('F/R vs S/R (With Backward Elimination Method)')
plt.xlabel('F/R')
plt.ylabel('S/R')
plt.show()