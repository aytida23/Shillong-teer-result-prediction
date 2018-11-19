"""
This is the implementation of polynomial regression on the dataset of shillong teer. Prediction of S/R with following
Independent Variables available.
"""

# Polynomial Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

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
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.1, random_state = 5)

# Fitting linear regression to the dataset
from sklearn.linear_model import LinearRegression
# Fitting polynomial regression to the dataset
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 5)
x_poly = poly_reg.fit_transform(x)
lin_reg_2 = LinearRegression()
lin_reg_2.fit(x_poly, y)

# predicting the test set with polynomial regression model
y_pred_poly = lin_reg_2.predict(poly_reg.fit_transform(x_test))

# Visualising the Test set results
plt.figure(figsize = (6,6))
plt.scatter(x_test[:,[20]], y_test, color='red')
plt.plot(x_test[:,[20]], y_pred_poly, color='purple')
plt.title('Prediction of S/R (Test set)')
plt.xlabel('F/R')
plt.ylabel('S/R')
plt.grid()
plt.show()

#getting r-squared value
from sklearn import metrics
print('R-Squared: ', metrics.explained_variance_score(y_test,y_pred_poly))

#printing our errors
print('MSE:',metrics.mean_squared_error(y_test,y_pred_poly))
print('MAE:',metrics.mean_absolute_error(y_test,y_pred_poly))
print('RMSE:',np.sqrt(metrics.mean_squared_error(y_test,y_pred_poly)))