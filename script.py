# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from sklearn import linear_model
import matplotlib.pyplot as plt  
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

data_train = pd.read_csv('Traffic_Volume.csv')
data_train.head()
data_train.columns

##Find the correlation between the time and traffic volume

##Split date_time column "2012-10-14 13:00:00" into date and time columns
new = data_train["date_time"].str.split(" ", n = 1, expand = True) 
data_train["time"] = new[1]
data_train["date"] = new[0]

#data_train.date

##Get the hours of the day
data_train["time"] = data_train["time"].str.split(":", n = 1, expand = True)
#data_train.time

f1 = data_train['time'].values
f2 = data_train['traffic_volume'].values
#plt.scatter(f1, f2)

dataX =data_train[["time"]]
#dataX
X_train = np.array(dataX).astype(str).astype(int)
y_train = data_train['traffic_volume'].values

regr = linear_model.LinearRegression()
regr.fit(X_train, y_train)

y_pred = regr.predict(X_train)

print('Coefficients: \n', regr.coef_)
print('Independent term: \n', regr.intercept_)
print("Mean squared error: %.2f" % mean_squared_error(y_train, y_pred))
print('Variance score: %.2f' % r2_score(y_train, y_pred))

#Predict traffic volume at 17:00
y_area = regr.predict([[17]])
print(int(y_area))

plt.scatter(f1, f2)
plt.plot(X_train, y_pred, color='red')
plt.show()