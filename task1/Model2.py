import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn import metrics

data = pd.read_csv('house_data.csv')
X1=data['bedrooms']
X2=data['bathrooms']
X3=data['sqft_living']
X4=data['sqft_lot']

Y=data['price']

cls = linear_model.LinearRegression()
X=np.expand_dims(X2, axis=1)
Y=np.expand_dims(Y, axis=1)
cls.fit(X,Y)
prediction= cls.predict(X)
plt.scatter(X, Y)
plt.xlabel('bathrooms', fontsize = 20)
plt.ylabel('price', fontsize = 20)
plt.plot(X, prediction, color='red', linewidth = 3)
plt.show()
print('bathrooms as X')
print('Co-efficient of linear regression',cls.coef_)
print('Intercept of linear regression model',cls.intercept_)
print('Mean Square Error', metrics.mean_squared_error(Y, prediction))

