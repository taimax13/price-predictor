import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn import metrics
from sklearn import svm
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression

from dataProcessor import DataProcessor

# URL of the dataset
url = 'https://drive.google.com/file/d/1kqnB4J8FuF1k8xLIvfbPE1jsqUwd3wVH/view?usp=sharing'

# Initialize and use DataProcessor
data_processor = DataProcessor(url)
dataset = data_processor.load_data()
dataset.head(5).to_csv('first_5_rows.csv', index=False)
data_processor.analyze_dataset()

df_final = data_processor.clean_data()
X_train, X_valid, Y_train, Y_valid = data_processor.split_data()

# Model evaluation function
score_func = lambda ytrue, ypred: np.sqrt(metrics.mean_squared_error(ytrue, ypred))

# Support Vector Regression
model_SVR = svm.SVR()  # Initialize model
model_SVR.fit(X_train, Y_train)
Y_pred = model_SVR.predict(X_valid)
print("SVR RMSE:", score_func(Y_valid, Y_pred))

# Random Forest Regressor
model_RFR = RandomForestRegressor(n_estimators=10)  # Initialize model
model_RFR.fit(X_train, Y_train)
Y_pred = model_RFR.predict(X_valid)
print("RFR RMSE:", score_func(Y_valid, Y_pred))

# Linear Regression
model_LR = LinearRegression()  # Initialize model
model_LR.fit(X_train, Y_train)
Y_pred = model_LR.predict(X_valid)
print("LR RMSE:", score_func(Y_valid, Y_pred))
