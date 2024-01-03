import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import numpy as np


# Load Data 

df = pd.read_csv('https://raw.githubusercontent.com/dataprofessor/data/master/delaney_solubility_with_descriptors.csv')


# # Data Preparation 

# y = df["logS"]

# x = df.drop("logS", axis=1)


# # Data Splitting

# x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=100)


# # Model Building 


# ## Linear Regression

# ### Training the model

# lr = LinearRegression()
# lr.fit(x_train, y_train)


# ### Applying the model to make a prediction

# y_lr_train_pred = lr.predict(x_train)
# y_lr_test_pred = lr.predict(x_test)


# ### Evaluate Model Performace

# lr_train_mse = mean_squared_error(y_train, y_lr_train_pred)
# lr_train_r2 = r2_score(y_train, y_lr_train_pred)

# lr_test_mse = mean_squared_error(y_test, y_lr_test_pred)
# lr_test_r2 = r2_score(y_test, y_lr_test_pred)

# print("LR MSE (Train): ", lr_train_mse)
# print("LR R2 (Train): ", lr_train_r2)
# print("LR MSE (Test): ", lr_test_mse)
# print("LR R2 (Test): ", lr_test_r2)
# print()

# lr_results = pd.DataFrame(["Linear Regression", lr_train_mse, lr_train_r2, lr_test_mse, lr_test_r2]).transpose()
# lr_results.columns = ['Method', 'Training MSE', 'Training R2', 'Test MSE', 'Test R2']
# print(lr_results)
# print()