# Code you have previously used to load data
import pandas as pd
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import numpy as np
from plots import *

# Set up code checking
# if not os.path.exists("../input/train.csv"):
#     os.symlink("../input/home-data-for-ml-course/train.csv", "../input/train.csv")
#     os.symlink("../input/home-data-for-ml-course/test.csv", "../input/test.csv")
# from learntools.core import binder
# binder.bind(globals())
# from learntools.machine_learning.ex7 import *

if os.name == 'nt':
    iowa_file_path =r"C:\Users\pedfernandez\PycharmProjects\kaggle\housing_prices_learning_comp\data\train.csv"
else:
    iowa_file_path = './data/train.csv'

home_data = pd.read_csv(iowa_file_path)

# Create target object and call it y
y = home_data.SalePrice
# Create X
features = ['LotArea'] #, 'YearBuilt', '1stFlrSF', '2ndFlrSF', 'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd'
# we add the categorical features
cat_features = ['Neighborhood']
# We store the numerical columns and will add the categorical variables transformed
df = home_data[features].copy()

#For each categorical feature there will be a transformed column.
# Names of transformed columns are the same starting with "T".
cat_Tfeatures = []
for cf in cat_features:
    cat_Tfeatures.append('T'+cf)

le = LabelEncoder()
# print(list(le.classes_))

# Create a new column with the transformed categorical data
for co, cf in zip(cat_Tfeatures, cat_features):
    le.fit(home_data[cf])
    df[co] = list(le.transform(home_data.loc[:, cf]))
# print(list(le.inverse_transform(list(home_data['TNeighborhood']))))
# print(list(home_data['Neighborhood']))

X = df

# Split into validation and training data
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)

# Let us plot a learning curve with the test error and validation error
# Define the model. Set random_state to 1
rf_model = RandomForestRegressor(n_estimators=100, random_state=1)
trEr = []
vaEr = []
szls = []
Nd = len(train_X)
tt = 1
for i in range(Nd):
    inpX = train_X.sample(n=i+1)
    inpy = train_y.sample(n=i+1)
    #Fit the model using a sample of size "i" randomly selected from train data
    rf_model.fit(inpX, inpy)
    #Compute the training error
    rf_tra_predictions = rf_model.predict(inpX)
    rf_tra_mae = mean_absolute_error(rf_tra_predictions, inpy)
    #Compute the validation error
    rf_val_predictions = rf_model.predict(val_X)
    rf_val_mae = mean_absolute_error(rf_val_predictions, val_y)
    #store the errors in lists
    trEr.append(rf_tra_mae)
    vaEr.append(rf_val_mae)
    szls.append(i+1)
    print("==========")
    print(i, " of ", Nd, " Training MAE for Random Forest Model: {:,.0f}".format(rf_tra_mae))
    print(i, " of ", Nd, " Validation MAE for Random Forest Model: {:,.0f}".format(rf_val_mae))
    if i > tt*100:
        tt = tt +1
        scatPlotTwoLists(szls, trEr,
                         szls, vaEr,
                         'size of train set', 'error', 'learning curve using LotArea and Neighborhood',
                         'lc_01.png', 100, True)
