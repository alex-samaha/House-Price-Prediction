"""
    Lukas Basse
    lukasbasse22@gmail.com
    NDL Housing Prices Prediction Modeling
"""

import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn.cross_validation import train_test_split
import matplotlib.pyplot as plt

df = pd.read_csv("train.csv")

# Grab the specific columns we want to use
df = df[['LotArea','OverallQual','YearBuilt','YearRemodAdd','1stFlrSF','2ndFlrSF',
        'FullBath','HalfBath','BedroomAbvGr','SalePrice']]

# Increment the index
df.index += 1

# Put the sales prices into a separate dataframe
df_prices = df[['SalePrice']]

# Get the sales prices for the test and train data
df_train_prices = df_prices[0:(len(df)//2)]
df_test_prices = df_prices[len(df)//2:]

# Drop the sales prices in the original dataframe
df = df.drop('SalePrice', 1)

# Take the first half
train = df[0:int(len(df)/2)]
test = df[int(len(df)/2):]

# Check first half
print(train.describe())

# Check second half
print(test.describe())

# Apply regression to the data
reg = linear_model.LinearRegression()

#x_train, x_test, y_train, y_test = train_test_split(train, df_train_prices, random_state=4)

# fits the linear model
#reg.fit(x_train, y_train)
reg.fit(train, df_train_prices)

# shows the optimal weights for each coefficient
reg.coef_

# predict the prices of each house
predicted_prices = reg.predict(test)

# print out the first 10 predicted prices
for x in range(10):
    print(predicted_prices[x])
    
print("Now the actual prices:")
    
# print out the first 10 actual prices
count = 0;
for index, row in df_test_prices.iterrows():
    if count == 10:
        break
    print(str(index) + ":" + str(row['SalePrice']))
    count += 1

predict_mean = np.mean(predicted_prices)
test_mean = np.mean(df_test_prices)

"""
print(predict_mean)
print(test_mean)

print("Standard deviation of predicted: " +str(np.std(predicted_prices)))
print("Standard deviation of test: " + str(np.std(test))

print(type(predicted_prices))
print(type(df_test_prices))
"""
# Create a new DataFrame for exporting
testData = test
#print(predicted_prices)
# Flatten the np array (remove internal dimensionality)
pred_prices = predicted_prices.flatten()
#print(pred_prices)
# Add the prices to the DataFrame
testData['predictedPrices'] = pred_prices
testData['actualPrices'] = df_test_prices
#print(testData.head)

# Export the data to a csv
testData.to_csv("prices_predicted.csv")

