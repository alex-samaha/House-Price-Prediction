import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn.cross_validation import train_test_split

# Read in the CSV file
df = pd.read_csv("train.csv")
# Grab the specific columns we want to use
df = df[['LotArea','OverallQual','YearBuilt','YearRemodAdd','1stFlrSF','2ndFlrSF',
        'FullBath','HalfBath','BedroomAbvGr','SalePrice']]
# Increment the index
df.index += 1
# Take the first half
train = df[0:int(len(df)/2)]
# Put the sales prices into a separate dataframe
df_prices = df[['SalePrice']]
# Drop SalePrice and take second half as test
df = df.drop('SalePrice', 1)
test = df[int(len(df)/2):]

# Check first half
print(train.describe())

# Check second half
print(test.describe())

# Apply regression to the data
reg = linear_model.LinearRegression()

x_train, x_test, y_train, y_test = train_test_split(df, df_prices, random_state=4)

# fits the linear model
reg.fit(x_train, y_train)

# shows the optimal weights for each coefficient
print(reg.coef_)

# predict the prices of each house
predicted_prices = reg.predict(x_test)

# print out the first 10 predicted prices
for x in range(10):
    print(predicted_prices[x])
    

# print out the first 10 actual prices
count = 0;
for index, row in df_prices.iterrows():
    if count == 10:
        break
    print(row['SalePrice'])
    count += 1

