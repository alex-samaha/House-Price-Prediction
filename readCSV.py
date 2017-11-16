"""
    Lukas Basse
    lukasbasse22@gmail.com
"""

import pandas as pd

def cleanCSV(fileHandle):
    # Read in the CSV file
    df = pd.read_csv(fileHandle)
    # Check that it worked
    print(df.head())
    # Grab the specific columns we want to use
    df = df[['LotArea','OverallQual','YearBuilt','YearRemodAdd','1stFlrSF','2ndFlrSF',
            'FullBath','HalfBath','BedroomAbvGr','SalePrice']]
    # Increment the index
    df.index += 1
    # Take the first half
    train = df[0:int(len(df)/2)]
    # Check first half
    print(train.head())
    # Drop SalePrice and take second half as test
    df = df.drop('SalePrice', 1)
    test = df[int(len(df)/2):]
    # Check second half
    print(test.head())
    # Save the data to a new csv
    train.to_csv("newTrain.csv")
    test.to_csv("newTest.csv")


cleanCSV("train.csv")
