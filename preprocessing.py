import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import mode
from grouping_features import *

def rmse(y_pred, y):
    return np.sqrt(np.mean(np.square(y_pred-y)))
def rmsle(y_pred, y):
    return np.sqrt(np.mean(np.square(np.log(y_pred+1) - np.log(y+1))))

def fill_missing(colx, value, train, test, val):
    train.loc[:, colx] = train.loc[:, colx].fillna(value)
    test.loc[:, colx] = test.loc[:, colx].fillna(value)
    val.loc[:, colx] = val.loc[:, colx].fillna(value)
def pre_process(train, val, test):
    ## Remove the repetitive features because they can cause problems in the fitting of gradient descent

    train = train.drop(['GarageArea', 'GarageYrBlt', 'TotalBsmtSF', 'TotRmsAbvGrd'], axis = 1)

    #Create new features, more logical features.
    train['Age'] = max(train['YearBuilt']) - train['YearBuilt']
    train['AgeMod'] = max(train['YearRemodAdd']) - train['YearRemodAdd']
    train['BsmtFinSF'] = train['BsmtFinSF1'] + train['BsmtFinSF2']
    train['TotArea'] = train['BsmtFinSF'] + train['BsmtUnfSF'] + train['1stFlrSF'] + train['2ndFlrSF'] + train['GrLivArea']

    # Manually fit the order of some features to make a straight line with the target value
    train['Age'] = train['Age']**0.5
    train['AgeMod']  = train['AgeMod']**0.5

    train = train.drop(['YearRemodAdd', 'YearBuilt'], axis = 1)

    train = train.drop(['BsmtFinSF1', 'BsmtFinSF2', 'MoSold', 'YrSold', 'MiscVal'], axis=1)

    # drop irrelevant categorical features

    fill_missing('LotFrontage', mode(train['LotFrontage'])[0][0], train, val, test)
    fill_missing('MasVnrArea', mode(train['MasVnrArea'])[0][0], train, val, test)

    train['PoolQC'] = train['PoolQC'].replace(to_replace = ['Ex', 'Fa', 'Gd'], value = [2,3,1])
    train['PoolQC'] = train['PoolQC'].fillna(0)
    train = train.drop(['PoolQC'], axis = 1)


    train['MiscFeature'] = train['MiscFeature'].fillna('Na')

    train['Alley'] = train['Alley'].fillna('Na')

    train['Fence'] = train['Fence'].fillna('Na')

    train['FireplaceQu'] = train['FireplaceQu'].fillna('Na')

    train['FireplaceQu'] = train['FireplaceQu'].replace(to_replace = ['Ex', 'Gd', 'TA','Fa', 'Po', 'Na' ], value = [5,4,3,2,1,0])

    train['GarageCond'] = train['GarageCond'].fillna('Na')

    train['GarageFinish'] = train['GarageFinish'].fillna('Na')

    for col in ['GarageQual', 'GarageCond']:
        train[col] = train[col].replace(to_replace = ['Ex', 'Gd', 'TA','Fa', 'Po', 'Na' ], value = [5,4,3,2,1,0])

    train['GarageFinish'] = train['GarageFinish'].replace(to_replace = ['Na', 'Unf', 'RFn', 'Fin'], value = [0,1,2,3])

    train['GarageType'] = train['GarageType'].fillna('Na')

    train['BsmtExposure'] = train['BsmtExposure'].fillna('Na')

    train['BsmtExposure'] = train['BsmtExposure'].replace(to_replace = ['Na', 'No', 'Mn', 'Av', 'Gd'], value = [0,1,2,3,4])

    train['BsmtFinType2'] = train['BsmtFinType2'].fillna('Na')

    train['BsmtFinType2'] = train['BsmtFinType2'].replace(to_replace= ['ALQ', 'GLQ', 'BLQ', 'LwQ', 'Rec', 'Unf', 'Na'], value = [1,1,1,1,1,1,0])

    train['BsmtFinType1'] = train['BsmtFinType1'].fillna('Na')

    train['BsmtQual'] = train['BsmtQual'].fillna('Na')
    train['BsmtCond'] = train['BsmtCond'].fillna('Na')

    for col in ['BsmtQual', 'BsmtCond']:
        train[col] = train[col].replace(to_replace = ['Ex', 'Gd', 'TA','Fa', 'Po', 'Na' ], value = [5,4,3,2,1,0])

    train['MasVnrType'] = train['MasVnrType'].fillna('Na')

    train = train.fillna(train['Electrical'].value_counts().index[0])



    train = train.drop(['Condition2'],axis=1)

    train = train.drop(['RoofMatl'],axis=1)

    train = train.drop(['BldgType'],axis=1)

    train = train.drop(['LotShape', 'Utilities', 'Exterior1st', 'Exterior2nd'],axis=1)


    train['HeatingQC'] = train['HeatingQC'].replace(to_replace = ['Ex', 'Gd', 'TA','Fa', 'Po', 'Na' ], value = [5,4,3,2,1,0])

    train['KitchenQual'] = train['KitchenQual'].replace(to_replace = ['Ex', 'Gd', 'TA','Fa', 'Po', 'Na' ], value = [5,4,3,2,1,0])

    train['Functional'] = train['Functional'].replace(to_replace = ['Typ', 'Min2', 'Mod','Maj1', 'Maj2' ], value = [5,4,3,2,1])
    
    return train

def group(train, val, test):
    groups = grouping('Neighborhood', train, 10000)
    val['Neighborhood']  = assign_grouping('Neighborhood', val, groups)
    test['Neighborhood'] = assign_grouping('Neighborhood', test, groups)

    groups = grouping('Functional', train, 30000)
    assign_grouping('Functional', val, groups)
    assign_grouping('Functional', test, groups)

    groups = grouping('PavedDrive', train, 30000)
    assign_grouping('PavedDrive', val, groups)
    assign_grouping('PavedDrive', test, groups)

    val['Fence'] = val['Fence'].fillna('Na')
    test['Fence'] = test['Fence'].fillna('Na')
    groups = grouping('Fence', train, 10000)
    assign_grouping('Fence', val, groups)
    assign_grouping('Fence', test, groups)

    test['SaleType'] = test['SaleType'].fillna('Na')
    groups = grouping('SaleType', train, 10000)
    assign_grouping('SaleType', val, groups)
    assign_grouping('SaleType', test, groups)

    groups = grouping('SaleCondition', train, 20000)
    assign_grouping('SaleCondition', val, groups)
    assign_grouping('SaleCondition', test, groups)
    
    return train, val, test
