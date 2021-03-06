import pandas as pd
import numpy as np
from pandas import Series, DataFrame
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.datasets import make_regression
from sklearn.linear_model import ElasticNet
import sklearn.preprocessing as preprocessing
from sklearn.ensemble import RandomForestRegressor

data_train = pd.read_csv("~/machine_l/database/House_Prices/train.csv")

data_train.info()
# data_train.OverallCond, OverallQual, MSSubClass, MSZoning, LotConfig, LandSlope,
# BldgType, HouseStyle, YearBuilt, YearRemodAdd, RoofStyle, RoofMatl, MasVnrArea, Foundation
# Heating, GarageQual, PoolQC, SaleType, SaleCondition, YrSold


def standardlize_feather(df):

    scaler = preprocessing.StandardScaler()
    df_std = df.values.reshape(-1,1)
    # age_scale_param = scaler.fit(Age_2)
    df_scaled = scaler.fit_transform(df_std)
    return df_scaled
MSSubClass_scale = standardlize_feather(data_train.MSSubClass)
print ('------------------------TEST-----------------------------------')
print (MSSubClass_scale)

dummies_MSZoning = pd.get_dummies(data_train['MSZoning'], prefix='MSZoning')
dummies_LotConfig = pd.get_dummies(data_train['LotConfig'], prefix='LotConfig')
dummies_LandSlope = pd.get_dummies(data_train['LandSlope'], prefix='LandSlope')
dummies_BldgType = pd.get_dummies(data_train['BldgType'], prefix='BldgType')
dummies_HouseStyle = pd.get_dummies(data_train['HouseStyle'], prefix='HouseStyle')
dummies_RoofStyle = pd.get_dummies(data_train['RoofStyle'], prefix='RoofStyle')
dummies_RoofMatl = pd.get_dummies(data_train['RoofMatl'], prefix='RoofMatl')
dummies_Foundation = pd.get_dummies(data_train['Foundation'], prefix='Foundation')
dummies_Heating = pd.get_dummies(data_train['Heating'], prefix='Heating')
dummies_GarageQual = pd.get_dummies(data_train['GarageQual'], prefix='GarageQual')
dummies_PoolQC = pd.get_dummies(data_train['PoolQC'], prefix='PoolQC')
dummies_SaleType = pd.get_dummies(data_train['SaleType'], prefix='SaleType')
dummies_SaleCondition = pd.get_dummies(data_train['SaleCondition'], prefix='SaleCondition')

def set_missing_values(df):
    
    # 把已有的数值型特征取出来丢进Random Forest Regressor中
    df_RFR = df
    df_RFR = pd.concat([df_RFR, MSSubClass_scale, OverallCond_scale, OverallQual_scale, YearBuilt_scale, \
    YearRemodAdd_scale, YrSold_scale, dummies_MSZoning, dummies_LotConfig,\
    dummies_LandSlope, dummies_BldgType, dummies_HouseStyle, dummies_RoofStyle,\
    dummies_RoofMatl, dummies_Foundation, dummies_Heating, dummies_GarageQual, \
    dummies_PoolQC, dummies_SaleType, dummies_SaleCondition], axis = 1)


    # 乘客分成已知年龄和未知年龄两部分
    known_df = df_RFR[df_RFR[0].notnull()].values
    unknown_df = df_RFR[df_RFR[0].isnull()].values
    
    # y即目标年龄
    y = known_df[:, 0]
    print('the data in y:', y)
    # X即特征属性值
    X = known_df[:, 1:]
    print('the data in X', X)

    # fit到RandomForestRegressor之中
    rfr = RandomForestRegressor(random_state=0, n_estimators=2000, n_jobs=-1)
    rfr.fit(X, y)
    
    # 用得到的模型进行未知年龄结果预测
    predictedValues = rfr.predict(unknown_df[:, 1:])
    df.loc[ (df_RFR[0].isnull()), df[0] ] = predictedValues 
    return df_RFR, rfr

MSSubClass_scale = standardlize_feather(data_train.MSSubClass)
print (MSSubClass_scale)
OverallCond_scale = standardlize_feather(data_train.OverallCond)
OverallQual_scale = standardlize_feather(data_train.OverallQual)
YearBuilt_scale = standardlize_feather(data_train.YearBuilt)
YearRemodAdd_scale = standardlize_feather(data_train.YearRemodAdd)
YrSold_scale = standardlize_feather(data_train.YrSold)


# MSSubClass_scale = data_train.MSSubClass / max(data_train.MSSubClass)
# OverallCond_scale = data_train.OverallCond / 10
# OverallQual_scale = data_train.OverallQual / 10
# YearBuilt_scale = data_train.YearBuilt / max(data_train.YearBuilt)
# YearRemodAdd_scale = data_train.YearRemodAdd / max(data_train.YearRemodAdd)
# YrSold_scale = data_train.YrSold / max(data_train.YrSold)


df_MasVnrArea, rfr = set_missing_values(data_train.MasVnrArea)

df = data_train.SalePrice
df = pd.concat([df, MSSubClass_scale, OverallCond_scale, OverallQual_scale, YearBuilt_scale, \
    YearRemodAdd_scale, YrSold_scale, dummies_MSZoning, dummies_LotConfig,\
    dummies_LandSlope, dummies_BldgType, dummies_HouseStyle, dummies_RoofStyle,\
    dummies_RoofMatl, dummies_Foundation, dummies_Heating, dummies_GarageQual, \
    dummies_PoolQC, dummies_SaleType, dummies_SaleCondition, df_MasVnrArea], axis = 1)

print('--------------TEST-----------------------')
df.info()
# print (df.isnull().sum().sort_values(ascending=False))

train_np = df.values
y = train_np[:,0]
X = train_np[:, 1:]

# X, y = make_regression(n_features=81, random_state=0)

clf = linear_model.LogisticRegression(C=1.0, penalty = 'l2', tol = 1e-6)
# clf = linear_model.LinearRegression().fit(X, y)
# clf = ElasticNet(random_state=0)

clf.fit(X, y)
print(clf)

data_test = pd.read_csv("~/machine_l/database/House_Prices/test.csv")
# data_test.info()

MSSubClass_MAX = max(data_test.MSSubClass) 
MSSubClass_scale = data_test.MSSubClass / MSSubClass_MAX
OverallCond_scale = data_test.OverallCond / 10
OverallQual_scale = data_test.OverallQual / 10
YearBuilt_scale = data_test.YearBuilt / max(data_test.YearBuilt)
YearRemodAdd_scale = data_test.YearRemodAdd / max(data_test.YearRemodAdd)
# MasVnrArea_scale = data_test.MasVnrArea / max(data_test.MasVnrArea)
YrSold_scale = data_test.YrSold / max(data_test.YrSold)


dummies_MSZoning = pd.get_dummies(data_test['MSZoning'], prefix='MSZoning')
dummies_LotConfig = pd.get_dummies(data_test['LotConfig'], prefix='LotConfig')
dummies_LandSlope = pd.get_dummies(data_test['LandSlope'], prefix='LandSlope')
dummies_BldgType = pd.get_dummies(data_test['BldgType'], prefix='BldgType')
dummies_HouseStyle = pd.get_dummies(data_test['HouseStyle'], prefix='HouseStyle')
dummies_RoofStyle = pd.get_dummies(data_test['RoofStyle'], prefix='RoofStyle')
dummies_RoofMatl = pd.get_dummies(data_test['RoofMatl'], prefix='RoofMatl')
dummies_Foundation = pd.get_dummies(data_test['Foundation'], prefix='Foundation')
dummies_Heating = pd.get_dummies(data_test['Heating'], prefix='Heating')
dummies_GarageQual = pd.get_dummies(data_test['GarageQual'], prefix='GarageQual')
dummies_PoolQC = pd.get_dummies(data_test['PoolQC'], prefix='PoolQC')
dummies_SaleType = pd.get_dummies(data_test['SaleType'], prefix='SaleType')
dummies_SaleCondition = pd.get_dummies(data_test['SaleCondition'], prefix='SaleCondition')

test_df = data_test.Id
test_df = pd.concat([test_df, MSSubClass_scale, OverallCond_scale, OverallQual_scale, YearBuilt_scale, \
    YearRemodAdd_scale, YrSold_scale, dummies_MSZoning, dummies_LotConfig,\
    dummies_LandSlope, dummies_BldgType, dummies_HouseStyle, dummies_RoofStyle,\
    dummies_RoofMatl, dummies_Foundation, dummies_Heating, dummies_GarageQual, \
    dummies_PoolQC, dummies_SaleType, dummies_SaleCondition], axis = 1)
test_df.drop(['Id'], axis=1, inplace=True)
df.drop(['SalePrice'], axis=1, inplace=True)

train_columns = df.columns.tolist()
test_columns = test_df.columns.tolist()

for i in range(len(train_columns)):
    if (train_columns[i] != test_columns[i]):
        print(train_columns[i])
        data = df.pop(train_columns[i])
        test_df.insert(i, train_columns[i], data)
        test_columns.insert(i, train_columns[i])
        print(test_columns[i+1])

print (len(test_columns))
# test_df.reindex(columns=test_columns, fill_value=0)

print (len(test_df.columns))
# col_name = test_df.columns.tolist()
# col_name.insert(1,'HouseStyle_2.5Fin')
# df.reindex(columns=col_name)
test_df.info()
# print (df.isnull().sum().sort_values(ascending=False))

predictions = clf.predict(test_df)
result = pd.DataFrame({'Id':data_test.Id.values, 'Prices':predictions.astype(np.int32)})
result.to_csv("~/machine_l/database/House_Prices/sale_price_predictions_log.csv", index=False)



    
























































































