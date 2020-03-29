import pandas as pd
import numpy as np
from pandas import Series, DataFrame
import matplotlib.pyplot as plt
from sklearn import linear_model

data_train = pd.read_csv("~/machine_l/database/House_Prices/train.csv")

# print(data_train.head(3))
# print(data_train.MSSubClass.values)

fig = plt.figure()
fig.set(alpha=0.2)

# plt.subplot2grid((2,3), (0,0))
# data_train.MSSubClass.value_counts().plot(kind='bar')
# plt.title(u"MSSubClass quantity")
# plt.ylabel(u"people quantity")

# data_train.SalePrice.astype(float)
# data_train.OverallCond.astype(float)
# data_train.SalePrice[data_train.OverallCond == 1].plot(kind='line')
# data_train.SalePrice[data_train.OverallCond == 2].plot(kind='line')
# data_train.SalePrice[data_train.OverallCond == 3].plot(kind='line')
# data_train.SalePrice[data_train.OverallCond == 4].plot(kind='line')
# data_train.SalePrice[data_train.OverallCond == 5].plot(kind='line')
# data_train.SalePrice[data_train.OverallCond == 6].plot(kind='line')
# data_train.SalePrice[data_train.OverallCond == 7].plot(kind='line')
# data_train.SalePrice[data_train.OverallCond == 8].plot(kind='line')
# data_train.SalePrice[data_train.OverallCond == 9].plot(kind='line')
# data_train.SalePrice[data_train.OverallCond == 10].plot(kind='line')
# plt.xlabel(u"salePrice")
# plt.ylabel(u"density")
# plt.title(u"condition price distribution")
# plt.legend((u"very poor", u"poor", u"fair", u"below average", u"average", u"above average", \
#     u"good", u"very good", u"excellent", u"very excellent"), loc='best')

plt.subplot2grid((3,4), (0,0))
plt.scatter(data_train.OverallCond, data_train.SalePrice)
plt.grid(b=True, which='major', axis='y')
plt.title(u"condition price distr")
plt.ylabel(u"price")

plt.subplot2grid((3,4), (0,1))
plt.scatter(data_train.OverallQual, data_train.SalePrice)
plt.grid(b=True, which='major', axis='y')
plt.title(u"quality price distr")
plt.ylabel(u"price")

plt.subplot2grid((3,4), (0,2))
plt.scatter(data_train.MSSubClass, data_train.SalePrice)
plt.grid(b=True, which='major', axis='y')
plt.title(u"MSSub price distr")
plt.ylabel(u"price")

plt.subplot2grid((3,4), (0,3))
plt.scatter(data_train.MSZoning, data_train.SalePrice)
plt.grid(b=True, which='major', axis='y')
plt.title(u"MSZoning price distr")
plt.ylabel(u"price")

plt.subplot2grid((3,4), (1,0))
plt.scatter(data_train.LotShape, data_train.SalePrice)
plt.grid(b=True, which='major', axis='y')
plt.title(u"LotShape price distr")
plt.ylabel(u"price")

plt.subplot2grid((3,4), (1,1))
plt.scatter(data_train.LandContour, data_train.SalePrice)
plt.grid(b=True, which='major', axis='y')
plt.title(u"LandContour price distr")
plt.ylabel(u"price")

plt.subplot2grid((3,4), (1,2))
plt.scatter(data_train.LandSlope, data_train.SalePrice)
plt.grid(b=True, which='major', axis='y')
plt.title(u"LandSlope price distr")
plt.ylabel(u"price")

plt.subplot2grid((3,4), (1,3))
plt.scatter(data_train.BldgType, data_train.SalePrice)
plt.grid(b=True, which='major', axis='y')
plt.title(u"BldgType price distr")
plt.ylabel(u"price")

plt.subplot2grid((3,4), (2,0))
plt.scatter(data_train.HouseStyle, data_train.SalePrice)
plt.grid(b=True, which='major', axis='y')
plt.title(u"HouseStyle price distr")
plt.ylabel(u"price")

plt.subplot2grid((3,4), (2,1))
plt.scatter(data_train.SaleCondition, data_train.SalePrice)
plt.grid(b=True, which='major', axis='y')
plt.title(u"RoofMatl price distr")
plt.ylabel(u"price")

# plt.show()

# data_train.OverallCond, OverallQual, MSSubClass, MSZoning, LotConfig, LandSlope,
# BldgType, HouseStyle, YearBuilt, YearRemodAdd, RoofStyle, RoofMatl, MasVnrArea, Foundation
# Heating, GarageQual, PoolQC, SaleType, SaleCondition, YrSold

MSSubClass_MAX = max(data_train.MSSubClass) 
# print(MSSubClass_MAX)

MSSubClass_scale = data_train.MSSubClass / MSSubClass_MAX
OverallCond_scale = data_train.OverallCond / 10
OverallQual_scale = data_train.OverallQual / 10
YearBuilt_scale = data_train.YearBuilt / max(data_train.YearBuilt)
YearRemodAdd_scale = data_train.YearRemodAdd / max(data_train.YearRemodAdd)
MasVnrArea_scale = data_train.MasVnrArea / max(data_train.MasVnrArea)
YrSold_scale = data_train.YrSold / max(data_train.YrSold)


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

df = data_train.SalePrice
df = pd.concat([df, MSSubClass_scale, OverallCond_scale, OverallQual_scale, YearBuilt_scale, \
    YearRemodAdd_scale, MasVnrArea_scale, YrSold_scale, dummies_MSZoning, dummies_LotConfig,\
    dummies_LandSlope, dummies_BldgType, dummies_HouseStyle, dummies_RoofStyle,\
    dummies_RoofMatl, dummies_Foundation, dummies_Heating, dummies_GarageQual, \
    dummies_PoolQC, dummies_SaleType, dummies_SaleCondition], axis = 1)


train_np = df.values
# train_np = np.array(df)
print(df)
print(train_np)
print (np.shape(train_np))

y = train_np[:,0]
X = train_np[:, 1:]
nn = np.isnan(X)
print (np.shape(np.isnan(X)))
clf = linear_model.LogisticRegression(C=1.0, penalty = 'l2', tol = 1e-6)
clf.fit(X, y)
print(clf)









































































































