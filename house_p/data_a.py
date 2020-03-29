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

plt.show()

# data_train.OverallCond, OverallQual, MSSubClass, MSZoning, LotConfig, LandSlope,
# BldgType, HouseStyle, YearBuilt, YearRemodAdd, RoofStyle, RoofMatl, MasVnrArea, Foundation
# Heating, GarageQual, PoolQC, SaleType, SaleCondition, YrSold