import pandas as pd
import numpy as np
from pandas import Series, DataFrame
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm
from sklearn.preprocessing import StandardScaler
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

data_train = pd.read_csv("~/machine_l/database/House_Prices/train.csv")

# data_train.OverallCond, OverallQual, MSSubClass, MSZoning, LotConfig, LandSlope,
# BldgType, HouseStyle, YearBuilt, YearRemodAdd, RoofStyle, RoofMatl, MasVnrArea, Foundation
# Heating, GarageQual, PoolQC, SaleType, SaleCondition, YrSold

data_train['SalePrice'].describe()
sns.distplot(data_train['SalePrice'])
# plt.show()

#correlation matrix
corrmat = data_train.corr()
f, ax = plt.subplots(figsize=(12, 9))
sns.heatmap(corrmat, vmax=.8, square=True)
# plt.show()


#saleprice correlation matrix
k = 10 #number of variables for heatmap
cols = corrmat.nlargest(k, 'SalePrice')['SalePrice'].index
cm = np.corrcoef(data_train[cols].values.T)
sns.set(font_scale=1.25)
hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', \
    annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)
# plt.show()


#scatterplot
# sns.set()
# cols = ['SalePrice', 'OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'FullBath', 'YearBuilt']
# sns.pairplot(data_train[cols], size = 2.5)
# plt.show()

total = data_train.isnull().sum().sort_values(ascending=False)
percent = (data_train.isnull().sum()/data_train.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
print (missing_data.head(20))

#dealing with missing data

print ('------------------------------TEST-----------------------------------')

data_train = data_train.drop((missing_data[missing_data['Total'] > 1]).index,1)
data_train = data_train.drop(data_train.loc[data_train['Electrical'].isnull()].index)
data_train.isnull().sum().max() #just checking that there's no missing data missing...


#standardizing data
# print (data_train['SalePrice'].values)
# print (data_train['SalePrice'][:,np.newaxis])
saleprice_scaled = StandardScaler().fit_transform(data_train['SalePrice'][:,np.newaxis])
low_range = saleprice_scaled[saleprice_scaled[:,0].argsort()][:10]
high_range= saleprice_scaled[saleprice_scaled[:,0].argsort()][-10:]
print('outer range (low) of the distribution:')
print(low_range)
print('\nouter range (high) of the distribution:')
print(high_range)







