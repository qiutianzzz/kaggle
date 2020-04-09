import pandas as pd
import numpy as np
from pandas import Series, DataFrame
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from scipy import stats
import warnings
from sklearn import linear_model
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
k = 15 #number of variables for heatmap
cols = corrmat.nlargest(k, 'SalePrice')['SalePrice'].index
# cm = np.corrcoef(data_train[cols].values.T)
# sns.set(font_scale=1.25)
# hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', \
#     annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)
# plt.show()

cols = np.delete(cols.values, 4)
cols = np.delete(cols, 4)
cols = np.delete(cols, 6)
cols = np.append(cols, ['MSZoning', 'LotConfig', 'LandSlope', 'BldgType', 'HouseStyle', 'RoofStyle', 'RoofMatl',
'Foundation', 'BsmtQual', 'Heating', 'HeatingQC', 'CentralAir', 'GarageQual', 'SaleType', 'SaleCondition'], axis =0)
print (cols)
#scatterplot
# sns.set()
# cols = ['SalePrice', 'OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'FullBath', 'YearBuilt']
# sns.pairplot(data_train[cols], size = 2.5)
# plt.show()

total = data_train[cols].isnull().sum().sort_values(ascending=False)
percent = (data_train[cols].isnull().sum()/data_train[cols].isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
# print (missing_data.head(20))


data_train = data_train[cols].drop((missing_data[missing_data['Total'] > 1]).index,1)

# data_train = data_train.drop(data_train.loc[data_train['Electrical'].isnull()].index)
# data_train.isnull().sum().max() #just checking that there's no missing data missing...

#standardizing data

saleprice_scaled = StandardScaler().fit_transform(data_train['SalePrice'][:,np.newaxis])
low_range = saleprice_scaled[saleprice_scaled[:,0].argsort()][:10]
high_range= saleprice_scaled[saleprice_scaled[:,0].argsort()][-10:]

data_train = pd.get_dummies(data_train)

#convert categorical variable into dummy
data_train.info()
train_np = data_train.values
y = train_np[:,0]
X = train_np[:, 1:]
print ('X shape is :', X.shape)

# X, y = make_regression(n_features=81, random_state=0)

clf = linear_model.LogisticRegression(C=1.0, penalty = 'l2', tol = 1e-6)
# clf = linear_model.LinearRegression().fit(X, y)
# clf = ElasticNet(random_state=0)

clf.fit(X, y)
print(clf)

data_test = pd.read_csv("~/machine_l/database/House_Prices/test.csv")

cols = np.delete(cols, 0)
df_test = data_test[cols]
df_test.info()


miss_data = pd.concat([df_test.GarageCars, df_test.BsmtFinSF1, \
                    df_test.OverallQual, df_test.GrLivArea, df_test['1stFlrSF'], df_test.FullBath,  
                    df_test.YearBuilt, df_test.YearRemodAdd, df_test.Fireplaces], axis=1)

# 乘客分成已知年龄和未知年龄两部分
known_garacars = miss_data[miss_data.GarageCars.notnull()].values
unknown_garacars = miss_data[miss_data.GarageCars.isnull()].values

# y即目标年龄
y = known_garacars[:, 0]
print('the data in y:', y)
# X即特征属性值
X = known_garacars[:, 4:]
print('the data in X', X)

# fit到RandomForestRegressor之中
rfr = RandomForestRegressor(random_state=0, n_estimators=2000, n_jobs=-1)
rfr.fit(X, y)

# 用得到的模型进行未知年龄结果预测
predictedCars = rfr.predict(unknown_garacars[:, 4:])
# 用得到的预测结果填补原缺失数据
df_test.loc[ (df_test.GarageCars.isnull()), 'GarageCars' ] = predictedCars 


#########################################################################################
known_garacars = miss_data[miss_data.BsmtFinSF1.notnull()].values
unknown_garacars = miss_data[miss_data.BsmtFinSF1.isnull()].values

# y即目标年龄
y = known_garacars[:, 3]
print('the data in y:', y)
# X即特征属性值
X = known_garacars[:, 4:]
print('the data in X', X)

# fit到RandomForestRegressor之中
rfr = RandomForestRegressor(random_state=0, n_estimators=2000, n_jobs=-1)
rfr.fit(X, y)

# 用得到的模型进行未知年龄结果预测
predictedCars = rfr.predict(unknown_garacars[:, 4:])
# 用得到的预测结果填补原缺失数据
df_test.loc[ (df_test.BsmtFinSF1.isnull()), 'BsmtFinSF1' ] = predictedCars

df_test.drop(['GarageQual', 'GarageYrBlt', 'BsmtQual', 'MasVnrArea'], axis=1, inplace = True)
df_test = pd.get_dummies(df_test)
data_train.info()
# df_test.info()
train_cols = data_train.columns.values

test_cols = df_test.columns.values
print ('------------------------------TEST-----------------------------------')
print(train_cols)
print(test_cols)
for i in range (data_train.shape[1]-1):
    if test_cols[i] != train_cols[i+1]:
        print ('------------------------------TEST-----------------------------------')
        print(i, test_cols[i], train_cols[i+1])
        df_test.insert(i-1, train_cols[i+1], 0) 
        test_cols = df_test.columns.values
        print(test_cols)


df_test.info()
# data_test = miss_data

predictions = clf.predict(df_test)
result = pd.DataFrame({'Id':data_test.Id.values, 'SalePrice':predictions.astype(np.int32)})
result.to_csv("~/machine_l/database/House_Prices/saleprice_predictions_0409.csv", index=False)











