
from sklearn.ensemble import RandomForestRegressor

import sklearn.preprocessing as preprocessing
from sklearn import linear_model

import pandas as pd
import numpy as np
from pandas import Series, DataFrame

data_train = pd.read_csv("~/machine_l/Database/titanic/train.csv", \
                usecols=["PassengerId","Survived", "Pclass","Name","Sex","Age","SibSp", \
                  "Parch", "Ticket", "Fare", "Cabin","Embarked"])

### 使用 RandomForestClassifier 填补缺失的年龄属性
def set_missing_ages(df):
    
    # 把已有的数值型特征取出来丢进Random Forest Regressor中
    age_df = df[['Age','Fare', 'Parch', 'SibSp', 'Pclass']]


    # 乘客分成已知年龄和未知年龄两部分
    known_age = age_df[age_df.Age.notnull()].values
    unknown_age = age_df[age_df.Age.isnull()].values

    # y即目标年龄
    y = known_age[:, 0]

    # X即特征属性值
    X = known_age[:, 1:]

    # fit到RandomForestRegressor之中
    rfr = RandomForestRegressor(random_state=0, n_estimators=2000, n_jobs=-1)
    rfr.fit(X, y)
    
    # 用得到的模型进行未知年龄结果预测
    predictedAges = rfr.predict(unknown_age[:, 1::])
    
    # 用得到的预测结果填补原缺失数据
    df.loc[ (df.Age.isnull()), 'Age' ] = predictedAges 
    
    return df, rfr

def set_Cabin_type(df):
    df.loc[ (df.Cabin.notnull()), 'Cabin' ] = "Yes"
    df.loc[ (df.Cabin.isnull()), 'Cabin' ] = "No"
    return df

data_train, rfr = set_missing_ages(data_train)
data_train = set_Cabin_type(data_train)
print(data_train.Cabin)

dummies_Cabin = pd.get_dummies(data_train['Cabin'], prefix='Cabin')

dummies_Embarked = pd.get_dummies(data_train['Embarked'], prefix='Embarked')
dummies_Sex = pd.get_dummies(data_train['Sex'], prefix='Sex')
dummies_Pclass = pd.get_dummies(data_train['Pclass'], prefix='Pclass')

df = pd.concat([data_train, dummies_Cabin, dummies_Embarked, dummies_Sex, dummies_Pclass], axis = 1)
df.drop(['Pclass', 'Name', 'Sex', 'Ticket', 'Cabin', 'Embarked'], axis=1, inplace = True)


scaler = preprocessing.StandardScaler()

Age_2 = df['Age'].values.reshape(-1,1)
age_scale_param = scaler.fit(Age_2)
df['Age_scaled'] = scaler.fit_transform(Age_2, age_scale_param)

Fare_2 = df['Fare'].values.reshape(-1,1)
fare_scale_param = scaler.fit(Fare_2)
df['Fare_scaled'] = scaler.fit_transform(Fare_2, fare_scale_param)


train_df = df.filter(regex='Survived|Age_.*|SibSp|Parch|Fare_.*|Cabin_.*|Embarked_.*|Sex_.*|Pclass_.*')
print(train_df)
train_np = train_df.values

y = train_np[:,0]

X = train_np[:, 1:]

clf = linear_model.LogisticRegression(C=1.0, solver = 'liblinear', penalty = 'l1', tol = 1e-6)
clf.fit(X, y)
print(clf)



data_test = pd.read_csv("~/machine_l/Database/titanic/test.csv", \
                usecols=["PassengerId", "Pclass","Name","Sex","Age","SibSp", \
                  "Parch", "Ticket", "Fare", "Cabin","Embarked"])
data_test.loc[ (data_test.Fare.isnull()), 'Fare' ] = 0
# 接着我们对test_data做和train_data中一致的特征变换
# 首先用同样的RandomForestRegressor模型填上丢失的年龄
tmp_df = data_test[['Age','Fare', 'Parch', 'SibSp', 'Pclass']]
null_age = tmp_df[data_test.Age.isnull()].values
# 根据特征属性X预测年龄并补上
X = null_age[:, 1:]
predictedAges = rfr.predict(X)
data_test.loc[ (data_test.Age.isnull()), 'Age' ] = predictedAges

data_test = set_Cabin_type(data_test)
dummies_Cabin = pd.get_dummies(data_test['Cabin'], prefix= 'Cabin')
dummies_Embarked = pd.get_dummies(data_test['Embarked'], prefix= 'Embarked')
dummies_Sex = pd.get_dummies(data_test['Sex'], prefix= 'Sex')
dummies_Pclass = pd.get_dummies(data_test['Pclass'], prefix= 'Pclass')


df_test = pd.concat([data_test, dummies_Cabin, dummies_Embarked, dummies_Sex, dummies_Pclass], axis=1)
df_test.drop(['Pclass', 'Name', 'Sex', 'Ticket', 'Cabin', 'Embarked'], axis=1, inplace=True)

test_Age = df_test['Age'].values.reshape(-1,1)
test_Fare = df_test['Fare'].values.reshape(-1,1)
test_age_scale_param = scaler.fit(test_Age)
test_fare_scale_param = scaler.fit(test_Fare)
df_test['Age_scaled'] = scaler.fit_transform(test_Age, test_age_scale_param)
df_test['Fare_scaled'] = scaler.fit_transform(test_Fare, test_fare_scale_param)

test = df_test.filter(regex='Age_.*|SibSp|Parch|Fare_.*|Cabin_.*|Embarked_.*|Sex_.*|Pclass_.*')
predictions = clf.predict(test)
result = pd.DataFrame({'PassengerId':data_test['PassengerId'].values, 'Survived':predictions.astype(np.int32)})
result.to_csv("~/machine_l/Database/titanic/logistic_regression_predictions.csv", index=False)









