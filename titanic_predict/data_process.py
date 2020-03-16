import pandas as pd
import numpy as np
from pandas import Series, DataFrame

             

data_train = pd.read_csv("~/machine_l/Database/titanic/train.csv", \
                usecols=["PassengerId","Survived", "Pclass","Name","Sex","Age","SibSp", \
                  "Parch", "Ticket", "Fare", "Cabin","Embarked"])

import matplotlib.pyplot as plt
fig = plt.figure()
fig.set(alpha=0.2)

plt.subplot2grid((2,3), (0,0))
data_train.Survived.value_counts().plot(kind='bar')
plt.title(u"Survive quantity")
plt.ylabel(u"people quantity")


plt.subplot2grid((2,3), (0,1))
data_train.Pclass.value_counts().plot(kind='bar')
plt.title(u"Classs Distibution")
plt.ylabel(u"people quantity")

plt.subplot2grid((2,3), (0,2))
plt.scatter(data_train.Survived, data_train.Age)
plt.grid(b=True, which='major', axis='y')
plt.title(u"Age surved distribution")
plt.ylabel(u"Age")

data_train.Pclass.astype(float)
plt.subplot2grid((2,3), (1,0), colspan=2)
data_train.Age[data_train.Pclass == 1].plot(kind='kde')
data_train.Age[data_train.Pclass == 2].plot(kind='kde')
data_train.Age[data_train.Pclass == 3].plot(kind='kde')
plt.xlabel(u"Age")
plt.ylabel(u"density")
plt.title(u"class age distribution")
plt.legend((u"top", u"second", u"third"), loc='best')


plt.subplot2grid((2,3), (1,2))
data_train.Embarked.value_counts().plot(kind='bar')
plt.title(u"Embarked quantity")
plt.ylabel(u"quantity")

# data_train.astype(float)
# plt.show()



fig_2 = plt.figure()
fig_2.set(alpha=0.2)  

Survived_0 = data_train.Pclass[data_train.Survived == 0].value_counts()
Survived_1 = data_train.Pclass[data_train.Survived == 1].value_counts()
df=pd.DataFrame({u'Survived':Survived_1, u'N_Survived':Survived_0})
df.plot(kind='bar', stacked=True)
plt.title(u"passenger class survived account")
plt.xlabel(u"Class") 
plt.ylabel(u"quantity") 
# plt.show()


#看看各性别的获救情况
fig = plt.figure()
fig.set(alpha=0.2)  # 设定图表颜色alpha参数

Survived_m = data_train.Survived[data_train.Sex == 'male'].value_counts()
Survived_f = data_train.Survived[data_train.Sex == 'female'].value_counts()
df=pd.DataFrame({u'male':Survived_m, u'female':Survived_f})
df.plot(kind='bar', stacked=True)
plt.title(u"sex survived account ")
plt.xlabel(u"Sex") 
plt.ylabel(u"quantity")
plt.show()
