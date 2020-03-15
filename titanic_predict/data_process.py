import pandas as pd
import numpy as np
from pandas import Series, DataFrame

# data_train = pd.read_csv("~/machine_l/Database/titanic/test.csv",sep=",")
# data_train = data_train.drop(data_train.columns[0], axis=1)
             

# data_train = pd.read_csv("~/machine_l/Database/titanic/test.csv", \
#                 usecols=["PassengerId","Survived","Pclass","Name","Sex","Age","SibSp", \
#                   "Parch", "Ticket", "Fare", "Cabin","Embarked"])


data_train = np.loadtxt("/home/leon/machine_l/Database/titanic/test.csv", delimiter=';', skiprows=1)
# usecols=range(1,10), unpack=True)

import matplotlib.pyplot as plt
fig = plt.figure()
fig.set(alpha=0.2)

print(data_train.PassengerId)
plt.subplot2grid((2,3), (0,0))
data_train.Survived.value_counts().plot(kind='bar')
plt.title(u"Survive Account（1 is survived）")
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

# data_train.Pclass.astype(float)
# plt.subplot2grid((2,3), (1,0), colspan=2)
# data_train.Age[data_train.Pclass == 1].plot(kind='kde')
# data_train.Age[data_train.Pclass == 2].plot(kind='kde')
# data_train.Age[data_train.Pclass == 3].plot(kind='kde')
# plt.xlabel(u"Age")
# plt.ylabel(u"density")
# plt.title(u"个等级的年龄分布")
# plt.legend((u"头等", u"second", u"third"), loc='best')


# plt.subplot2grid((2,3), (1,2))
# data_train.Embarked.value_counts().plot(kind='bar')
# plt.title(u"各口岸上船人数")
# plt.ylabel(u"人数")

# data_train.astype(float)
plt.show()