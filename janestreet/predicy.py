# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import xgboost as xgb

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
# for dirname, _, filenames in os.walk('/kaggle/input'):
#     for filename in filenames:
#         print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

train = pd.read_csv('.\\train.csv', nrows=2000000)
train.info()

# Drop rows with 'weight'=0 
# Trades with weight = 0 were intentionally included in the dataset for completeness, 
# although such trades will not contribute towards the scoring evaluation
train = train[train['weight']!=0]

# Create 'action' column (dependent variable)
# The 'action' column is defined as such because of the evaluation metric used for this project.
# We want to maximise the utility function and hence pi where pi=∑j(weightij∗respij∗actionij)
# Positive values of resp will increase pi
train['action'] = train['resp'].apply(lambda x:x>0).astype(int)

# We subsequently develop a weighted-classifier based on 'resp' and 'weight'. Hence, we calculate the weights first.
sample_weights = (train['resp'] * train['weight']).abs()

features = [col for col in list(train.columns) if 'feature' in col]

# First, we want to check if the target class is balanced or unbalanced in the training data
sns.set_palette("colorblind")
ax = sns.barplot(train['action'].value_counts().index, train['action'].value_counts()/len(train))
ax.set_title("Proportion of trades with action=0 and action=1")
ax.set_ylabel("Percentage")
ax.set_xlabel("Action")
sns.despine()
# Target class is fairly balanced with almost 50% of trades corresponding to each action

# Next, we plot a diagonal correlation heatmap to see if there are strong correlations between the features

# Compute the correlation matrix
corr = train[features].corr()

# Generate a mask for the upper triangle
mask = np.triu(np.ones_like(corr, dtype=bool))

# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(12, 10))

# Generate a custom diverging colormap
cmap = sns.diverging_palette(20, 230, as_cmap=True)
# Finally, we investigate if there are missing values and we impute them
missing_values = pd.DataFrame()
missing_values['feature'] = features
missing_values['num_missing'] = [train[i].isna().sum() for i in features]
missing_values.T
# There are quite a lot of missing values across the features
# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(corr, mask=mask, cmap=cmap, vmin=-1, vmax=1, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})
# There are strong correlations between several of the features

train_median = train.median()
train = train.fillna(train_median)

X_train = train[features]
y_train = train['action']

# Before we perform PCA, we need to normalise the features so that they have zero mean and unit variance
scaler = StandardScaler()
scaler.fit(X_train)
X_train_norm = scaler.transform(X_train)

pca = PCA()
comp = pca.fit(X_train_norm)

# We plot a graph to show how the explained variation in the 129 features varies with the number of principal components
plt.plot(np.cumsum(comp.explained_variance_ratio_))
plt.grid()
plt.xlabel('Number of Principal Components')
plt.ylabel('Explained Variance')
sns.despine();

# The first 15 principal components explains about 80% of the variation
# The first 40 principal components explains about 95% of the variation

# Using the first 40 principal components, we apply the PCA mapping on both the training and test set
pca = PCA(n_components=40).fit(X_train_norm)
X_train_transform = pca.transform(X_train_norm)

parameters = {'random_state': 42,
              'tree_method': 'gpu_hist',
              'eval_metric': 'auc',
              'objective': 'binary:logistic'}
              
# We create the XGboost specific DMatrix data format from the numpy array. 
d_train = xgb.DMatrix(X_train_transform, y_train)

cv_results = xgb.cv(parameters, d_train, nfold = 5, num_boost_round=1000, early_stopping_rounds = 10, metrics = 'auc')

# Print cv_results
print(cv_results)

# Print the AUC
print((cv_results["test-auc-mean"]).iloc[-1])

# The test AUC does not increase after about 380 rounds

clf = xgb.train(parameters, d_train, 380)

# We impute the missing values with the medians
def fillna_npwhere(array, values):
    if np.isnan(array.sum()):
        array = np.where(np.isnan(array), values, array)
    return array


import janestreet
env = janestreet.make_env() # initialize the environment
iter_test = env.iter_test() # an iterator which loops over the test set

for (test_df, sample_prediction_df) in iter_test:
    wt = test_df.iloc[0].weight
    if(wt == 0):
        sample_prediction_df.action = 0 
    else:
        sample_prediction_df.action = np.where(clf.predict(xgb.DMatrix(pca.transform(scaler.transform(fillna_npwhere(test_df[features].values,train_median[features].values)))))>=0.5,1,0).astype(int)
    env.predict(sample_prediction_df)
