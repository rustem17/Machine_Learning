import numpy as np
import pandas as pd
import datetime
import random

# tutorial: https://www.kaggle.com/lavanyashukla01/how-i-made-top-0-3-on-a-kaggle-competition

import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor, BaggingRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.linear_model import ElasticNet, ElasticNetCV
from sklearn.svm import SVR
from mlxtend.regressor import StackingCVRegressor
import lightgbm import LGBMRegressor
from xgboost import XGBRegressor

from scipy.stats import skew, norm
from scipy.special import boxcox1p
from scipy.stats import boxcox_normmax

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import scale
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
from sklearn.decomposition import PCA

pd.set_option('display.max_columns', None)

import warnings
warnings.filterwarnings(action = "ignore")
pd.options.display.max_seq_items = 8000
pd.options.display.max_rows = 8000

import os
print(os.listdir("../input/kernel-files"))

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
train.shape, test.shape

#_______________________

# EDA

train.head()

sns.set_style("white")
sns.set_color_codes(palette = 'deep')
f, ax = plt.subplots(figsize = (8, 7))

sns.distplot(train['SalePrice'], color = 'b');
ax.xaxis.grid(False)
ax.set(ylabel = 'Frequency')
ax.set(xlabel = 'SalePrice')
ax.set(title = 'SalePrice distribution')
sns.despine(trim = True, left = True)
plt.show()

print("Skewness: %f" % train['SalePrice'].skew())
print("Kurtosis: %f" % train['SalePrice'].kurt()) # kurtosis?

# observation of features

numeric_dtypes = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
numeric = []

for i in train.columns:
    if train[i].dtype in numeric_dtypes:
        if i in ['TotalSF', 'Total_Bathrooms', 'Total_porch_sf', 'haspool', 'hasbsmt', 'hasfireplace']:
            pass
        else:
            numeric.append(i)

fig, axs = plt.subplots(ncols = 2, nrows = 0, figsize = (12, 120))
plt.subplots_adjust(right = 2)
plt.subplots_adjust(top = 2)

sns.color_palette("husl", 8)
for i, feature in enumerate(list(train[numeric]), 1):
    if (feature == 'MiscVal'):
        break
    plt.subplot(len(list(numeric)), 3, i)
    sns.scatterplot(x = feature, y = 'SalePrice', hue = 'SalePrice', palette = 'Blues', data = train)

    plt.xlabel('{}'.format(feature), size = 15, labelpad = 12.5)
    plt.ylabel('SalePrice', size = 15, labelpad = 12.5)

    for j in range(2):
        plt.tick_params(axis = 'x', labelsize = 12)
        plt.tick_params(axis = 'y', labelsize = 12)

    plt.legend(loc = 'best', prop = {'size' : 10})

plt.show()

#_______


corr = train.corr()
plt.subplots(figsize = (15, 12))
sns.heatmap(corr, vmax = 0.9, cmap = 'Blues', square = True)

# how certain features relate with SalePrice

data = pd.concat([train['SalePrice'], train['OverallQual']], axis = 1)
f, ax = plt.subplots(figsize = (8, 6))
fig = sns.boxplot(figsize = (8, 6))
fig.axis(ymin = 0, ymax = 800000);

data = pd.concat([train['SalePrice'], train['YearBuilt']], axis = 1)
f, ax = plt.subplots(figsize = (16, 8))
fig = sns.boxplot(x = train['YearBuilt'], y = 'SalePrice', data = data)
fig.axis(ymin = 0, ymax = 800000);
plt.xticks(rotation = 45);

data = pd.concat([train['SalePrice'], train['TotalBsmtSF']], axis = 1)
data.plot.scatter(x = 'TotalBsmtSF', y = 'SalePrice', alpha = 0.3, ylim = (0, 800000));

data = pd.concat([train['SalePrice'], train['LotArea']], axis = 1)
data.plot.scatter(x = 'LotArea', y = 'SalePrice', alpha = 0.3, ylim = (0, 800000));

data = pd.concat([train['SalePrice'], train['GrLivArea']], axis = 1)
data.plot.scatter(x = 'GrLivArea', y = 'SalePrice', alpha = 0.3, ylim = (0, 800000));

train_ID = train['Id']
test_ID = test['Id']
train.drop(['Id'], axis = 1, inplace = True)
test.drop(['Id'], axis = 1, inplace = True)
train.shape, test.shape

#___________________

# Feature Engineering

sns.set_style('white')
sns.set_color_codes(palette = 'deep')
f, ax = plt.subplots(figsize = (8, 7))

sns.distplot(train['SalePrice'], color = 'b');
ax.xaxis.grid(False)
ax.set(ylabel = 'Frequency')
ax.set(xlabel = 'SalePrice')
ax.set(title = 'SalePrice distribution')
sns.despine(trim = True, left = True)
plt.show()

# SalePrice is skewed to the right <--- problem, because most ML models do not do well with non-normally distributed data; can apply a log(1 + x) transform

train['SalePrice'] = np.log1p(train['SalePrice'])

sns.set_style('white')
sns.set_color_codes(palette = 'deep')
f, ax = plt.subplots(figsize = (8, 7))

sns.distplot(train['SalePrice'], fit = norm, color = 'b');

# get the fitted parameters used by the function
(mu, sigma) = norm.fit(train['SalePrice'])
print('\n mu = {:.2f} and sigma = {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)], loc = 'best')

# let's plot distribution
ax.xaxis.grid(False)
ax.set(ylabel = 'Frequency')
ax.set(xlabel = 'SalePrice')
ax.set(title = 'SalePrice distribution')
sns.despine(trim = True, left = True)

plt.show()

#_____________________

# remove outliers
train.drop(train[(train['OverallQual'] < 5) & (train['SalePrice'] > 200000)].index, inplace = True)
train.drop(train[(train['GrLivArea'] > 4500) & (train['SalePrice'] < 300000)].index, inplace = True)
train.reset_index(drop = True, inplace = True)

# split features and labels
train_labels = train['SalePrice'].reset_index(drop = True)
train_features = train.drop(['SalePrice'], axis = 1)
test_features = test

#combine train and test features in order to apply feature transformation to the entire dataset
all_features = pd.concat([train_features, test_features]).reset_index(drop = True)
all_features.shape


# Fill missing values

def  percent_missing(df):

    data = pd.DataFrame(df)
    df_cols = lit(pd.DataFrame(data))
    dict_x = {}
    for i in range(0, len(df_cols)):
        dict_x.update({df_cols[i]: round(data[df_cols[i]].isnull().mean() * 100, 2)})

    return dict_x

missing = percent_missing(all_features)
df_miss = sorted(missing.items(), key = lambda x : x[1], reverse = True)
print('Percent of missing data')
df_miss[0:10]

# visualize missing values
sns.set_style('white')
f, ax = plt.subplots(figsize = (8, 7))
sns.set_color_codes(palette = 'deep')
missing = round(train.isnull().mean() * 100, 2)
missing = missing[missing > 0]
missing.sort_values(inplace = True)
missing.plot.bar(color = 'b')


# tweak visual presentation
ax.xaxis.grid(False)
ax.set(ylabel = 'Percent of missing values')
ax.set(xlabel = 'Features')
ax.set(title = 'Percent missing data by feature')
sns.despine(trim = True, left = True)


# some of non-numeric predictors are stored as numbers; let's convert into strings
all_features['MSSubClass'] = all_features['MSSubClass'].apply(str)
all_features['YrSold'] = all_features['YrSold'].astype(str)
all_features['MoSold'] = all_features['MoSold'].astype(str)


def handle_missing(features):

    features['Functional'] = features['Functional'].fillna('Typ')
    features['Electrical'] = features['Electrical'].fillna('SBrkr')
    features['KitchenQual'] = features['KitchenQual'].fillna('TA')
    features['Exterior1st'] = features['Exterior1st'].fillna(features['Exterior1st'].mode()[0])
    features['Exterior2nd'] = features['Exterior2nd'].fillna(features['Exterior2nd'].mode()[0])
    features['SaleType'] = features['SaleType'].fillna(features['SaleType'])
    features['MSZoning'] = features.groupby('MSSubClass')['MSZoning'].transform(lambda x: x.fillna(x.mode()[0]))

    features['PoolQC'] = features['PoolQC'].fillna('None')

    for col in ('GarageYrBlt', 'GarageArea', 'GarageCars'):
        features[col] = features[col].fillna('None')

    for col in ['GarageType', 'GarageFinish', 'GarageQual', 'GarageCond']:
        features[col] = features[col].fillna('None')

    for col in ('BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2'):
        features[col] = features[col].fillna('None')

    features['LotFrontage'] = features.groupby('Neighbourhood')['LotFrontage'].transform(lambda x: x.fillna(x.median()))

    objects = []
    for i in features.columns:
        if features[i].dtype == object:
            objects.append(i)
    features.update(features[objects].fillna('None'))

    numeric_dtypes = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    numeric = []

    for i in features.columns:
        if features[i].dtype in numeric_dtypes:
            numeric.append(i)

    features.update(features[numeric].fillna(0))
    return features

all_features = handle_missing(all_features)

missing = percent_missing(all_features)
df_miss = sorted(missing.items(), key = lambda x: x[1], reverse = True)
print('Percent of missing data')
df_miss[0:10]

# no missing values anymore

# Fix skewed features

numeric_dtypes = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
numeric = []

for i in all_features.columns:
    if all_features[i].dtype in numeric_dtypes:
        numeric.append(i)

sns.set_style('white')
f, ax = plt.subplots(figsize = (8, 7))

ax.set_xscale('log')
ax = sns.boxplot(data = all_features[numeric], orient = 'h', palette = 'Set1')
ax.xaxis.grid(False)
ax.set(ylabel = 'Feature names')
ax.set(xlabel = 'Numeric values')
ax.set(title = 'Numeric Distribution of Features')
sns.despine(trim = True, left = True)

# find skewed numerical features
skew_features = all_features[numeric].apply(lambda x: skew(x)).sort_values(ascending = False)

high_skew = skew_features[skew_features > 0.5]
skew_index = high_skew.index

print('There are {} numerical features with Skew > 0.5 :'.format(high_skew.shape[0]))
skewness = pd.DataFrame({'Skew' : high_skew})
skew_features.head(10)

# normalize skewed features

for i in skew_index:
    all_features[i] = boxcox1p(all_features[i], boxcox_normmax(all_features[i] + 1))

# check if we handled all skewed values

sns.set_style('white')
f, ax = plt.subplots(figsize = (8, 7))
ax.set_xscale('log')
ax = sns.boxplot(data = all_features[skew_index], orient = 'h', palette = 'Set1')
ax.xaxis.grid(False)
ax.set(ylabel = 'Feature name')
ax.set(xlabel = 'Numeric values')
ax.set(title = 'Numeric Distribution of Features')
sns.despine(trim = True, left = True)

# Creating more features

all_features['BsmtFinType1_Unf'] = 1 * (all_features['BsmtFintype1'] == 'Unf')
all_features['HasWoodDeck'] = (all_features['WoodDeckSF'] == 0) * 1
all_features['HasOpenPorch'] = (all_features['OpenPorchSF'] == 0) * 1
all_features['HasEnclosedPorch'] = (all_features['EnclosedPorch'] == 0) * 1
all_features['Has3SsnPorch'] = (all_features['3SsnPorch'] == 0) * 1
all_features['HasScreenPorch'] = (all_features['ScreenPorch'] == 0) * 1
all_features['YearsSinceRemodel'] = all_features['YrSold'].astype(int) - all_features['YearRemodAdd'].astype(int)

all_features['Total_Home_Quality'] = all_features['OverallQual'] + all_features['OverallCond']
all_features = all_features.drop(['Utilities', 'Street', 'PoolQC',], axis = 1)
all_features['TotalSF'] = all_features['TotalBsmtSF'] + all_features['1stFlrSF'] + all_features['2ndFlrSF']
all_features['YrBltAndRemod'] = all_features['YearBuilt'] + all_features['YearRemodAdd']

all_features['Total_sqr_footage'] = (all_features['BsmtFinSF1'] + all_features['BsmtFinSF2'] + all_features['1stFlrSF'] + all_features['2ndFlrSF'])
all_features['Total_Bathrooms'] = (all_features['FullBath'] + (0.5 * all_features['HalfBath']) + all_features['BsmtFullBath' + (0.5 * all_features['BsmtHalfBath']))
all_features['Total_porch_sf'] = (all_features['OpenPorchSF'] + all_features['3SsnPorch'] + all_features['EnclosedPorch'] + all_features['ScreenPorch'] + all_features['WoodDeckSF'])

all_features['TotalBsmtSF'] = all_features['TotalBsmtSF'].apply(lambda x: np.exp(6) if x <= 0.0 else x)
all_features['2ndFlrSF'] = all_features['2ndFlrSF'].apply(lambda x: np.exp(6.5) if x <= 0.0 else x)
all_features['GarageArea'] = all_features['GarageArea'].apply(lambda x: np.exp(6) if x <= 0.0 else x)
all_features['GarageCars'] = all_features['GarageCars'].apply(lambda x: 0 if x <= 0.0 else x)
all_features['LotFrontage'] = all_features['LotFrontage'].apply(lambda x: np.exp(4.2) if x <= 0.0 else x)
all_features['MasVnrArea'] = all_features['MasVnrArea'].apply(lambda x: np.exp(4) if x <= 0.0 else x)
all_features['BsmtFinSF1'] = all_features['BsmtFinSF1'].apply(lambda x: np.exp(6.5) if x <= 0.0 else x)

all_features['haspool'] = all_features['PoolArea'].apply(lambda x: 1 if x > 0 else 0)
all_features['has2ndfloor'] = all_features['2ndFlrSF'].apply(lambda x: 1 if x > 0 else 0)
all_features['hasgarage'] = all_features['GarageArea'].apply(lambda x: 1 if x > 0 else 0)
all_features['hasbsmt'] = all_features['TotalBsmtSF'].apply(lambda x : 1 if x > 0 else 0)
all_features['hasfireplace'] = all_features['Fireplaces'].apply(lambda x: 1 if x > 0 else 0)

# Feature tranformations

def logs(res, ls):
    m = res.shape[1]
    for l in ls:
        res = res.assign(newcol = pd.Series(np.log(1.01 + res[l])).values)
        res.columns.values[m] = l + '_log'
        m += 1
    return res

log_features = ['LotFrontage', 'LotArea', 'MasVnrArea', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUndSF', 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 'LowQualFinSF', 'GrLivArea',
    'BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr', 'TotRmsAbvGrd', 'Fireplaces', 'GarageCars', 'GarageArea', 'WoodDeckSF',
    'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea', 'MiscVal', 'YearRemodAdd', 'TotalSF']

all_features = logs(all_features, log_features)

# Encode (?) categorical features

all_features = pd.get_dummies(all_features).reset_index(drop = True)
all_features.shape
all_features.head()

all_features = all_features.loc[:, ~all_features.columns.duplicated()]

# Recreate trainigs and test sets

X = all_features.iloc[:len(train_labels), :]
X_test = all_features.iloc[len(train_labels):, :]
X.shape, train_labels.shape, X_test.shape

numeric_dtypes = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
numeric = []

for i in X.columns:
    if X[i].dtype in numeric_dtypes:
        if i in ['TotalSF', 'Total_Bathrooms', 'Total_porch_sf', 'haspool', 'hasgarage', 'hasbsmt', 'hasfireplace']:
            pass
        else:
            numeric.append(i)

fig, axs = plt.subplots(ncols = 2, nrows = 0, figsize = (12, 150))
plt.subplots_adjust(right = 2)
plt.subplots_adjust(top = 2)
sns.color_palette('husl', 8)

for i, feature in enumerate(list(X[numeric]), 1):
    if (feature == 'MiscVal'):
        break
    plt.subplot(len(list(numeric)), 3, i)
    sns.scatterplot(x = feature, y = 'SalePrice', hue = 'SalePrice', palette = 'Blues', data = train)

    plt.xlabel('{}'.format(feature), size = 15, labelpad = 12.5)
    plt.ylabel('SalePrice', size = 15, labelpad = 12.5)

    for j in range(2):
        plt.tick_params(axis = 'x', labelsize = 12)
        plt.tick_params(axis = 'y', labelsize = 12)

    plt.legend(loc = 'best', prop = {'size' : 10})

plt.show()

#_____________________________________________________


# Train

#cross validation for each model
scores = {}

score = cv_rmse(lightgbm)
print('lightgbm: {:.4f} ({:.4f})'.format(score.mean(), score.std()))
scores['lgb'] = (score.mean(), score.std())

score = cv_rmse(xgboost)
print('xgboost: {:.4f} ({:.4f})'.format(score.mean(), score.std()))
scores['xgb'] = (score.mean(), score.std())

score = cv_rmse(svr)
print('SVR: {:.4f} ({:.4f})'.format(score.mean(), score.std()))
scores['svr'] = (score.mean(), score.std())

score = cv_rmse(ridge)
print('ridge: {:.4f} ({:.4f})'.format(score.mean(), score.std()))
scores['ridge'] = (score.mean(), score.std())

score = cv_rmse(rf)
print('rf: {:.4f} ({:.4f})'.format(score.mean(), score.std()))
scores['gbr'] = (score.mean(), score.std())

score = cv_rmse(gbr)
print('gbr: {:.4f} ({:.4f})'.format(score.mean(), score.std()))
scores['gbr'] = (score.mean(), score.std())

#_____

# fit models

stack_gen_model = stack_ge.fit(np.array(X), np.array(train_labels))
lgb_model_full_data = lightgbm.fit(X, train_labels)
xgb_model_full_data = xgboost.fit(X, train_labels)
svr_model_full_data = svr.fit(X, train_labels)
ridge_model_full_data = ridge.fit(X, train_labels)
rf_model_full_data = rf.fit(X, train_labels)
gbr_model_full_data = gbr.fit(X, train_labels)

# blend models and get predictions

def blended_predictions(X):
    return (
        (0.1 * ridge_model_full_data.predict(X)) + \
        (0.2 * svr_model_full_data.predict(X)) + \
        (0.1 * gbr_model_full_data.predict(X)) + \
        (0.1 * xgb_model_full_data.predict(X)) + \
        (0.1 * lgb_model_full_data.predict(X)) + \
        (0.05 * rf_model_full_data.predict(X)) + \
        (0.35 * stack_gen_model.predict(np.array(X)))
    )

blended_score = rmsle(train_labels, blended_predictions(X))
scores['blended'] = (blended_score, 0)
print('RMSLE score on train data: ')
print(blended_score)

#_______________
# identify the best performing model

sns.set_style('white')
fig = plt.figure(figsize = (24, 12))

ax = sns.pointplot(x = list(scores.keys()), y = [score for score, _ in score.values()], markers = ['o'], linestyles = ['-'])
for i, score in enumerate(score.values()):
    ax.text(i, score[0] + 0.002, '{:.6f}'.format(score[0]), horizontalalignment = 'left', size = 'large', color = 'black', weight = 'semibold')

plt.ylabel('Score (RMSE)', size = 20, labelpad = 12.5)
plt.xlabel('Model', size = 20, labelpad = 12.5)
plt.tick_params(axis = 'x', labelsize = 13.5)
plt.tick_params(axis = 'y', labelsize = 12.5)

plt.title('Scores of Models', size = 20)

plt.show()

# blended outperforms other models

# Final

submission = pd.read_csv('sample_submission.csv')
submission.shape

# append predictions from blended models
submission.iloc[:, 1] = np.floor(np.expm1(blended_predictions(X_test)))

# fix outlier predictions
q1 = submission['SalePrice'].quantile(0.0045)
q2 = submission['SalePrice'].quantile(0.99)

submission['SalePrice'] = submission['SalePrice'].apply(lambda x: x if x > q1 else x * 0.77)
submission['SalePrice'] = submission['SalePrice'].apply(lambda x: x if x < q2 else x * 1.1)
submission.to_csv('submission_regression1.csv', index = False)

# scale predictions
submission['SalePrice'] *= 1.001619
submission.to_csv('submission_regression2.csv', index = False)