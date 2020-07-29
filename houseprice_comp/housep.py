import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

#tutorial: https://www.kaggle.com/pmarcelino/comprehensive-data-exploration-with-python

from scipy.stats import norm
from sklearn.preprocessing import StandardScaler
from scipy import stats

import warnings
warnings.filterwarnings('ignore')
%matplotlib inline

df_train = pd.read_csv('train.csv')
df_train.columns

#________________________________

# create sheet with the following rows: variable, type (numerical or categorical), segment (building, space, location), expectation, conclusion, comments
# then check the expectations by drawing scatter plots or box plots between those variables and 'SalePrice', filling in the Conlcusion about variables
# based on conclusion choose only 'High' variables

# ended up with OverallQual, YearBuilt, TotalBsmtSF, GrLivArea

#________________________________

df_train['SalePrice'].describe()

sns.distplot(df_train['SalePrice']);

print("Skewness: %f" % df_train['SalePrice'].skew())
print("Kurtosis: %f" % df_train['SalePrice'].kurt())

# Relationship with numerical variables (*under this title)

var = 'GrLivArea'
data = pd.concat([df_train['SalePrice'], df_train[var]], axis = 1)
data.plot.scatter(x = var, y = 'SalePrice', ylim = (0, 800000));

# *upper lines see that SalePrice and GrLivArea have almost linear relationship

var = 'TotalBsmtSF'
data = pd.concat([df_train['SalePrice'], df_train[var]], axis = 1)
data.plot.scatter(x = var, y = 'SalePrice', ylim = (0, 800000));

# *upper lines see that SalePrice and TotalBsmtSF have strong (?) linear reaction

# Relationship with categorical features (*under this title)

var = 'OverallQual'
data = pd.concat([df_train['SalePrice'], df_train[var]], axis = 1)
f, ax = plt.subplots(figsize = (8, 6))
fig = sns.boxplot(x = var, y = "SalePrice", data = data)
fig.axis(ymin = 0, ymax = 800000);

# see that almost linear relationship between SalePrice and OverallQual

var = 'YearBuilt'
data = pd.concat([df_train['SalePrice'], df_train[var]], axis = 1)
f, ax = plt.subplots(figsize = (16, 8))
fig = sns.boxplot(x = var, y = "SalePrice", data = data)
fig.axis(ymin = 0, ymax = 800000);
plt.xticks(rotation = 90);

# see that SalePrice is higher in new houses rather than in old, however not strong tendency

# trick here was to choose right features rather than feature engineering (definition of new complex features)

#________________________________


# Correlation matrix (heatmap style)

corrmat = df_train.corr()
f, ax = plt.subplots(figsize = (12, 9))
sns.heatmap(corrmat, vmax = .8, square = True);

# SalePrice correlation matrix (zoomed heatmap style)

k = 10
cols = corrmat.nlargest(k, 'SalePrice')['SalePrice'].index
cm = np.corrcoef(df_train[cols].values.T)
sns.set(font_scale = 1.25)
hm = sns.heatmap(cm, cbar = True, annot = True, square = True, fmt = '.2f', annot_kws = {'size' : 10}, yticklabels = cols.values, xticklabels = cols.labels)
plt.show()

# see that: GarageCars and GarageArea are twin bros (choose first); TotalBsmtSF and 1stFloor are twin bros (choose first); FullBath; TotRmsAbvGrd and GrLivArea are twin bros (choose first)
# about YearBuilt <--- have to do time-series analysis on this variable; it's only slightly correlated with SalePrice

# Scatter plots between SalePrice and correlated variables

sns.set()
cols = ['SalePrice', 'OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'FullBath', 'YearBuilt']
sns.pairplot(df_train[cols], size = 2.5)
plt.show();

# Missing data

total = df_train.isnull().sum().sort_values(ascending = False)
percent = (df_train.isnull().sum() / df_train.isnull().count()).sort_values(ascending = False)
missing_data = pd.concat([total, percent], axis = 1, keys = ['Total', 'Percent'])
missing_data.head(20)

# all of the variables shown in table delete except Electrical; there just delete the observation with missing data

df_train = df_train.drop((missing_data[missing_data['Total'] > 1]).index, 1)
df_train = df_train.drop(df_train.loc[df_train['Electrical'].isnull()].index)

df_train.isnull().sum().max()

# Univariate analysis

saleprice_scaled = StandardScaler().fit_transform(df_train['SalePrice'][:, np.newaxis]);
low_range = saleprice_scaled[saleprice_scaled[:, 0].argsort()][:10]
high_range = saleprice_scaled[saleprice_scaled[:, 0].argsort()][-10:]

print('outer range (low) of the distribution:')
print(low_range)
print('\nouter range (high) of the distribution:')
print(high_range)

# some of values are outliers like 7.xxxx; low range not far from 0 and similar; high range far from 0;

# Bivariate analysis

var = 'GrLivArea'
data = pd.concat([df_train['SalePrice'], df_train[var]], axis = 1)
data.plot.scatter(x = var, y = 'SalePrice', ylim = (0, 800000));

# two values with big GrLivArea and low SalePrice are out of trend completely and should be deleted; two points 7.xxxx follow the trend, no to delete;

#deleting points
df_train.sort_values(by = 'GrLivArea', ascending = False)[:2]
df_train = df_train.drop(df_train[df_train['Id'] == 1299].index)
df_train = df_train.drop(df_train[df_train['Id'] == 524].index)

# bivariate analysis SalePrice and TotalBsmtSF

var = 'TotalBsmtSF'
data = pd.concat([df_train['SalePrice'], df_train[var]], axis = 1)
data.plot.scatter(x = var, y = 'SalePrice', ylim = (0, 800000));

# decided to not delete anything

#________________________________


# Four assumptions has to be tested: normality, homoscedasticity, linearity, absence of correlated errors

# normality (test SalePrice in a very lean way)

sns.distplot(df_train['SalePrice'], fit = norm)
fig = plt.figure()
res = stats.probplot(df_train['SalePrice'], plot = plt)

# it shows positive skewness and not following the diagonal line
# positive skewness is resolved by log transformations

df_train['SalePrice'] = np.log(df_train['SalePrice'])

sns.distplot(df_train['SalePrice'], fit = norm);
fig = plt.figure()
res = stats.probplot(df_train['SalePrice'], plot = plt)

# let's see at GrLivArea

sns.distplot(df_train['GrLivArea'], fit = norm);
fig = plt.figure()
res = stats.probplot(df_train['GrLivArea'], plot = plt)

# positive skewness again

df_train['GrLivArea'] = np.log(df_train['GrLivArea'])

sns.distplot(df_train['GrLivArea'], fit = norm);
fig = plt.figure()
res = stats.probplot(df_train['GrLivArea'], plot = plt)

# let's see at TotalBsmtSF

sns.distplot(df_train['TotalBsmtSF'], fit = norm);
fig = plt.figure()
res = stats.probplot(df_train['GrLivArea'], plot = plt)

# looks like positive skewness, but have a lot of 0 values, which means no log transformation is allowed

# let's do log transformation to non-zero values only

df_train['HasBsmt'] = pd.Series(len(df_train['TotalBsmtSF']), index = df_train.index)
df_train['HasBsmt'] = 0
df_train.loc[df_train['TotalBsmtSF'] > 0, 'HasBsmt'] = 1

df_train.loc[df_train['HasBsmt'] == 1, 'TotalBsmtSF'] = np.log(df_train['TotalBsmtSF'])

sns.distplot(df_train[df_train['TotalBsmtSF'] > 0]['TotalBsmtSF'], fit = norm);
fig = plt.figure()
res = stats.probplot(df_train[df_train['TotalBsmtSF'] > 0]['TotalBsmtSF'], plot = plt)

#________________________________

# homoscedasticity
# departures from an equal dispersion are shown by such shapes as cones or diamonds

plt.scatter(df_train['GrLivArea'], df_train['SalePrice']);

# no conic dispersion that was before; disappeared due to normality!!!

plt.scatter(df_train[df_train['TotalBsmtSF'] > 0]['TotalBsmtSF'], df_train[df_train['TotalBsmtSF'] > 0]['SalePrice']);

# no conic shape that was before also, equal levels of variance across the range --> very good

# convert categorical variable into dummy

df_train = pd.get_dummies(df_train)
