import pandas as pd
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import seaborn as sns
sns.set()
from matplotlib import pyplot as plt
%config InlineBackend.figure_format = 'retina'

df = pd.read_csv('telecom_churn.csv')

df['International plan'] = pd.factorize(df['International plan'])[0]
df['Voice mail plan'] = pd.factorize(df['Voice mail plan'])[0]
df['Churn'] = df['Churn'].astype('int')

states = df['State']
y = df['Churn']

df.drop(['State', 'Churn'], axis = 1, inplace = True)

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.neighbours import KNegihborsClassifier
from sklearn.preprocessing import StandardScaler

x_train, x_holdout, y_train, y_holdout = train_test_split(df.values, y, test_size = 0.3, random_state = 17)

tree = DecisionTreeClassifier(max_depth = 5, random_state = 17)
knn = KNeighborsClassifier(n_neighbours = 10)

tree.fit(x_train, y_train)

scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_holdout_scaled = scaler.transform(x_holdout)
knn.fit(x_train_scaled, y_train)

from sklearn.metrics import accuracy_score

tree_pred = tree.predict(x_holdout)
accuracy_score(y_holdout, knn_pred)

from sklearn.model_selection import GridSearchCV, cross_val_score

tree_params = {'max_depth' : range(1, 11), 'max_features' : range(4, 19)}
tree_grid = GridSearchCV(tree, tree_params, cv = 5, n_jobs = -1, verbose = True)
tree_grid.fit(x_train, y_train)

tree_grid.best_params_

tree_grid.best_score_

accuracy_score(y_holdout, tree_grid.predict(x_holdout))

tree_graph_to_png(tree = tree_grid.best_estimator_, feature_names = df.columns, png_file_to_save = './img/dtknn_lesson_dt1.png')

from sklearn.pipeline import Pipeline

knn_pipe = Pipeline([('scaler', StandardScaler()), ('knn', KNeighborsClassifier(n_jobs = -1))])
knn_params = {'knn__n_neighbors' : range(1, 10)}

knn_grid = GridSearchCV(knn_pipe, knn_params, cv = 5, n_jobs = -1, verbose = True)

knn_grid.fit(x_train, y_train)
knn_grid.best_params_, knn_grid.best_score_

accuracy_score(y_holdout, knn_grid.predict(x_holdout))

from sklearn.ensemble import RandomForestClassifier

forest = RandomForestClassifier(n_estimators = 100, n_jobs = -1, random_state = 17)
print(np.mean(cross_val_score(forest, x_train, y_train, cv = 5)))

forest_params = {'max_depth' : range(6, 12), 'max_features' : range(4, 19)}
forest_grid = GridSearchCV(forest, forest_params, cv = 5, n_jobs = -1, verbose = True)
forest_grid.fit(x_train, y_train)
forest_grid.best_params_, forest_grid.best_score_

accuracy_score(y_holdout, forest_grid.predict(x_holdout))

# complex case for decision trees

def form_linearly_separable_data (n = 500, x1_min = 0, x1_max = 30, x2_min = 0, x2_max = 30):

    data, target = [], []
    for i in range(n):
        
        x1 = np.random.randint(x1_min, x1_max)
        x2 = np.random.randint(x2_min, x2_max)

        if np.abs(x1 - x2) > 0.5:
            data.append([x1, x2])
            target.append(np.sign(x1 - x2))
        
    return np.array(data), np.array(target)

x, y = form_linearly_separable_data()

plt.scatter(x[:, 0], x[:, 1], c = y, cmap = 'autumn', edgecolors = 'black');

tree = DecisionTreeClassifier(random_state = 17).fit(x, y)

xx, yy = get_grid(x)
predicted = tree.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
plt.pcolormesh(xx, yy, predicted, cmap = 'autumn')
plt.scatter(x[:, 0], x[:, 1], c = y, s = 100, cmap = 'autumn', edgecolors = 'black', linewidth = 1.5)
plt.title('easy task. decision tree compexifies everything');

tree_graph_to_png(tree = tree, feature_names = ['x1', 'x2'], png_file_to_save = './img/dtknn_lesson_dt2.png')

knn = KNeighborsClassifier(n_neighbors = 1).fit(x, y)

xx, yy = get_grid(x)

predicted = knn.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
plt.pcolormesh(xx, yy, predicted, cmap = 'autumn')
plt.scatter(x[:, 0], x[:, 1], c = y, s = 100, cmap = 'autumn', edgecolors = 'black', linewidth = 1.5);
plt.title('easy task, knn. not bad');

from sklearn.datasets import load_digits

data = load_digits()
x, y = data.data, data.target

x[0, :].reshape([8, 8])

f, axes = plt.subplots(1, 4, sharey = True, figsize = (16, 6))

for i in range(4):
    axes[i].imshow(x[i, :].reshape([8, 8]), cmap = 'Greys');

x_train, x_holdout, y_train, y_holdout = train_test_split(x, y, test_size = 0.3, random_state = 17)

tree = DecisionTreeClassifier(max_depth = 5, random_state = 17)
knn_pipe = Pipeline([('scaler', StandardScaler()), ('knn', KNeighborsClassifier(n_neighbors = 10))])

tree.fit(x_train, y_train)
knn_pipe.fit(x_train, y_train);

tree_pred = tree.predict(x_holdout)
knn_pred = knn_pipe.predict(x_holdout)
accuracy_score(y_holdout, knn_pred), accuracy_score(y_holdout, tree_pred)

tree_params = {'max_depth' : [1, 2, 3, 5, 10, 20, 30, 40, 50, 64], 'max_features' : [1, 2, 3, 5, 10, 20, 30, 50, 64]}

tree_grid = GridSearchCV(tree, tree_params, cv = 5, n_jobs = -1, verbose = True)

tree_grid.fit(x_train, y_train)

tree_grid.best_params_, tree_grid.best_score_

np.mean(cross_val_score(KNeighborsClassifier(n_neighbors = 1), x_train, y_train, cv = 5))
np.mean(cross_val_score(RandomForestClassifier(random_state = 17), x_train, y_train, cv = 5))

# complex case for the nearest neoghbors method

def form_noisy_data (n_obj = 1000, n_feat = 100, random_seed = 17):

    np.seed = random_seed
    y = np.random.choice([-1, 1], size = n_obj)

    x1 = 0.3 * y
    x_other = np.random.random(size = [n_obj, n_feat - 1])
    return np.hstack([x1.reshape([n_obj, 1]), x_other]), y

x, y = form_noisy_data()


x_train, x_holdout, y_train, y_holdout = train_test_split(x, y, test_size = 0.3, random_state = 17)

from sklearn.model_selection import cross_val_score

cv_scores, holdout_scores = [], []
n_neighb = [1, 2, 3, 5] + list(range(50, 550 ,50))

for k in n_neighb:

    knn_pipe = Pipeline([('scaler', StandardScaler()), ('knn', KNeighborsClassifier(n_neighbors = k))])
    cv_scores.append(np.mean(cross_val_score(knn_pipe, x_train, y_train, cv = 5)))
    knn_pipe.fit(x_train, y_train)
    holdout_scores.append(accuracy_score(y_holdout, knn_pipe.predict(x_holdout)))

plt.plot(n_neighb, cv_scores, label = 'CV')
plt.plot(n_neighb, holdout_scores, label = 'holdout')
plt.title('easy task. knn fails')
plt.legend();

tree = DecisionTreeClassifier(random_state = 17, max_depth = 1)
tree_cv_score = np.mean(cross_val_score(tree, x_train, y_train, cv = 5))
tree.fit(x_train, y_train)
tree_holdout_score = accuracy_score(y_holdout, tree.predict(x_holdout))
print('Decision tree. CV: {}, holdout: {}'.format(tree_cv_score, tree_holdout_score))