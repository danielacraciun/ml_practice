import numpy as np
from sklearn import datasets
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from scipy.stats import uniform as sp_rand

# load the diabetes datasets
dataset = datasets.load_diabetes()

# prepare a range of alpha values to test
alphas = np.array([1, 0.1, 0.01, 0.0001, 0.001, 0])

# prepare a uniform distribution to sample for the alpha parameter
param_grid = {'alpha': sp_rand()}

# Grid Search for Algorithm Tuning
# create and fit a ridge regression model, testing each alpha
# model = Ridge()
# grid = GridSearchCV(estimator=model, param_grid=dict(alpha=alphas))
# grid.fit(dataset.data, dataset.target)
# print(grid)
# # summarize the results of the grid search
# print(grid.best_score_)
# print(grid.best_estimator_.alpha)

# Randomized Search for Algorithm Tuning
# create and fit a ridge regression model, testing random alpha values
model = Ridge()
rsearch = RandomizedSearchCV(estimator=model, param_distributions=param_grid, n_iter=100)
rsearch.fit(dataset.data, dataset.target)
print(rsearch)
# summarize the results of the random parameter search
print(rsearch.best_score_)
print(rsearch.best_estimator_.alpha)
