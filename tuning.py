# Grid Search for Algorithm Tuning
# TODO: how to work with Naive Bayes priors (also look for SVM and and Decision trees maybe can be tuned)

from pandas import read_csv
import numpy
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import GaussianNB

names2 = [
    'age',
    'sex',
    'cp',
    'trestbps',
    'chol',
    'fbs',
    'restecg',
    'thalach',
    'exang',
    'oldpeak',
    'slope',
    'ca',
    'thal',
    'num',]

data2 = read_csv('data.txt', names=names2, sep=' ')
array = data2.values
X = array[:,0:13]
Y = array[:,13]

priors = numpy.array([0.6, 0.4])
param_grid = dict(priors=priors)
# model = Ridge()
model = GaussianNB()

grid = GridSearchCV(estimator=model, param_grid=param_grid)
grid.fit(X, Y)
print(grid.best_score_)
print(grid.best_estimator_.alpha)
