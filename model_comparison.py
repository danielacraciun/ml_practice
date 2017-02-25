# Compare Algorithms
from pandas import read_csv
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC, SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier

from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB

# load dataset
# url = "https://goo.gl/vhm1eU"
# names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
# dataframe = read_csv(url, names=names)
# array = dataframe.values
# X = array[:,0:8]
# Y = array[:,8]
# url2 = "http://archive.ics.uci.edu/ml/machine-learning-databases/statlog/heart/heart.dat"
# names2 = [
#     'age',
#     'sex',
#     'cp',
#     'trestbps',
#     'chol',
#     'fbs',
#     'restecg',
#     'thalach',
#     'exang',
#     'oldpeak',
#     'slope',
#     'ca',
#     'thal',
#     'num',]
#
# data2 = read_csv('data.txt', names=names2, sep=' ')
# array = data2.values
# X = array[:,0:13]
# Y = array[:,13]
# prepare models

names2 = [
    'age',
    'sex',
    'trestbps',
    'chol',
    'thalach',
    'num',]

data2 = read_csv('data2.txt', names=names2, sep=' ')
array = data2.values
X = array[:,0:5]
Y = array[:,5]

models = []
models.append(('LR', LogisticRegression()))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('QDA', QuadraticDiscriminantAnalysis()))
models.append(('LSVC', LinearSVC()))
models.append(('SVC', SVC(kernel='sigmoid')))
models.append(('DTC', DecisionTreeClassifier()))
models.append(('KNC',  KNeighborsClassifier()))
models.append(('MLP',  MLPClassifier()))

models.append(('GPC',  GaussianProcessClassifier()))
models.append(('RFC',  RandomForestClassifier()))
models.append(('ABC',  AdaBoostClassifier()))
models.append(('GNB',  GaussianNB()))

# evaluate each model in turn
results = []
names = []
scoring = 'accuracy'
for name, model in models:
	kfold = KFold(n_splits=10)
	cv_results = cross_val_score(model, X, Y, cv=kfold, scoring=scoring)
	results.append(cv_results)
	names.append(name)
	msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
	print(msg)
