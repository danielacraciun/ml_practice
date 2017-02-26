# Cross Validation Classification LogLoss
from pandas import read_csv
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression

url = "https://goo.gl/vhm1eU"
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
dataframe = read_csv(url, names=names)

url2 = "http://archive.ics.uci.edu/ml/machine-learning-databases/statlog/heart/heart.dat"
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
    'num',
]
data2 = read_csv(url2, names=names2, sep=' ')

array = data2.values
X = array[:,0:13]
Y = array[:,13]
kfold = KFold(n_splits=25)
model = LogisticRegression()
# log loss: scoring = 'neg_log_loss'
metrics = ['accuracy', 'adjusted_rand_score', 'average_precision', 'f1',
'f1_micro', 'f1_samples', 'f1_weighted', 'neg_log_loss', 'f1_macro', 
'neg_mean_absolute_error', 'neg_mean_squared_error',
'neg_median_absolute_error', 'precision', 'precision_macro', 'precision_micro',
'precision_samples', 'precision_weighted', 'r2', 'recall', 'recall_macro',
'recall_micro', 'recall_samples', 'recall_weighted', 'roc_auc']

scoring = 'accuracy'
results = cross_val_score(model, X, Y, cv=kfold, scoring=scoring)
print("Logloss: {} ({})".format(results.mean(), results.std()))
