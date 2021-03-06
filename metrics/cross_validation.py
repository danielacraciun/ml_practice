# Evaluate using Cross Validation
from pandas import read_csv
from sklearn.model_selection import KFold, LeaveOneOut, cross_val_score, LeavePOut
from sklearn.linear_model import LogisticRegression

url = "https://goo.gl/vhm1eU"
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
dataframe = read_csv(url, names=names)
array = dataframe.values
X = array[:,0:8]
Y = array[:,8]

# kfold = KFold(n_splits=15)
# loo = LeaveOneOut()
lpo = LeavePOut(p=2)
model = LogisticRegression()
results = cross_val_score(model, X, Y, cv=lpo)
print("Accuracy: {} ({})".format(results.mean()*100.0, results.std()*100.0))
