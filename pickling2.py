from pandas import read_csv
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import pickle
from sklearn.naive_bayes import GaussianNB
import numpy as np
# Random Forest Classification
from pandas import read_csv
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier

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

test_size = 0.2
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size)
# Fit the model on 33%
num_trees = 200
max_features = 3
model = RandomForestClassifier(n_estimators=num_trees, max_features=max_features)

model.fit(X_train, Y_train)
# save the model to disk
filename = 'finalized_model.sav'
pickle.dump(model, open(filename, 'wb'))

# some time later...

# load the model from disk
loaded_model = pickle.load(open(filename, 'rb'))
result = loaded_model.score(X_test, Y_test)
print(result)

values = np.array([20.0, 0.0, 98.0, 210.0, 112.0]).reshape(1, -1)
result = np.array([1.0]).reshape(1, -1)
result = loaded_model.score(values,result)
print(result)

model.fit(values, result)
# save the model to disk
filename = 'finalized_model.sav'
pickle.dump(model, open(filename, 'wb'))

# ...

loaded_model = pickle.load(open(filename, 'rb'))
values = np.array([52.0, 0.0, 110.0, 190.0, 102.0]).reshape(1, -1)
result = np.array([1.0]).reshape(-1, 1)
result = loaded_model.score(values,result)
print(result)
