import itertools
import numpy as np
import matplotlib.pyplot as plt

from sklearn import svm, datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from pandas import read_csv

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
classes=["absence", "presence"]
# Loading and centenring the data
data2 = read_csv(url2, names=names2, sep=' ')
array = data2.values
x = array[:,0:13]
y = array[:,13]

# Split the data into a training set and a test set
X_train, X_test, y_train, y_test = train_test_split(x, y, random_state=0)

# Run classifier, using a model that is too regularized (C too low) to see
# the impact on the results
classifier = svm.SVC(kernel='linear', C=0.01)
y_pred = classifier.fit(X_train, y_train).predict(X_test)
print(classification_report(y_test, y_pred, target_names=classes))
