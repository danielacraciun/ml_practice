from numpy import loadtxt
from xgboost import XGBClassifier, plot_importance
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from pandas import read_csv
from matplotlib import pyplot

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

# Loading and centering the data
data2 = read_csv(url2, names=names2, sep=' ')
array = data2.values
X = array[:,0:13]
Y = array[:,13]

# split data into train and test sets
seed = 7
test_size = 0.33
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, random_state=seed)

# fit model no training data
model = XGBClassifier()
model.fit(X_train, y_train)

# make predictions for test data
y_pred = model.predict(X_test)
predictions = [round(value) for value in y_pred]

# evaluate predictions
accuracy = accuracy_score(y_test, predictions)
print("Accuracy: %.2f%%" % (accuracy * 100.0))

# plot feature importance
plot_importance(model)
pyplot.show()
