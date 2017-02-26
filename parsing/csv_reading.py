# Load CSV using Pandas from URL
import pandas
import matplotlib.pyplot as plt
import numpy

from pandas.tools.plotting import scatter_matrix
from sklearn.preprocessing import StandardScaler

# diabetes dataset UCI
# reading and getting the data
url = "https://goo.gl/vhm1eU"
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
data = pandas.read_csv(url, names=names)
array = data.values

# separate array into input and output components
X = array[:,0:8]
Y = array[:,8]
scaler = StandardScaler().fit(X)
rescaledX = scaler.transform(X)
numpy.set_printoptions(precision=3)
# print(rescaledX[0:5,:])

# heart disease datatset
# reading and getting the data
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
data2 = pandas.read_csv(url2, names=names2, sep=' ')

# description = data2.describe()
# scatter_matrix(data2)
# plt.show()

# separate array into input and output components
array = data2.values
X = array[:,0:13]
Y = array[:,13]
scaler = StandardScaler().fit(X)
rescaledX = scaler.transform(X)

# summarize transformed data
numpy.set_printoptions(precision=2)
print(rescaledX[0:5,:])
