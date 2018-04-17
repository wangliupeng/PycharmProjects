from __future__ import print_function
from sklearn import datasets
from sklearn.cross_validation import train_test_split
from sklearn.neighbors import KNeighborsClassifier

iris = datasets.load_iris()
X = iris.data
Y = iris.target
#print(X)
#print(Y)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3)
#print(X_train)
knn = KNeighborsClassifier()
knn.fit(X_train, Y_train)
print(knn.predict(X_test))

