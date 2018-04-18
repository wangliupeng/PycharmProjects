import numpy as np
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier

bag_clf = BaggingClassifier(DecisionTreeClassifier(), n_estimators=500, max_samples=100, bootstrap=True, n_jobs=-1 )

X = np.arange(10).reshape((5, 2))

print("X=", X)
loo = LeaveOneOut()
for X_train, X_test in loo.split(X):
    print('X_train:%s , X_test: %s ' % (X_train, X_test))
