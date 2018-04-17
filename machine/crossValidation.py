# import numpy as np
# from sklearn.model_selection import KFold
#
# X= np.arange(10).reshape((5, 2))
# print("X=", X)
# kf = KFold(n_splits=2)
# for train_index, test_index in kf.split(X):
#     print('X_train:%s ' % X[train_index])
#     print('X_test: %s ' % X[test_index])

import numpy as np
from sklearn.model_selection import LeaveOneOut

X= np.arange(10).reshape((5, 2))
print("X=", X)
loo = LeaveOneOut()
for train_index, test_index in loo.split(X):
    print('X_train:%s ' % X[train_index])
    print('X_test: %s ' % X[test_index])