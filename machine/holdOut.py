import numpy as np
from sklearn.model_selection import train_test_split

X, Y = np.arange(10).reshape((5, 2)), range(5)
print("X=", X)
print("Y=", Y)
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.30, random_state=42)
print("X_train=", X_train)
print("X_test=", X_test)
print("Y_train=", Y_train)
print("Y_test=", Y_test)