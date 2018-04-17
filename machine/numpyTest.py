from numpy import *
randMat = mat(random.rand(4, 4))
print("randMat=",randMat)
print("randMat^{-1}=", randMat.I) # inverse of randMat
print(randMat.I*randMat)

