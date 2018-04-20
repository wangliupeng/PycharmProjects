import kNN

# group, label = kNN.creatDataSet()
# print(group)
# print(label)
#
# inX = [[0, 0],[1, 2]]
# for inx in inX:
#     distance = kNN.classify0(inx, group, label, 2)
#     print(distance)

# datingMatrix, datingLabel = kNN.fileToMatrix('datingTestSet2.txt')
#
# print(datingMatrix)
# # print(datingLabel)
#
# normDatingMatrix = kNN.normDatingData(datingMatrix)
#
# print(normDatingMatrix)

# kNN.datingClassTest()

# kNN.classifyPerson()

# testVector = kNN.imgToVector('digits/trainingDigits/4_113.txt')
# print(testVector[0, 0:5])

# kNN.handWritingClassTest()

classifierResult = kNN.classifyPicture('digits/5.jpeg') #testT() #
print(classifierResult)