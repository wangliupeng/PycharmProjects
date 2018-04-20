from numpy import * #导入科学计算包NumPy
import numpy as np
import operator # 运算符模块
from os import listdir
from PIL import Image

def creatDataSet(): # 创建了数据集和标签,用于分类
    group = array([[1.0, 1.1], [1.0, 1.0], [0.0, 0.0], [0.0, 0.1]])
    labels = ['A', 'A', 'B', 'B']
    return group, labels

#简单分类器
def classify0(inX, dataSet, labels, k):
    datasetSize = dataSet.shape[0]
    diffMat = tile(inX, (datasetSize, 1)) - dataSet #复制inX成多个数组与dataSet作差
    print(diffMat.shape)
    squDiffMat = diffMat**2 #每个元素进行平方
    squDistances = squDiffMat.sum(axis=1) #对应相加
    distances = squDistances**0.5 #每个元素进行开方
    sortedDistIndicies = distances.argsort() #对元素按从小到大进行排序
    classCount = {}
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]] #找出对应标签
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1 #构造字典
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]

##文本处理
def fileToMatrix(filename):
    file = open(filename) #读入文件
    lines = file.readlines() #按行读取
    numberOfLines = len(lines) #统计行数
    datingMatrix = zeros((numberOfLines, 3)) #约会对象特征矩阵
    datingLabels = [] #约会对象类型
    index = 0
    for line in lines:
        line = line.strip() #去除首尾空格
        listFromLine = line.split('\t')  #使用tab字符把整行数据分割为数据列表
        datingMatrix[index, :] = listFromLine[0:3] #把前三个数据存储到对象特征矩阵
        datingLabels.append(int(listFromLine[-1])) #最后一个数据强制转化为整型后，存储到约会对象类型中
        index += 1
    return datingMatrix, datingLabels

#把约会对象的特征数据归一化
def normDatingData(dataSet):
    minVals = dataSet.min(0)
    maxVals = dataSet.max(0)
    ranges = maxVals - minVals
    normDataSet = zeros(shape(dataSet))
    rows = dataSet.shape[0]
    normDataSet = dataSet - tile(minVals, (rows, 1))
    normDataSet = normDataSet/tile(ranges, (rows, 1))
    return normDataSet, minVals, ranges

def datingClassTest():
    hoRatio = 0.10  # hold out 10%
    datingMatrix, datingLabels = fileToMatrix('datingTestSet2.txt')  # load data setfrom file
    normMat, minVals, ranges = normDatingData(datingMatrix)
    rows = normMat.shape[0]
    numTestVecs = int(rows * hoRatio)
    errorCount = 0.0
    for i in range(numTestVecs):
        classifierResult = classify0(normMat[i, :], normMat[numTestVecs:rows, :], datingLabels[numTestVecs:rows], 3)
        print("分类器返回结果: %d, 真正的结果: %d" % (classifierResult, datingLabels[i]))
        if (classifierResult != datingLabels[i]): errorCount += 1.0
    print("错误率: %f" % (errorCount / float(numTestVecs)))

def classifyPerson():
    resultList = ['毫无魅力', '魅力一般', '极具魅力']
    percentTats = float(input("玩视频游戏所耗时间百分比："))
    ffMiles = float(input("每年获得的飞行常客里程数："))
    iceCream = float(input("每周消费的冰淇淋公升数："))
    datingMatrix, datingLabels = fileToMatrix('datingTestSet2.txt')
    normMat, minVals, ranges = normDatingData(datingMatrix)
    person = array([ffMiles, percentTats, iceCream])
    normPerson = (person-minVals)/ranges
    classifierResult = classify0(normPerson, normMat, datingLabels, 5)
    print("此人魅力值：", resultList[classifierResult-1])

def imgToVector(filename):
    imgVector = zeros((1, 1024))
    file = open(filename)
    for i in range(32):
        line = file.readline()
        for j in range(32):
            imgVector[0, 32*i+j] = int(line[j])
    return imgVector

def handWritingClassTest():
    hwLabels = []
    trainingFileList = listdir('digits/trainingDigits') #导入所有训练文件
    numTrain = len(trainingFileList) #统计训练文件数目
    trainingMatrix = zeros((numTrain, 1024))
    # print(numTrain)
    for i in range(numTrain):
        fileNameStr = trainingFileList[i] #找处文件名
        fileStr = fileNameStr.split('.')[0] #文件名中点前的部分
        classNumStr = int(fileStr.split('_')[0]) #文件名中_前的部分
        hwLabels.append(classNumStr) #标签——与文件中数字一致的，文件名中的数字
        trainingMatrix[i, :] = imgToVector('digits/trainingDigits/%s' % fileNameStr)
    testFileList = listdir('digits/testDigits')
    errorCount = 0.0
    numTest = len(testFileList)
    for i in range(numTest):
        fileNameStr = testFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])
        testVector = imgToVector('digits/testDigits/%s' % fileNameStr)
        classifierResult = classify0(testVector, trainingMatrix, hwLabels, 3)
        print("分类器返回的是 %d, 实际上是 %d" % (classifierResult, classNumStr))
        if(classifierResult != classNumStr): errorCount += 1.0
    print("\n 总错误个数：%d" % errorCount)
    print("\n 错误率：%f" % (errorCount/float(numTest)))

#picture of other type
def classifyPicture(filename):
    hwLabels = []
    trainingFileList = listdir('digits/trainingDigits')  # 导入所有训练文件
    numTrain = len(trainingFileList)  # 统计训练文件数目
    trainingMatrix = zeros((numTrain, 1024))
    for i in range(numTrain):
        fileNameStr = trainingFileList[i]  # 找处文件名
        fileStr = fileNameStr.split('.')[0]  # 文件名中点前的部分
        classNumStr = int(fileStr.split('_')[0])  # 文件名中_前的部分
        hwLabels.append(classNumStr)  # 标签——与文件中数字一致的，文件名中的数字
        trainingMatrix[i, :] = imgToVector('digits/trainingDigits/%s' % fileNameStr)

    im = Image.open(filename)
    out = im.resize((32, 32), Image.ANTIALIAS)
    out = out.convert("L")
    data = out.getdata()
    data = np.array(data, dtype='int')
    # print(data)
    data[data < 255] = int(0)   #设为0
    data[data > 1] = int(1)  #255为黑，设为1
    # print(data.dtype)
    # mMat = np.reshape(data, (32, 32))
    # np.savetxt("5.txt", mMat, "%d")
    classifierResult = classify0(data, trainingMatrix, hwLabels, 10)
    return classifierResult

# def testT():
#     hwLabels = []
#     trainingFileList = listdir('digits/trainingDigits')  # 导入所有训练文件
#     numTrain = len(trainingFileList)  # 统计训练文件数目
#     trainingMatrix = zeros((numTrain, 1024))
#     for i in range(numTrain):
#         fileNameStr = trainingFileList[i]  # 找处文件名
#         fileStr = fileNameStr.split('.')[0]  # 文件名中点前的部分
#         classNumStr = int(fileStr.split('_')[0])  # 文件名中_前的部分
#         hwLabels.append(classNumStr)  # 标签——与文件中数字一致的，文件名中的数字
#         trainingMatrix[i, :] = imgToVector('digits/trainingDigits/%s' % fileNameStr)
#     testVector = imgToVector('digits/testDigits/1_11.txt')
#     classifierResult = classify0(testVector, trainingMatrix, hwLabels, 3)
#     return classifierResult