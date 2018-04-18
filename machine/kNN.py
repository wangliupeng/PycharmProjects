from numpy import * #导入科学计算包NumPy
import operator # 运算符模块

def creatDataSet(): # 创建了数据集和标签,用于分类
    group = array([[1.0, 1.1], [1.0, 1.0], [0.0, 0.0], [0.0, 0.1]])
    labels = ['A', 'A', 'B', 'B']
    return group, labels

#简单分类器
def classify0(inX, dataSet, labels, k):
    datasetSize = dataSet.shape[0]
    diffMat = tile(inX, (datasetSize, 1)) - dataSet #复制inX成多个数组与dataSet作差
    squDiffMat = diffMat**2 #每个元素进行平方
    squDistances = squDiffMat.sum(axis=1) #对应相加
    distances = squDistances**0.5 #每个元素进行开方
    sortedDistIndicies = distances.argsort() #对元素按从小到大进行排序
    classCount = {}
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]] #找出对应标签
        print(voteIlabel)
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
    normMat = normDatingData(datingMatrix)
    rows = normMat.shape[0]
    numTestVecs = int(rows * hoRatio)
    errorCount = 0.0
    for i in range(numTestVecs):
        classifierResult = classify0(normMat[i, :], normMat[numTestVecs:rows, :], datingLabels[numTestVecs:rows], 3)
        print("分类器返回结果: %d, 真正的结果: %d" % (classifierResult, datingLabels[i]))
        if (classifierResult != datingLabels[i]): errorCount += 1.0
    print("错误率: %f" % (errorCount / float(numTestVecs)))

def classifyPerson():
    resultList = ['不喜欢', '魅力一般', '极具魅力']
    percentTats = float(input("玩视频游戏所耗时间百分比："))
    ffMiles = float(input("每年获得的飞行常客里程数："))
    iceCream = float(input("每周消费的冰淇淋公升数："))
    datingMatrix, datingLabels = fileToMatrix('datingTestSet2.txt')
    normMat, minVals, ranges = normDatingData(datingMatrix)
    person = array([ffMiles, percentTats, iceCream])
    normPerson = (person-minVals)/ranges
    classifierResult = classify0(normPerson, normMat, datingLabels, 5)
    print("你喜欢这个人的可能性：", resultList[classifierResult-1])
