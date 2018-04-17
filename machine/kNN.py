from numpy import * #导入科学计算包NumPy
import operator # 运算符模块

def creatDataSet(): # 创建了数据集和标签,用于分类
    group = array([[1.0, 1.1], [1.0, 1.0], [0.0, 0.0], [0.0, 0.1]])
    labels = ['A', 'A', 'B', 'B']
    return group, labels

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
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1 #构造字典
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]
