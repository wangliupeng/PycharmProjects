import kNN
import matplotlib.pyplot as plt

datingMatrix, datingLabel = kNN.fileToMatrix('datingTestSet2.txt')
normDatingMatrix = kNN.normDatingData(datingMatrix)

fig = plt.figure()
ax = fig.add_subplot(111)

ax.scatter(datingMatrix[:, 0], datingMatrix[:, 1], datingLabel, datingLabel)
plt.xlabel('每年获得的飞行常客里程数')
plt.ylabel('玩视频游戏所耗时间百分比')

# ax.scatter(datingMatrix[:, 0], datingMatrix[:, 2], datingLabel, datingLabel)
# plt.xlabel('每年获得的飞行常客里程数')
# plt.ylabel('每周消费的冰淇淋公升数')

# ax.scatter(datingMatrix[:, 1], datingMatrix[:, 2], datingLabel, datingLabel)
# plt.xlabel('玩视频游戏所耗时间百分比')
# plt.ylabel('每周消费的冰淇淋公升数')

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.show()


