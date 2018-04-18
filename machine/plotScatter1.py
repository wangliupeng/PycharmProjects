"""使用scatter()绘制散点图"""
import matplotlib.pyplot as plt
import kNN

group, label = kNN.creatDataSet()

x_coord = group[:, 0]
y_coord = group[:, 1]

# 作出散点图
plt.scatter(x_coord, y_coord, marker='s')

# 添加labels
for x, y, z in zip(x_coord, y_coord, label):
    plt.annotate(
        '%s' %z,
        xy=(x, y),
        xytext=(-8, 8),
        textcoords='offset points',
        ha='center',
        va='top')

plt.xlim([-0.2, 1.2])
plt.ylim([-0.2, 1.2])
plt.show()
