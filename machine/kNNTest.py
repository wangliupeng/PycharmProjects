import kNN

group, label = kNN.creatDataSet()

# print(group)
# print(label)

inX = [0, 0]
distance = kNN.classify0(inX, group, label, 2)
print(distance)