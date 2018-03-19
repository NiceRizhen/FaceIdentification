# -*- coding: utf-8 -*-

import numpy as np


a = []
a.append('aaa')
a.append('bbb')
a.append('ccc')

b = []
b.append('a')
b.append('b')


al = []
al.append(1)
al.append(2)
al.append(3)

bl = []
bl.append(11)
bl.append(22)
print (a + b)
data = np.hstack((a,b))
label = np.hstack((al, bl))


print(data.shape)
arr = np.array([data, label])
arr = np.transpose(arr)
np.random.shuffle(arr)
print (arr.shape)

image_list = list(arr[:, 0])
label_list = list(arr[:, 1])
label_list = [int(i) for i in label_list]
print (image_list)


