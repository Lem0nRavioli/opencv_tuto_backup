import numpy as np

data = np.array([[[42, 13]],
                 [[49, 473]],
                 [[588, 462]],
                 [[577, 16]]], dtype=int)
'''
print(data[:,:,0] + data[:, :, 1])
data_total = data[:,:,0] + data[:, :, 1]
print(np.argmax(data_total))
print(np.argmin(data_total))
data_new = np.array([], dtype=int)
data_new = np.append(data_new, data[np.argmax(data_total)])
data_new = np.append(data_new, data[np.argmin(data_total)])
print(data_new)

print("zdqidfohjzqdbvgqhzijopdqzgduihoqjz")'''


data = data.reshape((4,2))
total = data.sum(1)
diff = np.diff(data, axis=1)
print(diff)
print(total)
print(data)
data_new = np.zeros((4,1,2), dtype=np.int32)
data_new[3] = data[np.argmax(total)]
data_new[0] = data[np.argmin(total)]
data_new[2] = data[np.argmax(diff)]
data_new[1] = data[np.argmin(diff)]

print(data_new)
# print(data.max(axis=2))