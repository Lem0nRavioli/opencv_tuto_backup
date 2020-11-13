from sklearn import datasets
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# from document_scanner import tile_resize


digits = datasets.load_digits()

print(digits.data.shape)
print(digits.images[0].shape)

plt.gray()
plt.imshow(digits.images[3])
plt.show()