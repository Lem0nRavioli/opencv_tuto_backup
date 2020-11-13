import os
from PIL import Image
import matplotlib.pyplot as plt


entries = os.listdir("../v2_train/")
entries_dat = ([entry for entry in entries if os.path.splitext(entry)[1] == ".dat"])

# dat_content = [i.strip().split() for i in open("v2_train/image1.dat").readlines()]
# print(dat_content[2:])


def extract_table(path, filename):
    filepath = os.path.join(path, filename)
    dat_content = [i.strip().split() for i in open(filepath).readlines()][2:]
    return dat_content

# print(extract_table("v2_train", entries_dat[0]))


# extract pixel values from jpg file
im = Image.open('../v2_train/image1.jpg', 'r')
pix_val = list(im.getdata())

# reshape pixel values
pix_list = []
index = 0
for i in range(480):
    pix_list.append(pix_val[index: index+640])
    index += 640

# print the image, need to be put in 640x480 first
plt.imshow(pix_list)
plt.show()
