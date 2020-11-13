import cv2
import numpy as np
import os
import pandas as pd


index = [i for i in np.arange(100) if (i - 9) % 10 != 0 and i <= 88]  # index sudoku pic for good cropping


def preprocessing(img):
    img = cv2.resize(img,(640, 480))
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_blur = cv2.GaussianBlur(img_gray, (5, 5), 1)
    img_canny = cv2.Canny(img_blur, 30, 30)  # 30, 50 good one
    kernel = np.ones((5,5))
    img_dilation = cv2.dilate(img_canny, kernel, iterations=2)
    img_erode = cv2.erode(img_dilation, kernel, iterations=1)

    return img_erode


def get_contours(img, img_to_draw):
    biggest = np.array([])
    max_area = 0
    contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        # print(area)  # check how many things detected
        if area > 5000:  # ignore noise shapes, 220'000 seems to be an average sudoku board size
            # (target, contour, index, color, thickness)
            # cv2.drawContours(img_to_draw, cnt, -1, (255, 0, 0), 3)  # draw a rough rectangle around board
            peri = cv2.arcLength(cnt, True)  # set false if not closed perimeter
            approx = cv2.approxPolyDP(cnt, .02*peri, True)
            # print(len(approx))
            if area > max_area and len(approx) == 4:
                biggest = approx
                max_area = area
    cv2.drawContours(img_to_draw, biggest, -1, (255, 0, 0), 20)  # draw points
    return biggest


def reorder_points(data):
    # https://youtu.be/WQeoO7MI0Bs 2:37:00
    data = data.reshape((4, 2))
    total = data.sum(1)
    diff = np.diff(data, axis=1)
    data_new = np.zeros((4, 1, 2), dtype=np.int32)
    data_new[3] = data[np.argmax(total)]
    data_new[0] = data[np.argmin(total)]
    data_new[2] = data[np.argmax(diff)]
    data_new[1] = data[np.argmin(diff)]
    return data_new


def get_warp(img, biggest):
    width, height = (500, 500)  # define warped image size
    pts1 = np.float32(reorder_points(biggest))  # define corners of image to warp
    pts2 = np.float32([[0, 0], [width, 0], [0, height], [width, height]])  # define the new image
    matrix = cv2.getPerspectiveTransform(pts1, pts2)  # warping function
    img_warped = cv2.warpPerspective(img, matrix, (width, height))  # create warp image
    return img_warped


def extract_table(path, filename=None):
    if filename:
        path = os.path.join(path, filename)
        print(path)
    dat_content = [i.strip().split() for i in open(path).readlines()][2:]
    dat_content = [[int(x) for x in y] for y in dat_content]
    return dat_content


def generate_tiles(path):
    img = cv2.imread(path)
    img = cv2.resize(img, (640, 480))
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_proc = preprocessing(img)
    img_cont = img.copy()
    biggest = get_contours(img_proc, img_cont)
    img_warp = get_warp(img_gray, biggest)[2:498, 2:498]
    s = 55  # 62
    tiles = [img_warp[x:x + s, y:y + s] for x in range(0, img_warp.shape[0], s) for y in range(0, img_warp.shape[1], s)]
    tiles_clean = []
    clear_pic = lambda x: 255 if x > 120 else 0
    for i in index:
        tile = tiles[i][10:50,10:45]  # crop as much white as i can
        tile = cv2.resize(tile, (28, 28))
        tile = [[255 if x > 110 else x for x in row] for row in tile]
        tiles_clean.append(tile)

    return tiles_clean


picname = "image112.jpg"
datname = "image112.dat"
folder = "v2_train"
path_pic = os.path.join(folder, picname)
path_dat = os.path.join(folder, datname)
size = (640, 480)
# image = open_resize_image(path_pic, size)
tiles = generate_tiles("v2_train/image112.jpg")

print(np.shape(tiles))

print(tiles[0][14])
cv2.imshow("tile", np.uint8(tiles[2]))
cv2.waitKey()