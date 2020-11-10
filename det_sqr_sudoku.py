import cv2
import numpy as np


def get_contours(img, img_to_draw):
    contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        print(area)
        # if area > 500: doesn't work well in case of noisy picures
        cv2.drawContours(img_to_draw, cnt, -1, (255, 0, 0), 2)  # (target, contour, index, color, thickness)
        peri = cv2.arcLength(cnt, True)  # set false if not closed perimeter
        approx = cv2.approxPolyDP(cnt, .02 * peri, True)
        print(len(approx))  # print the amount of sides, ish
        objectCorners = len(approx)
        x, y, w, h = cv2.boundingRect(approx)
        cv2.rectangle(img_to_draw, (x, y), (x + w, y + h), (0, 255, 255), 1)


def detect_shape(path):
    img = cv2.imread(path)
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    imgBlur = cv2.GaussianBlur(imgGray, (7, 7), 5)
    imgCanny = cv2.Canny(imgBlur, 20, 20)
    get_contours(imgCanny, img)

    cv2.imshow("Shape contour", img)
    cv2.waitKey()


detect_shape("v2_train/image1.jpg")