import cv2
import numpy as np


def get_contours(img, img_to_draw):
    contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        print(area)
        # cv2.drawContours(img_to_draw, cnt, -1, (255, 0, 0), 3)  # (target, contour, index, color, thickness)
        if area > 500:  # ignore noise shapes
            cv2.drawContours(img_to_draw, cnt, -1, (255, 0, 0), 3)
            peri = cv2.arcLength(cnt, True)  # set false if not closed perimeter
            approx = cv2.approxPolyDP(cnt, .02*peri, True)
            # print(len(approx))  # print the amount of sides, ish
            object_corners = len(approx)
            x, y, w, h = cv2.boundingRect(approx)
            if object_corners == 3: object_type = "Tri"
            if object_corners == 4:
                asp_ratio = w / float(h)  # if close to 1, it's square, else rectangle
                if .95 < asp_ratio < 1.05:
                    object_type = "Sqr"
                else: object_type = "Rect"
            if object_corners > 4: object_type = "Crcl"

            cv2.rectangle(img_to_draw, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(img_to_draw, object_type, (x+(w//2)-10, y+(h//2)-10), cv2.FONT_HERSHEY_COMPLEX, .5,
                        (0, 0, 0), 2)



img_sudo = cv2.imread("v2_train/image1.jpg")
img = cv2.imread("shapes.jpg")
img = img[150:700, :]  # remove useless text

# shape detection
imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
imgBlur = cv2.GaussianBlur(imgGray, (7, 7), 1)  # (source, kernel, sigma) sigma to increase blur
imgCanny = cv2.Canny(imgBlur, 50, 50)
imgContour = img.copy()
get_contours(imgCanny, imgContour)


# cv2.imshow("Original", img_sudo)
# cv2.imshow("Shape Original", img)
# cv2.imshow("Shape Blur", imgBlur)
# cv2.imshow("Shape Canny", imgCanny)
cv2.imshow("Shape Contour", imgContour)

cv2.waitKey()