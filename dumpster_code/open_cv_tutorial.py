import cv2
import numpy as np

kernel = np.ones((5, 5), np.uint8)  # to define a filter (see deep computer vision)
# import and show image
img = cv2.imread("v2_train/image1.jpg")  # read image file

# drawing on canvas
canvas = np.zeros((512, 512, 3), np.uint8)  # create a canvas to draw on (with rgb)
canvas[:] = 255, 0, 0  # blue
cv2.line(canvas, (0, 0), (300, 300), (0, 255, 0), 3)  # (imgToApply,Start,Stop,Color,Thickness)
cv2.rectangle(canvas, (0, 0), (250, 350), (0, 0, 255), cv2.FILLED)  # cv2.FILLED can be change by thickness value
cv2.circle(canvas, (450, 350), 30, (0, 255, 255), cv2.FILLED)  # (imgToApply, Center, Radius, Color, Thickness)
# (imgToApply, Text, StartPos, Font, FontSize, Color, LineThickness)
cv2.putText(canvas, "OPENCV TUTORIAL", (100, 100), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 150, 0), 2)

# cropping and resizing image
print(img.shape)  # return size of image, as np shape (640x480 color image => (480,640,3))
imgResize = cv2.resize(img, (300, 200))
imgCropped = img[25:455, 60:570]

# various image alteration
imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # convert RGB to GRAY
imgBlur = cv2.GaussianBlur(imgGray, (7, 7), 0)  # adding blur
imgCanny = cv2.Canny(imgGray, 50, 50)  # find edges/forms thresholds define how hard the lines have to be to be noticed
imgDilation = cv2.dilate(imgCanny, kernel, iterations=1)  # dilate the image to increase thickness of found edges
imgEroded = cv2.erode(imgDilation, kernel, iterations=1)


# warp image
width, height = (500, 500)  # define warped image size
pts1 = np.float32([[60, 25], [570, 25], [60, 455], [570, 455]])  # define corners of image to warp
pts2 = np.float32([[0, 0], [width, 0], [0, height], [width, height]])  # define the new image
matrix = cv2.getPerspectiveTransform(pts1, pts2)  # warping function
imgWarped = cv2.warpPerspective(img, matrix, (width, height))  # create warp image


# join image
# those are numpy funciton, the images join have to be similar in rgb channels and sizing
imgHorizontal = np.hstack((img,img))
imgVerti = np.vstack((img, img))


# detect color / first value of createTrackbar is initial value, set it with value found in color detection
def empty(a): pass
cv2.namedWindow("TrackBars")
cv2.resizeWindow("TrackBars", 640, 240)
cv2.createTrackbar("Hue Min", "TrackBars", 0, 179, empty)
cv2.createTrackbar("Hue Max", "TrackBars", 179, 179, empty)
cv2.createTrackbar("Sat Min", "TrackBars", 0, 255, empty)
cv2.createTrackbar("Sat Max", "TrackBars", 255, 255, empty)
cv2.createTrackbar("Val Min", "TrackBars", 0, 255, empty)
cv2.createTrackbar("Val Max", "TrackBars", 255, 255, empty)

while True:
    img_dustin = cv2.imread("re_dustin.png")
    imgHSV = cv2.cvtColor(img_dustin, cv2.COLOR_BGR2HSV)
    h_min = cv2.getTrackbarPos("Hue Min", "TrackBars")
    h_max = cv2.getTrackbarPos("Hue Max", "TrackBars")
    sat_min = cv2.getTrackbarPos("Sat Min", "TrackBars")
    sat_max = cv2.getTrackbarPos("Sat Max", "TrackBars")
    val_min = cv2.getTrackbarPos("Val Min", "TrackBars")
    val_max = cv2.getTrackbarPos("Val Max", "TrackBars")
    print(h_min,h_max,sat_min,sat_max,val_min,val_max)
    lower = np.array([h_min, sat_min, val_min])
    upper = np.array([h_max, sat_max, val_max])
    imgMask = cv2.inRange(imgHSV, lower, upper)
    # mask updated with fidgeted values to isolate only blue part of hat in picture
    # mask_updated = cv2.inRange(imgHSV, np.array([49, 44, 155]), np.array([126, 255, 255]))
    imgResult = cv2.bitwise_and(img_dustin, img_dustin, mask=imgMask)

    cv2.imshow("Dustin Original", img_dustin)
    cv2.imshow("Dustin HSV", imgHSV)
    cv2.imshow("Dustin Mask", imgMask)
    cv2.imshow("Dustin Result", imgResult)
    # try with those values for keeping only blue part of the hat : "49 126 44 255 155 255"

    cv2.waitKey(1)  # pause execution for arg ms (0 or none = infinite)


# cv2.imshow("Ori", img)  # display image file (but continue execution)
# cv2.imshow("Gray", imgGray)
# cv2.imshow("Blur", imgBlur)
# cv2.imshow("Canny", imgCanny)
# cv2.imshow("Dilation", imgDilation)
# cv2.imshow("Erode", imgEroded)
# cv2.imshow("Resized", imgResize)
# cv2.imshow("Cropped", imgCropped)
# cv2.imshow("Canvas", canvas)
# cv2.imshow("Warped", imgWarped)
# cv2.imshow("Horizontal join", imgHorizontal)
# cv2.imshow("Vertical join", imgVerti)
# cv2.imshow("Dustin Original", img_dustin)
# cv2.imshow("Dustin HSV", imgHSV)

cv2.waitKey()  # pause execution for arg ms (0 or none = infinite)



# https://rogerdudler.github.io/git-guide/
## to delete existing git :rd .git /S/Q


# git init (first time)
# git add filename # or . for all files
# git commit -m "update message"
# git remote add origin <github link> for first commit
# git push origin <branch name> (ex master/main etc)
## git push --force origin master (for forcing the pull thingy)


# type nul>filename # create file on windows cmd
# remove from git repo : git rm <file>