import cv2


# import and show image
img = cv2.imread("v2_train/image1.jpg")  # read image file
cv2.imshow("Output", img)  # display image file (but continue execution)
cv2.waitKey()  # pause execution for arg ms (0 or none = infinite)


# import and show video
video_cap = cv2.VideoCapture("video_test.mp4")  # read video
while True:
    success, img = video_cap.read()
    cv2.imshow("Video", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):  # break loop if "q" key is pressed (binary related, did not understand)
        break


## import webcam video
# cam_cap = cv2.VideoCapture(0)  # argument is webcam id, starting at 0
# cam_cap.set(3,640)  # width is id 3
# cam_cap.set(4,480)  # length is id 4
# cam_cap.set(10,100)  # brightness is id 10


# preprocessing sequence
'''img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img_proc = preprocessing(img)
img_cont = img.copy()
biggest = get_contours(img_proc, img_cont)


img_warp = get_warp(img_gray, biggest)[2:498, 2:498]

# create a cut location, then divide the picture considering the cut size
# cropped the image a bit and played with the size of the cut to get correct vision on numbers
# (fucked by border thickness variations)
s = 55  # 62
tiles = [img_warp[x:x+s, y:y+s] for x in range(0, img_warp.shape[0], s) for y in range(0, img_warp.shape[1],s)]
tiles_clean = []
for i in index:
    tiles_clean.append(tiles[i])
print(np.shape(tiles_clean))
tile_test_vanilla = tiles[0]


# cv2.imshow("Preprocessed", img_proc)
# cv2.imshow("Contour", img_cont)
# cv2.imshow("Warp", img_warp)  # img of 500 x 500
# cv2.imshow("first_digit", first_digit)

'''