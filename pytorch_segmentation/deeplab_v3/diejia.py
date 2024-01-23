import cv2

img1 = cv2.imread("test_img/boat1.png")
img2 = cv2.imread("test_img/boat1predicted.png")

res = cv2.addWeighted(img1, 0.6, img2, 0.7, 0)

cv2.imwrite("test_img/boat1_result.png", res)