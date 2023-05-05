import cv2
import numpy as np

#sobel test to image
image_path = 'occyte8.jpg'
image = cv2.imread(image_path)
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
FILLTER=9
THREHOLD=35
AERA=22000
blurred_image = cv2.GaussianBlur(gray_image, (FILLTER,FILLTER), 1)
sobel_x = cv2.Sobel(blurred_image, cv2.CV_64F, 1, 0, ksize=3)
sobel_y = cv2.Sobel(blurred_image, cv2.CV_64F, 0, 1, ksize=3)
# Calculate the magnitude and angle of the gradients
magnitude = np.sqrt(np.square(sobel_x) + np.square(sobel_y))
angle = np.arctan2(sobel_y, sobel_x) * 180 / np.pi
# Normalize the magnitude to the range [0, 255]
result1 = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
threshold_value = THREHOLD
_, result2 = cv2.threshold(result1, threshold_value, 255, cv2.THRESH_BINARY)
#find contour
def getContours(img):
    contours,hierarchy = cv2.findContours(img,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        print(area)
        if area>AERA:
            cv2.drawContours(imgcnt, cnt, -1, (0, 255, 0), 3)
            peri = cv2.arcLength(cnt,True)
            print(peri)
            approx = cv2.approxPolyDP(cnt,0.02*peri,True)
            print(len(approx))

imgcnt=image.copy()
getContours(result2)

cv2.imshow('Original Image', image)
cv2.imshow('Thresholded Image', result2)
cv2.imshow('Image with Contours', imgcnt)
cv2.imwrite("sobelcnt oocyte8.png",imgcnt)

cv2.waitKey(0)
cv2.destroyAllWindows()
