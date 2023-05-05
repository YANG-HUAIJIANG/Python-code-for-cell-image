import numpy as np
import cv2

#Load the image using cv2.imread and convert it to grayscale using cv2.cvtColor
img = cv2.imread('occyte3.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#Apply Gaussian blur to the image to remove noise
blur = cv2.GaussianBlur(gray, (9,9), 1)
#Apply Canny edge detection to the blurred image
edges = cv2.Canny(blur, 70, 150)
threshold_value = 40
_, result = cv2.threshold(edges, threshold_value, 255, cv2.THRESH_BINARY)
# Find the contours using cv2.findContours and draw them using cv2.drawContours
contours, hierarchy = cv2.findContours(result, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours(img, contours, -1, (0,255,0), 2)

cv2.imshow('Original Image', img)
cv2.imshow('Thresholded Image', result)
#cv2.imshow('Image with Contours', result)
cv2.imwrite("cannycnt oocyte3.png",img)
cv2.waitKey(0)
cv2.destroyAllWindows()

