import cv2
import numpy as np
import matplotlib.pyplot as plt

#Load the image using cv2.imread and convert it to grayscale using cv2.cvtColor
img = cv2.imread('occyte3.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#Apply Gaussian blur to the image to remove noise
blur = cv2.GaussianBlur(gray, (3,3), 1)
#Apply Canny edge detection to the blurred image
edges = cv2.Canny(blur, 10, 200)
#Identify and label the polar body and oocyte using contour detection.
# Find the contours using cv2.findContours and draw them using cv2.drawContours
contours, hierarchy = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours(img, contours, -1, (0,255,0), 2)
# 计算评估指标
gt=cv2.imread("Drawing of gray oocyte3.png",cv2.IMREAD_GRAYSCALE)
intersection = cv2.bitwise_and(edges, gt)
union = cv2.bitwise_or(edges, gt)
precision = cv2.countNonZero(intersection) / cv2.countNonZero(edges)
recall = cv2.countNonZero(intersection) / cv2.countNonZero(gt)
f1_score = 2 * precision * recall / (precision + recall)

print('Precision:', precision)
print('Recall:', recall)
print('F1 Score:', f1_score)
#Display the resulting image using matplotlib.pyplot
cv2.imshow("edges",edges)
cv2.imshow("stand",gt)
cv2.imshow("orig",img)
plt.imshow(img)
plt.show()
