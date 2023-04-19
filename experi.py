import cv2
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

image = cv2.imread("occyte4.jpg", 0)
x = cv2.Sobel(image,cv2.CV_16S,1,0)
y = cv2.Sobel(image,cv2.CV_16S,0,1)
absX = cv2.convertScaleAbs(x) # return to uint8
absY = cv2.convertScaleAbs(y)
edges = cv2.addWeighted(absX,0.5,absY,0.5,0)

# 计算评估指标
# 计算评估指标
gt=cv2.imread("imagej oocyte4 edges.png",cv2.IMREAD_GRAYSCALE)
intersection = cv2.bitwise_and(edges, gt)
union = cv2.bitwise_or(edges, gt)
precision = cv2.countNonZero(intersection) / cv2.countNonZero(edges)
recall = cv2.countNonZero(intersection) / cv2.countNonZero(gt)
f1_score = 2 * precision * recall / (precision + recall)
gt=cv2.imread("imagej oocyte4 edges.png",0)
cv2.imshow("image",image)
cv2.imshow('Result', edges)
cv2.imshow("stand",gt)
print('Precision:', precision)
print('Recall:', recall)
print('F1 Score:', f1_score)
cv2.waitKey(0)
cv2.destroyAllWindows()
