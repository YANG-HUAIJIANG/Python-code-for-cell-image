import cv2
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

image = cv2.imread("occyte4.jpg")
gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
image = cv2.GaussianBlur(gray, (5,5), 1)
x = cv2.Sobel(image,cv2.CV_16S,1,0)
y = cv2.Sobel(image,cv2.CV_16S,0,1)
absX = cv2.convertScaleAbs(x) # return to uint8
absY = cv2.convertScaleAbs(y)
Result = cv2.addWeighted(absX,0.5,absY,0.5,0)

ret,thresh = cv2.threshold(Result, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

# 对二值化图像进行膨胀操作，去除噪点
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
dilate = cv2.dilate(thresh, kernel)

# 对膨胀后的图像进行轮廓检测
contours, hierarchy = cv2.findContours(dilate, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# 遍历轮廓，找到卵母和极体位置
for cnt in contours:
    area = cv2.contourArea(cnt)
    if area < 500:
        continue
    rect = cv2.minAreaRect(cnt)
    box = cv2.boxPoints(rect)
    box = np.int0(box)
    cv2.drawContours(image,[box],0,(0,0,255),2)
# cv2.imshow("absX", absX)  #x direction
# cv2.imshow("absY", absY)   #y direction
cv2.imshow("image",image)
cv2.imshow('Result', Result)
cv2.imwrite('gray oocyte4.png',image)
cv2.imwrite('sobel oocyte4.png',Result)

cv2.waitKey(0)
cv2.destroyAllWindows()