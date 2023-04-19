
import cv2
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

image = cv2.imread("occyte4.jpg", 0)
x = cv2.Sobel(image,cv2.CV_16S,1,0)
y = cv2.Sobel(image,cv2.CV_16S,0,1)
absX = cv2.convertScaleAbs(x) # return to uint8
absY = cv2.convertScaleAbs(y)
Result = cv2.addWeighted(absX,0.5,absY,0.5,0)

# cv2.imshow("absX", absX)  #x direction
# cv2.imshow("absY", absY)   #y direction
cv2.imshow("image",image)
cv2.imshow('Result', Result)
cv2.imwrite('gray oocyte4.png',image)
cv2.imwrite('sobel oocyte4.png',Result)

cv2.waitKey(0)
cv2.destroyAllWindows()