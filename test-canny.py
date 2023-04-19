import numpy as np
import cv2
import math
import matplotlib.pyplot as plt

sigma1 = sigma2 = 1
sum = 0
gaussian = np.zeros([5, 5])
for i in range(5):
    for j in range(5):
        gaussian[i, j] = math.exp(-1 / 2 * (np.square(i - 2) / np.square(sigma1)  # 生成二维高斯分布矩阵
                                            + (np.square(j - 2) / np.square(sigma2)))) / (2 * math.pi * sigma1 * sigma2)
        sum = sum + gaussian[i, j]

gaussian = gaussian / sum

plt.show()

# 转化为灰度值图像

img = cv2.imread('occyte4.jpg', 1)

# img5 =  cv2.imread(wx1.png,0)
# 二 通道图转换成灰度图


def rgb2gray(rgb):
    return np.dot(rgb[..., :3], [0.114, 0.587, 0.299])


gray = rgb2gray(img)  # 灰度图
W, H = gray.shape
image=gray

# 三 生成梯度图
new_gray = cv2.GaussianBlur(gray, (5, 5), 0)  # 高斯模糊
# 求梯度幅值
W1, H1 = new_gray.shape
dx = np.zeros([W1 - 1, H1 - 1])
dy = np.zeros([W1 - 1, H1 - 1])
d = np.zeros([W1 - 1, H1 - 1])  # 图像幅度值

# dx1 = np.zeros([W1 - 1, H1 - 1])
# dy1 = np.zeros([W1 - 1, H1 - 1])
# d1 = np.zeros([W1 - 1, H1 - 1])#图像幅度值
for i in range(W1 - 1):
    for j in range(H1 - 1):
        dy[i, j] = new_gray[i + 1, j] - new_gray[i, j]
        dx[i, j] = new_gray[i, j + 1] - new_gray[i, j]
        d[i, j] = np.sqrt(np.square(dx[i, j]) + np.square(dy[i, j]))  # 图像梯度幅值作为图像强度值

#        dy1[i, j] = new_gray2[i + 1, j] - new_gray2[i, j]
#       dx1[i, j] = new_gray2[i, j + 1] - new_gray2[i, j]
#       d1[i, j] = np.sqrt(np.square(dx1[i, j]) + np.square(dy1[i, j]))  # 图像梯度幅值作为图像强度值

# 四 非极大值抑制
W2, H2 = d.shape
NMS = np.copy(d)
NMS[0, :] = NMS[W2 - 1, :] = NMS[:, 0] = NMS[:, H2 - 1] = 0
for i in range(1, W2 - 1):
    for j in range(1, H2 - 1):
        if d[i, j] == 0:
            NMS[i, j] = 0
        else:
            gradX = dx[i, j]
            gradY = dy[i, j]
            gradTemp = d[i, j]

            # 如果Y方向幅度值较大
            if np.abs(gradY) > np.abs(gradX):
                weight = np.abs(gradX) / np.abs(gradY)
                grad2 = d[i - 1, j]
                grad4 = d[i + 1, j]
                # 如果x,y方向梯度符号相同
                if gradX * gradY > 0:
                    grad1 = d[i - 1, j - 1]
                    grad3 = d[i + 1, j + 1]
                # 如果x,y方向梯度符号相反
                else:
                    grad1 = d[i - 1, j + 1]
                    grad3 = d[i + 1, j - 1]
                gradTemp1 = weight * grad1 + (1 - weight) * grad2
                gradTemp2 = (1 - weight) * grad3 + weight * grad4
                # 如果X方向幅度值较大
            else:
                weight = np.abs(gradY) / np.abs(gradX)
                grad2 = d[i, j - 1]
                grad4 = d[i, j + 1]
                # 如果x,y方向梯度符号相同
                if gradX * gradY > 0:
                    grad1 = d[i + 1, j - 1]
                    grad3 = d[i - 1, j + 1]
                # 如果x,y方向梯度符号相反
                else:
                    grad1 = d[i - 1, j - 1]
                    grad3 = d[i + 1, j + 1]
                gradTemp1 = (1 - weight) * grad1 + weight * grad2
                gradTemp2 = weight * grad3 + (1 - weight) * grad4

            if gradTemp >= gradTemp1 and gradTemp >= gradTemp2:
                NMS[i, j] = gradTemp
            else:
                NMS[i, j] = 0

W3, H3 = NMS.shape
DT = np.zeros([W3, H3])
TL = 0.1 * np.max(NMS)
TH = 0.15 * np.max(NMS)

# 当 “实际梯度 < 低阈值” 该点被录取为“非边缘点（背景点）
# 当 “实际梯度 > 高阈值” 该点被录取为“边缘点”
# 当低阈值< 实际梯度低< 高阈值 ”  ，需要判别
# 如果周围八邻阈的点有大于高阈值的，凡梯度大者被录取为边缘点。
for i in range(1, W3 - 1):
    for j in range(1, H3 - 1):
        if (NMS[i, j] < TL):  # 小于低阈值背景点
            DT[i, j] = 0
        elif (NMS[i, j] > TH):  # 大于高阈值边缘点
            DT[i, j] = 255
        elif (np.any((NMS[i - 1, j - 1:j + 2] > TH)).any() or NMS[i, j - 1] > TH or NMS[
            i, j + 1] > TH  # 低阈值 < x < 高阈值 比较八邻域
              or np.any((NMS[i + 1, j - 1:j + 2] > TH))):
            DT[i, j] = 255

cv2.imshow("result", DT)
cv2.imwrite('canny occyte4.png',DT)
cv2.waitKey(0)
cv2.destroyAllWindows()
