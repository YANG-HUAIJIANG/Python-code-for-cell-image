import cv2
import numpy as np
from scipy.spatial.distance import directed_hausdorff

# Load input image and ground truth edge image
input_image = cv2.imread('sobelcnt oocyte4.png', cv2.IMREAD_GRAYSCALE)
ground_truth_edge_image = cv2.imread('manucnt occyte4.jpg', cv2.IMREAD_GRAYSCALE)

# Perform Sobel edge detection
sobel_x = cv2.Sobel(input_image, cv2.CV_64F, 1, 0, ksize=3)
sobel_y = cv2.Sobel(input_image, cv2.CV_64F, 0, 1, ksize=3)
sobel = np.hypot(sobel_x, sobel_y)
sobel = np.uint8(sobel / np.max(sobel) * 255)

# Threshold the Sobel image to get a binary edge image
_, sobel_binary = cv2.threshold(sobel, 100, 255, cv2.THRESH_BINARY)

# Convert images to sets of points
def img_to_points(img):
    points = np.argwhere(img > 0)
    return points

detected_edge_points = img_to_points(sobel_binary)
ground_truth_edge_points = img_to_points(ground_truth_edge_image)

# Compute Hausdorff distance
hausdorff_distance = max(directed_hausdorff(detected_edge_points, ground_truth_edge_points)[0],
                         directed_hausdorff(ground_truth_edge_points, detected_edge_points)[0])

print('Hausdorff distance:', hausdorff_distance)
