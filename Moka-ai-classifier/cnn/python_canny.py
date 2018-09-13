import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

img = cv.imread('/home/vincent/Documents/MOKA/Test/2_Double/58_AC-AD_IMG_3825/image-0104.jpg')
edges = cv.Canny(img, 100, 200)

plt.subplot(121),plt.imshow(img,cmap = 'gray')
plt.title('Original Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(edges,cmap = 'gray')
plt.title('Edge Image'), plt.xticks([]), plt.yticks([])

plt.show()
