# coding:utf-8
# 用GrabCut进行前景检测的例子
# Author: Jiahui Tang

import cv2
import numpy as np
import sys
from matplotlib import pyplot as plt

filePath = sys.argv[1]

#img = cv2.imread('4.jpg')
img = cv2.imread(filePath)
mask = np.zeros(img.shape[:2], np.uint8)

bgdModel = np.zeros((1,65), np.float64)
fgdModel = np.zeros((1,65), np.float64)

rect = (100, 50, 421, 378)
cv2.grabCut(img, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)

mask2 = np.where((mask==2) | (mask==0), 0, 1).astype('uint8')
img = img*mask2[:,:,np.newaxis]


plt.subplot(121), plt.imshow(img)
plt.title('grabcut'), plt.xticks([]), plt.yticks([])
#plt.subplot(122), plt.imshow(cv2.cvtColor(cv2.imread('4.jpg'), cv2.COLOR_BGR2RGB))
plt.subplot(122), plt.imshow(cv2.cvtColor(cv2.imread(filePath),cv2.COLOR_BGR2RGB))
plt.title('original'), plt.xticks([], plt.yticks([]))
plt.show()
