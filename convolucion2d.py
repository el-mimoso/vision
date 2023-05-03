import numpy as np
import cv2


def completeConvolution(img, kernel):
    nrows, ncols = img.shape
    krows, kcols = kernel.shape
    pad_size = (krows - 1) // 2
    padded_img = np.pad(img, pad_size, mode='constant')
    output = np.zeros((nrows, ncols))
    for i in range(nrows):
        for j in range(ncols):
            output[i, j] = np.sum(
                padded_img[i:i+krows, j:j+kcols] * kernel)
    return output


img = cv2.imread('img/baboon.jpeg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

convImg = completeConvolution(img, np.ones((3, 3)) / 9)
cv2.imshow('convImg', convImg)
