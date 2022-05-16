import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt #library for plot (histogram)

img = cv.imread('./image.jpg')
#convert to grayscale image
graysignal = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv.imwrite('grayimage.jpg', graysignal)
#calculate histogram
hist= cv.calcHist([graysignal],[0],None,[256],[0,256])

#make array flatten
y=hist.ravel()
x=[i for i in range(len(hist))]
plt.bar(x,y)
#Can use with function hist(image, bins, range)
#plt.hist(graysignal.ravel(), 256, [0,256])
plt.show()
cv.waitKey()

img2 = cv.imread('./low.jpg')
# normalize float versions
new_image = cv.normalize(img2, None, alpha=0, beta=1.2, norm_type=cv.NORM_MINMAX, dtype=cv.CV_32F)
# scale to uint8
new_image = np.clip(new_image, 0, 1)
new_image = (255*new_image).astype(np.uint8)
#write to an output image
cv.imwrite('low_stretch.jpg', new_image)

cv.imshow('Original Image', img2)
cv.imshow('New Image', new_image)
cv.waitKey()
