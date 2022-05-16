#1. The source code  (python) of a genaral 3x3 filter. (different filters (also with negative coefficents ) must be showed.
#2. At least 2 example input pictures and output picture of 2 filters (1 with positive coefficents of the masker and 1 with negative coefficient)  
#3. The code and the input/output pictures  from a median filter
import cv2 as cv
import numpy as np

#read image
img = cv.imread('image.jpg')
img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
#create a kernel 3x3
kernel1 = np.array([[-1, -1,  -1],
                    [-1,  9, -1],
                    [-1, -1,  -1]])
#mean filter
mean = np.array([[1, 1, 1],
                [1, 1, 1],
                [1, 1, 1]])

img_k = cv.filter2D(src=img_gray, ddepth=-1, kernel=kernel1)
img_m = cv.filter2D(src=img_gray, ddepth=-1, kernel=mean)
cv.imwrite('image_k.jpg', img_k)
cv.imwrite('image_mean.jpg', img_m)


img_noise = cv.imread('noise.jpg')
# Laplacian High Pass Filter
lap_filter = cv.Laplacian(img_noise, ddepth=-1, ksize=7, scale=1, borderType=cv.BORDER_DEFAULT)
cv.imwrite('laplacian.jpg', lap_filter)

# Apply Gaussian Blur
gau_blur = cv.GaussianBlur(img_noise,(3,3),0)
cv.imwrite('GaussianBlur.jpg', gau_blur)
#median filter
img2 = cv.medianBlur(img, 5)
cv.imwrite('image_blur.jpg', img2)
