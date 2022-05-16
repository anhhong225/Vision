#1.After the lesson you finish the assignment with the Erosion and the Dilation.
#First you have to think how to make binairy images (thresholding)
#2. You write a (python) program (NOT using libraries) for opening and closing
#3. You write a (python) program  for a repeating erosion, with the condition that the last pixel has remain.
#Submit all the program code and the input eaxample and output result in 1 .pdf file

import cv2 as cv
import numpy as np


#read the image
img = cv.imread('gray.jpeg')
#Using function to convert image to black and white image
(thresh, im_bw) = cv.threshold(img, 127, 255, cv.THRESH_BINARY)
cv.imwrite('BinaryImage.jpg', im_bw)
#cv.imshow('Binary image', im_bw)
#cv.waitKey()
#cv.destroyAllWindows()

#Read the image for erosion
img1= cv.imread('mor.jpg',0)

def erosion(image):
    #Acquire size of the image
    m,n = image.shape
    # Define the structuring element
    SE= np.ones((5,5), dtype=np.uint8)
    constant= (5-1)//2

    #Define new image
    imgErode= np.zeros((m,n), dtype=np.uint8)
    #Erosion for morphology
    for i in range(constant, m-constant):
      for j in range(constant,n-constant):
        temp= image[i-constant:i+constant+1, j-constant:j+constant+1]
        product= temp*SE
        imgErode[i,j]= np.min(product)
    return imgErode

def dilation(image):
    #Acquire size of the image
    m,n = image.shape
    #Define new image to store the pixels of dilated image
    imgDilate= np.zeros((m,n), dtype=np.uint8)
    #Define the structuring element 
    SED= np.array([[0,1,0], [1,1,1],[0,1,0]])
    constant1=1
    #Dilation operation without using inbuilt CV2 function
    for i in range(constant1, m-constant1):
      for j in range(constant1,n-constant1):
        temp= image[i-constant1:i+constant1+1, j-constant1:j+constant1+1]
        product= temp*SED
        imgDilate[i,j]= np.max(product)
    return imgDilate

def opening(image):
    o1 = erosion(image)
    o2 = dilation(o1)
    return o2

def closing(image):
    c1 = dilation(image)
    c2 = erosion(c1)
    return c2

final_o = opening(img1)
final_c = closing(img1)
cv.imshow('origin', img1)
cv.imshow('opening', final_o)
cv.imshow('closing', final_c)
cv.waitKey()
cv.destroyAllWindows()

