import cv2 as cv
image = cv.imread("ci.png")
#convert image into greyscale mode
gray_image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

#find threshold of the image
_, thrash = cv.threshold(gray_image, 240, 255, cv.THRESH_BINARY)
contours, _ = cv.findContours(thrash, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)

for contour in contours:
    shape = cv.approxPolyDP(contour, 0.01*cv.arcLength(contour, True), True)
    x_cor = shape.ravel()[0]
    y_cor = shape.ravel()[1]
    
    if len(shape) == 4:
        #shape cordinates
        x,y,w,h = cv.boundingRect(shape)

        #width:height
        aspectRatio = float(w)/h
        cv.drawContours(image, [shape], 0, (0,255,0), 4)
        if aspectRatio >= 0.9 and aspectRatio <=1.1:
            cv.putText(image, "Square", (x_cor, y_cor), cv.FONT_HERSHEY_COMPLEX, 0.5, (0,0,0))
cv.imshow("Shape", image)
cv.waitKey(0)
cv.destroyAllWindows()