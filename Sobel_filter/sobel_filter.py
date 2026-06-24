import cv2
import numpy as np

img = cv2.imread("sample.jpg")

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

sobel_x = cv2.Sobel(gray,cv2.CV_64F,1,0,ksize=3
)
sobel_y = cv2.Sobel(gray,cv2.CV_64F,0,1,ksize=3)

gradient = np.sqrt(sobel_x**2 + sobel_y**2)

gradient = cv2.normalize(gradient,None,0,255,cv2.NORM_MINMAX)

gradient = np.uint8(gradient)

cv2.imwrite("edges.png", gradient)

cv2.imshow("Original Image", img)

cv2.imshow("Grayscale Image", gray)

cv2.imshow("Sobel X", np.uint8(np.absolute(sobel_x)))

cv2.imshow("Sobel Y", np.uint8(np.absolute(sobel_y)))

cv2.imshow("Final Edge Detected Output", gradient)

cv2.waitKey(0)

cv2.destroyAllWindows()