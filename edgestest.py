import numpy as np
import cv2
import glob, os
import imutils

gray = cv2.imread("establishing.jpg", 0)
gray = imutils.resize(gray, width=400)

laplacian = cv2.Laplacian(gray, cv2.CV_64F)

cv2.imshow("frame", laplacian)

cv2.waitKey(0)
cv2.destroyAllWindows()
