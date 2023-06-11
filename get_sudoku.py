import numpy as np
import cv2 as cv

def biggest_contour(contours):
    biggest = np.array([])
    max_area = 0
    for contour in contours:
        area = cv.contourArea(contour)
        if area > 50:
            perimiter = cv.arcLength(contour,True)
            approx = cv.approxPolyDP(contour, 0.02*perimiter, True)
            if area > max_area and len(approx) == 4:
                biggest = approx
                max_area = area

    return biggest, max_area

def reorder(points):
    reordered_points = np.zeros( (4, 1, 2), dtype = np.int32)
    points = points.reshape((4, 2))
    add = points.sum(1)
    reordered_points[0] = points[np.argmin(add)]
    reordered_points[3] = points[np.argmax(add)]
    diff = np.diff(points, axis = 1)
    reordered_points[1] = points[np.argmin(diff)]
    reordered_points[2] = points[np.argmax(diff)]
    
    return reordered_points



img = cv.imread("sudoku.jpg")
img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
img_blur = cv.GaussianBlur(img_gray, (5, 5), 1)
img_threshold = cv.adaptiveThreshold(img_blur, 255, 1, 1, 11, 2)
contours, hierarchy = cv.findContours(img_threshold, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
img_countours = img.copy()
biggest, max_area = biggest_contour(contours)
biggest = reorder(biggest)


cv.drawContours(img_countours, contours, -1, (0, 255, 0), 3)
cv.imshow("image",img_countours)
cv.waitKey(0)
cv.destroyAllWindows()

