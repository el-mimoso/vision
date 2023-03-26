import cv2
import numpy as np

cap = cv2.VideoCapture(0)

lower_thresh = 120
upper_thresh = 255

while True:
    _, frame = cap.read()
    #se convierte el frame a grises
    imgray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # desenfoque gaussiano para suavizar los bordes
    imgray = cv2.GaussianBlur(imgray, (7, 7), 0)
    # binarizamos la imagen y se buscan contornos
    ret, thresh = cv2.threshold(imgray, lower_thresh, upper_thresh, 0)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    print("Number of contours = " + str(len(contours)))
    # dibujamos contornos
    cv2.drawContours(frame, contours, -1, (0, 255, 0), 3)
    cv2.drawContours(imgray, contours, -1, (0, 255, 0), 3)
    
    # hsv = cv2.cvtColor(blurred_frame, cv2.COLOR_BGR2HSV)
    cv2.imshow("Frame", frame)
    cv2.imshow("Mask", thresh)
    key = cv2.waitKey(1)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()
