import cv2
import numpy as np

cap = cv2.VideoCapture(0)

# Valores de umbral
lower_thresh = 120
upper_thresh = 255

# CAMBIAR PARA ALTERNAR ENTRE MODO UMBRAL O ROI
roi = True


def drawBox(img, bbox):
    """recibe regresa imagen y recuadro en region de interes 
    muestra status de tracking en la imagen
     args: 
     img o frame
     recuadro ROI
    """
    x, y, w, h = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
    cv2.rectangle(img, (x, y), ((x + w), (y + h)), (255, 0, 255), 3, 3)
    cv2.putText(img, "Tracking ;) ", (100, 75),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)


# Utilizando region de interes
if roi:
    _, frame = cap.read()
    tracker = cv2.legacy_TrackerMOSSE.create()
    # selector de ROI con openCV
    bbox = cv2.selectROI("Tracking", frame, False)
    tracker.init(frame, bbox)
    while True:
        success, frame = cap.read()
        success, bbox = tracker.update(frame)
        #se dibuja cuadro de seguimiento o mensaje de error
        if success:
            drawBox(frame, bbox)
        else:
            cv2.putText(frame, "Lost :(", (100, 75),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.imshow("Tracking", frame)
        key = cv2.waitKey(1)
        if key == 27:
            break
# utilizando umbral y contorno de imagen
else:
    while True:
        _, frame = cap.read()
        # se convierte el frame a grises
        imgray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # desenfoque gaussiano para suavizar los bordes
        imgray = cv2.GaussianBlur(imgray, (7, 7), 0)
        # binarizamos la imagen y se buscan contornos
        ret, thresh = cv2.threshold(imgray, lower_thresh, upper_thresh, 0)
        contours, hierarchy = cv2.findContours(
            thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
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
