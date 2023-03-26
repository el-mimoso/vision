import numpy as np
import cv2
import urllib.request


# cargamos imagenes
req = urllib.request.urlopen(
    'https://img.freepik.com/psd-gratis/cartelera-vacia-ciudad_132075-5621.jpg')
req2 = urllib.request.urlopen(
    'https://miestilobajio.com/wp-content/uploads/2023/02/14821_image_328941591_1256795045050171_1762822632699214048_n.jpg')

arr = np.asarray(bytearray(req.read()), dtype=np.uint8)
arr2 = np.asarray(bytearray(req2.read()), dtype=np.uint8)

frame = cv2.imdecode(arr, -1)
img = cv2.imdecode(arr2, -1)


circles = np.zeros((4, 2), int)
counter = 0

rows, cols, ch = frame.shape
width, height, chanels = img.shape
circleColor = (0, 150, 255)


def mousePoints(event, x, y, flags, params):
    """
    recibe el evento del raton y almacena las coordenas X Y en un array
    """
    global counter
    if event == cv2.EVENT_LBUTTONDOWN:
        circles[counter] = x, y
        counter = counter + 1
        print(circles)


def masking(alpha, foreground, background):
    """
    Suma la imagen de primer plano con el fondo. 
    args:
    alpha: mascara de imagen
    foreground: imagen de primer plano
    background: imagen de fondo
    """

    
    foreground = foreground.astype(float)
    background = background.astype(float)
    alpha = alpha.astype(float)/255
    # cv2.imshow("alpha", alpha)
    #multiplicar proyeccion por mascara
    foreground = cv2.multiply(alpha, foreground)
    # cv2.imshow("primer plano", foreground)
    #multiplicar (1- mascara) por fondo
    background = cv2.multiply(1.0 - alpha, background)
    # cv2.imshow("fondo", background)
    #sumar primerplano y fondo
    outImage = cv2.add(foreground, background)/255
    # cv2.imshow("masking", outImage)
    return outImage


while True:
    if counter == 4:
        # puntos de origen
        imgorigin = np.float32(
            [[0, 0], [height, 0], [0, width], [height, width]])
        # puntos seleccionados con el raton
        selectedPts = np.float32(
            [circles[0], circles[1], circles[3], circles[2]])

        matrix = cv2.getPerspectiveTransform(imgorigin, selectedPts)
        # proyeccion de imagen
        warp = cv2.warpPerspective(
            img, matrix, (cols, rows))
        cv2.imshow("Proyeccion", warp)
        # mascara de proyeccion
        warpMask = cv2.warpPerspective(
            np.ones(img.shape)*255, matrix, (cols, rows))
        cv2.imshow("mascara", warpMask)

        finalImg = masking(alpha=warpMask, foreground=warp, background=frame)

        cv2.imshow("perspectiva", finalImg)
    for x in range(0, 4):
        #dibujar los circulos 
        cv2.circle(frame, (circles[x][0], circles[x][1]), 3, circleColor, cv2.FILLED)

    cv2.imshow("original image", frame)
    cv2.setMouseCallback("original image", mousePoints)
    cv2.waitKey(1)
