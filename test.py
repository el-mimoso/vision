# import cv2
# import numpy as np
# from matplotlib import pyplot as plt
# import urllib.request

# # req2 = urllib.request.urlopen('https://i.stack.imgur.com/sMtYT.jpg')
# req2 = urllib.request.urlopen('https://miestilobajio.com/wp-content/uploads/2023/02/14821_image_328941591_1256795045050171_1762822632699214048_n.jpg')
# req = urllib.request.urlopen('https://i.stack.imgur.com/2L67d.jpg')

# arr = np.asarray(bytearray(req.read()), dtype=np.uint8)
# arr2 = np.asarray(bytearray(req2.read()), dtype=np.uint8)

# frame = cv2.imdecode(arr,-1)
# img = cv2.imdecode(arr2,-1)
# img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)


# # img = cv2.imread("Dylan.jpg")
# # frame=cv2.imread("frames.jpg")
# rows,cols,ch = frame.shape

# # x,y points are cw from top left
# # src_interest_pts = np.float32([[0,0],[640,0],[640,480],[0,480]])
# src_interest_pts = np.float32([[0,0],[775,0],[775,447],[0,447]])
# # Affine_interest_pts = np.float32([[551,224],[843,67],[903,301],[608,455]])
# Projective_interest_pts = np.float32([[195,56],[494,158],[432,498],[36,183]])

# # M = cv2.estimateAffine2D(src_interest_pts ,Affine_interest_pts)[0]
# # Affinedst = cv2.warpAffine(img,M,(cols,rows))

# M=cv2.getPerspectiveTransform(src_interest_pts ,Projective_interest_pts)
# Projectivedst=cv2.warpPerspective(img,M,(cols,rows))

# # dst=Affinedst+Projectivedst
# plt.subplot(121),plt.imshow(img),plt.title('Input')
# plt.subplot(122),plt.imshow(Projectivedst),plt.title('Output')
# plt.show()


# import cv2

# # Read the images
# # foreground = cv2.imread("img/puppets.png")
# # background = cv2.imread("img/ocean.png")
# # alpha = cv2.imread("img/alpha.png")

# foreground = cv2.imread("img/warp.jpeg")
# background = cv2.imread("img/frame.jpeg")
# alpha = cv2.imread("img/mask.jpeg")


# # Convert uint8 to float
# foreground = foreground.astype(float)
# background = background.astype(float)

# # Normalize the alpha mask to keep intensity between 0 and 1
# alpha = alpha.astype(float)/255
# print(alpha.shape)
# # cv2.imshow("alpha", alpha)
# # Multiply the foreground with the alpha matte
# foreground = cv2.multiply(alpha, foreground)
# cv2.imshow("primer plano", foreground)
# # Multiply the background with ( 1 - alpha )
# background = cv2.multiply(1.0 - alpha, background)
# cv2.imshow("fondo", background)
# # Add the masked foreground and background.
# outImage = cv2.add(foreground, background)/255

# # Display image
# cv2.imshow("outImg", outImage)
# cv2.waitKey(0)

import numpy as np

import cv2


img = cv2.imread('img/puppets.png')

imgray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

ret, thresh = cv2.threshold(imgray, 127, 255, 0)

contours, hierarchy = cv2.findContours(
    thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

print("Number of contours = " + str(len(contours)))

print(contours[0])


cv2.drawContours(img, contours, -1, (0, 255, 0), 3)

cv2.drawContours(imgray, contours, -1, (0, 255, 0), 3)


cv2.imshow('Image', img)

cv2.imshow('Image GRAY', imgray)

cv2.waitKey(0)

cv2.destroyAllWindows()
