import numpy as np
import cv2

image = cv2.imread('images/prueba12.jpg')

lower = np.array([0, 0, 0])
upper = np.array([15, 15, 15])
shapeMask = cv2.inRange(image, lower, upper)

# find the contours in the mask
im2, contours, hierarchy = cv2.findContours(shapeMask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
print "I found %d black shapes" % (len(contours))
cv2.imshow("Mask", shapeMask)
cv2.waitKey()

# loop over the contours
for c in contours:
    # draw the contour and show it
    cv2.drawContours(image, [c], -1, (0, 255, 0), 2)

cv2.imshow("Image", image)
cv2.waitKey(0)
