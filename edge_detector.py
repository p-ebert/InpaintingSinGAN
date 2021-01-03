import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

img=cv.imread("Input/Inpainting/nature_real.jpg")
edges=cv.Canny(img, 100, 500)
plt.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))
plt.show()
plt.imshow(cv.cvtColor(edges, cv.COLOR_BGR2RGB))

cv.imwrite("Input/Inpainting/nature_edges.jpg",edges)
