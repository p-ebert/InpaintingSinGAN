from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv


img = cv.imread("./Input/Inpainting/nature_edges.jpg")

img=np.array(img)
x1=110
x2=125
y1=70
y2=85
mask=img.copy()
mask[x1:x2,y1:y2,:]=[255,255,255]


mask[:x1,:,:]=[0,0,0]
mask[:,:y1,:]=[0,0,0]
mask[:,y2:,:]=[0,0,0]
mask[x2:,:,:]=[0,0,0]

plt.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))
plt.show()
plt.imshow(cv.cvtColor(mask, cv.COLOR_BGR2RGB))
plt.show()

cv.imwrite("./Input/Inpainting/nature_edges_mask.jpg", mask)
