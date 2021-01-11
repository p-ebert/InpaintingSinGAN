from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv

img = cv.imread("./Input/Inpainting/nature_edges.jpg")
img.shape
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

#In[]:
img = cv.imread("./Input/Inpainting/nature_edges.jpg")
mask = cv.imread("./Input/Inpainting/nature_edges_mask.jpg")

img=np.array(img)

mask=1-mask/255

img=img*mask.astype("uint8")


mask = cv.imread("./Input/Inpainting/nature_edges_mask.jpg")
mask=mask[:,:,0]
res_FSRFAST=img.copy()
#mask1=cv.bitwise_not(mask1)
#mask2=cv.bitwise_not(mask2)
#img = cv.imread("./Input/Inpainting/nature_edges.jpg")
mask=cv.bitwise_not(mask)
cv.xphoto.inpaint(img, mask, res_FSRFAST, cv.xphoto.INPAINT_FSR_BEST)
kernel=np.array([[-1,-1,-1],[-1,9,-1],[-1,-1,-1]])*0.8
im=cv.filter2D(res_FSRFAST, -1, kernel)
plt.imshow(im)

cv.imwrite("./Input/Inpainting/nature_edges_propagated.jpg", im)
