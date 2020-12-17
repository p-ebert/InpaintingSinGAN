from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import cv2

img = Image.open("./Segmented_image/mountain.jpg")

img=np.array(img)
x1=105
x2=165
y1=235
y2=260
mask=img.copy()
mask[x1:x2,y1:y2,:]=[255,255,255]
mask[:x1,:,:]=[0,0,0]
mask[:,:y1,:]=[0,0,0]
mask[:,y2:,:]=[0,0,0]
mask[x2:,:,:]=[0,0,0]

cv2.imwrite("./Segmented_image/mountain_mask.jpg",mask)


#Image.fromarray(mask).save("./Segmented_image/mountain_mask.jpg")
"""
for i in range(x1,x2):
    for j in range(y1,y2):
        img[i,j,:]=(((y2-j)/(y2-y1))*img[i,y1-1,:]+((j-y1)/(y2-y1))*img[i,y2+1,:]+((x2-i)/(x2-x1))*img[x1-1,j,:]+((i-x1)/(x2-x1))*img[x2+1,j,:])*0.5
"""
for i in range(x1,x2):
    for j in range(y1,y2):
        img[i,j,:]=((y2-j)/(y2-y1))*img[i,y1-1,:]+((j-y1)/(y2-y1))*img[i,y2+1,:]

#cv2.imwrite("./Segmented_image/mountain_horizontal_average.jpg",img)
plt.imshow(img)
#plt.savefig("mountain_horizontal_average.jpg")
