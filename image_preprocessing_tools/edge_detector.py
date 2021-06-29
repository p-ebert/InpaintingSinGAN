import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

#Importing original image and mask
img = cv.imread("Evaluation/mountain_2.jpg")
mask = cv.imread("Evaluation/mountain_2_mask.jpg")

#Formating qnd qpplying mqsk
mask=1-mask/255
masked_image = img*mask
masked_image = masked_image.astype(np.uint8)

plt.imshow(cv.cvtColor(masked_image, cv.COLOR_BGR2RGB))
plt.show()


#mask = cv.bitwise_not(mask)
mask = mask[:,:,0].astype(np.uint8)
mask = 1-mask
inpainted_edges = masked_image.copy()
#cv.xphoto.inpaint(masked_image, mask, inpainted_edges, cv.xphoto.INPAINT_FSR_FAST)
inpainted_edges = cv.inpaint(masked_image, mask,30, cv.INPAINT_TELEA)
plt.imshow(cv.cvtColor(inpainted_edges, cv.COLOR_BGR2RGB))
plt.show()
cv.imwrite("Evaluation/mountain.jpg", inpainted_edges)


#%%
img = cv.imread("Input/Inpainting/image_8M.jpeg")
mask = cv.imread("Input/Inpainting/image_8M_mask.jpeg")
mask=1-mask/255
masked_image = img*mask
masked_image = masked_image.astype(np.uint8)

edges=cv.Canny(masked_image, 20, 150)

plt.imshow(edges, cmap="gray")
plt.show()

plt.imshow(mask, cmap="gray")
plt.show()
mask

#mask = cv.bitwise_not(mask)
mask = mask[:,:,0].astype(np.uint8)
mask = 1-mask
inpainted_edges = edges.copy()
#cv.xphoto.inpaint(masked_image, mask, inpainted_edges, cv.xphoto.INPAINT_FSR_FAST)
inpainted_edges = cv.inpaint(edges, mask, 3, cv.INPAINT_TELEA)
plt.imshow(cv.cvtColor(inpainted_edges, cv.COLOR_BGR2RGB))
plt.show()
#cv.imwrite("Input/Inpainting/mountain_1_TELEA_filled.png", inpainted_edges)


#%%
masked_image = masked_image.astype(np.uint8)
edges=cv.Canny(masked_image, 20, 150)

plt.imshow(edges, cmap="gray")
plt.show()

#edges = cv.blur(edges,(30,30))

#kernel=np.array([[-1,-1,-1],[-1,9,-1],[-1,-1,-1]])
#edge=cv.filter2D(edges, -1, kernel)

inpainted_edges=edges.copy()

mask = mask[:,:,0].astype(np.uint8)

for i in range(edges.shape[0]):
    for j in range(edges.shape[1]):
        if mask[i,j]==0:
            edges[i-1:i+1,j-1:j+1]=0

plt.imshow(edges, cmap="gray")
plt.show()

mask=1-mask
mask=cv.bitwise_not(mask)
#cv.xphoto.inpaint(edges, mask, inpainted_edges, cv.xphoto.INPAINT_FSR_BEST)
inpainted_edges = cv.inpaint(edges, mask, 5, cv.INPAINT_NS)
plt.imshow(inpainted_edges, cmap="gray")
plt.show()
