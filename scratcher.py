import cv2
import numpy as np
import matplotlib.pyplot as plt
import colour_fill

#Mask creator
img = cv2.imread("./Evaluation/image_5.jpeg")
mask = np.zeros(img.shape)


img[80:120,310:350,:] = np.ones(3)*255
mask[80:120,310:350,:] = np.ones(3)*255

plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.show()
plt.imshow(mask)
plt.show()
#%%
cv2.imwrite("./Evaluation/image_5M.jpeg", img)
cv2.imwrite("./Evaluation/image_5M_mask.jpeg", mask)

#%%
for i in range(1,8):
    img = cv2.imread("./Evaluation/image_%sM.jpeg"%i)
    mask = cv2.imread("./Evaluation/image_%sM_mask.jpeg"%i)

    mask=1-mask/255
    img = img.astype(np.uint8)
    #mask = cv.bitwise_not(mask)
    #mask = 1- mask[:,:,0].astype(np.uint8)
    inp = colour_fill.weighted_average_colour(img, mask)
    #inp = cv2.inpaint(img, mask, 30, cv2.INPAINT_TELEA)
    plt.imshow(cv2.cvtColor(inp, cv2.COLOR_BGR2RGB))

    #cv2.imwrite("./Evaluation/image_%sM_telea30.jpeg" %i, inp)
