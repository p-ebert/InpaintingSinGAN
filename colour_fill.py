import cv2
import numpy as np
import matplotlib.pyplot as plt

def weighted_average_colour(img, mask):

    #mask = 1-mask/255

    for i in range(mask.shape[0]):
        for j in range(mask.shape[1]):
            if mask[i,j,0]<0.1:

                for x in range(i):
                    if mask[i-x,j,0]>=0.9:
                        up_colour=img[i-x,j,:]
                        up_distance = x
                        break

                for y in range(mask.shape[0]-i):
                    if mask[i+y,j,0]>=0.9:
                        down_colour=img[i-x,j,:]
                        down_distance = y
                        break

                for z in range(j):
                    if mask[i,j-z,0]>=0.9:
                        right_colour=img[i,j-z,:]
                        right_distance = z

                        break

                for w in range(mask.shape[1]-j):
                    if mask[i,j+w,0]>=0.9:
                        left_colour=img[i,j+w,:]
                        left_distance = w
                        break

                total_distance = left_distance + right_distance + up_distance + down_distance

                img[i,j] =  (left_distance/total_distance) * left_colour + (right_distance/total_distance) * right_colour + (up_distance/total_distance) * up_colour + (down_distance/total_distance) * down_colour

    return img

def navier_stokes_filler(img, mask):
    #Formating and applying the mask
    #mask=1-mask/255
    masked_image = img*mask
    masked_image = masked_image.astype(np.uint8)

    #mask = cv.bitwise_not(mask)
    mask = mask[:,:,0].astype(np.uint8)
    mask = 1-mask

    return cv2.inpaint(masked_image, mask, 3, cv.INPAINT_NS)

def telea_filler(img, mask):
    #Formating and applying the mask
    #mask=1-mask/255
    masked_image = img*mask
    masked_image = masked_image.astype(np.uint8)

    #mask = cv.bitwise_not(mask)
    mask = mask[:,:,0].astype(np.uint8)
    mask = 1-mask

    return cv2.inpaint(masked_image, mask, 3, cv.INPAINT_TELEA)


#%%
if __name__=="__main__":
    mask = cv2.imread("./Input/Inpainting/mountain_2_mask.jpg")
    img=cv2.imread("./Input/Inpainting/mountain_2.jpg")

    mask = 1-mask/255

    for i in range(mask.shape[0]):
        for j in range(mask.shape[1]):
            if mask[i,j,0]<0.1:

                for x in range(i):
                    if mask[i-x,j,0]>=0.9:
                        up_colour=img[i-x,j,:]
                        up_distance = x
                        break

                for y in range(mask.shape[0]-i):
                    if mask[i+y,j,0]>=0.9:
                        down_colour=img[i-x,j,:]
                        down_distance = y
                        break

                for z in range(j):
                    if mask[i,j-z,0]>=0.9:
                        right_colour=img[i,j-z,:]
                        right_distance = z

                        break

                for w in range(mask.shape[1]-j):
                    if mask[i,j+w,0]>=0.9:
                        left_colour=img[i,j+w,:]
                        left_distance = w
                        break

                total_distance = left_distance + right_distance + up_distance + down_distance

                img[i,j] =  (left_distance/total_distance) * left_colour + (right_distance/total_distance) * right_colour + (up_distance/total_distance) * up_colour + (down_distance/total_distance) * down_colour
