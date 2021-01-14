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
                        right_distance = 1/z

                        break

                for w in range(mask.shape[1]-j):
                    if mask[i,j+w,0]>=0.9:
                        left_colour=img[i,j+w,:]
                        left_distance = 1/w
                        break

                total_distance = w * z * x * y /(x*y*w+y*w*z+x*w*z+x*y*z) #+ up_distance + down_distance

                img[i,j] =  ((1/w)*total_distance) * left_colour + ((1/z)*total_distance)* right_colour + ((1/x)*total_distance) * up_colour +((1/y)*total_distance) * down_colour

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
    mask = cv2.imread("./Input/Inpainting/image_5M_mask.jpeg")
    img=cv2.imread("./Input/Inpainting/image_5M.jpeg")

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
                        right_distance = 1/z

                        break

                for w in range(mask.shape[1]-j):
                    if mask[i,j+w,0]>=0.9:
                        left_colour=img[i,j+w,:]
                        left_distance = 1/w
                        break

                total_distance = w * z * x * y /(x*y*w+y*w*z+x*w*z+x*y*z) #+ up_distance + down_distance

                img[i,j] =  ((1/w)*total_distance) * left_colour + ((1/z)*total_distance)* right_colour + ((1/x)*total_distance) * up_colour +((1/y)*total_distance) * down_colour



    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
