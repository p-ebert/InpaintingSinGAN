import cv2
import numpy as np
import matplotlib.pyplot as plt

def mean_colour_rectangular(masked_img, mask):
    img=masked_img.copy()
    coordinates=np.where(mask==0)

    coordinates[1][0::]

    x1=110#coordinates[0][0]
    x2=125#coordinates[0][-1]+3
    y1=70#coordinates[1][0]
    y2=85#coordinates[1][-1]+3

    for i in range(x1,x2):
        for j in range(y1,y2):
            img[i,j,:]=((y2-j)/(y2-y1))*img[i,y1-1,:]+((j-y1)/(y2-y1))*img[i,y2+1,:]

    return img




if __name__=="__main__":

    masked_img = cv2.imread("./Input/Inpainting/nature_edges.jpg")
    mask = cv2.imread("./Input/Inpainting/nature_edges_mask.jpg")

    mask=1-mask/255

    img=mean_colour_rectangular(masked_img, mask)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.show()
    #cv2.imwrite("test.png",mean_colour_rectangular(masked_img, mask))
