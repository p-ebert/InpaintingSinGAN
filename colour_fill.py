import cv2
import numpy as np
import matplotlib.pyplot as plt

def mean_colour_rectangular(masked_img, mask):
    img=masked_img.copy()
    coordinates=np.where(mask==0)

    coordinates[1][0::]

    x1=coordinates[0][0]
    x2=coordinates[0][-1]+3
    y1=coordinates[1][0]
    y2=coordinates[1][-1]+3

    x1=110
    x2=125
    y1=70
    y2=85

    for i in range(x1,x2):
        for j in range(y1,y2):
            img[i,j,:]=((y2-j)/(y2-y1))*img[i,y1-1,:]+((j-y1)/(y2-y1))*img[i,y2+1,:]
    return img

if __name__=="__main__":
    x1=110
    x2=125
    y1=70
    y2=85

    edges = cv2.imread("./Input/Inpainting/nature_edges_propagated.jpg")
    mask = cv2.imread("./Input/Inpainting/nature_edges_mask.jpg")
    img=cv2.imread("./Input/Inpainting/nature_real.jpg")
    edges

    coloured=mean_colour_rectangular(img, mask)

    mask=1-mask/255

    #edges[x2-1:x2,y1:y2,:]
    edgel=edges[x1:x2,y1:y1+1,:]
    edger=edges[x1:x2,y2-1:y2,:]
    edget=edges[x1:x1+1,y1:y2,:]
    edgeb=edges[x2-1:x2,y1:y2,:]
    edge_b_bool=(edgeb>=240)[:,:,0]
    edge_t_bool=(edget>=240)[:,:,0]
    edge_l_bool=(edgel>=240)[:,:,0]
    edge_r_bool=(edger>=240)[:,:,0]



    edge_b_markers=[img[x2,y1+i] for i in range(len(edge_b_bool[0])) if edge_b_bool[0][i]]
    edge_t_markers=[img[x1-1,y1+i] for i in range(len(edge_t_bool[0])) if edge_t_bool[0][i]]
    edge_l_markers=[img[x1+i,y1-1] for i in range(len(edge_l_bool[0])) if edge_l_bool[0][i]]
    edge_r_markers=[img[x1+i,y2] for i in range(len(edge_r_bool[0])) if edge_r_bool[0][i]]

    n_b=len(edge_b_markers)
    n_t=len(edge_t_markers)
    n_l=len(edge_l_markers)
    n_r=len(edge_r_markers)

    mean_marker=(np.mean(np.array(edge_b_markers), axis=0)*len(edge_b_markers)+np.mean(np.array(edge_t_markers), axis=0)*len(edge_t_markers))/(n_b+n_t)
    mean_marker

    bool=np.tile(False,img.shape)
    mask_bool=edges[x1:x2,y1:y2,:]>=127
    bool[x1:x2,y1:y2,:]=mask_bool

    b, g, r=cv2.split(coloured)
    b[bool[:,:,0]]=255#mean_marker[0]
    g[bool[:,:,0]]=255#mean_marker[1]
    r[bool[:,:,0]]=255#mean_marker[2]

    colour_edges=cv2.merge([b, g,r])

    plt.imshow(cv2.cvtColor(colour_edges, cv2.COLOR_BGR2RGB))
