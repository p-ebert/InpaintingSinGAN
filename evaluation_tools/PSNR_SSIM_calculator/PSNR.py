# -*- coding: utf-8 -*-
"""
Created on Fri Jul 31 18:47:00 2020

@author: lodado

#*codes from other sites
"""
from skimage.measure import compare_ssim
from math import log10, sqrt 
import cv2 
import numpy as np 


#https://www.geeksforgeeks.org/python-peak-signal-to-noise-ratio-psnr/
def PSNR1(original, compressed):

    mse = np.mean((original - compressed) ** 2) 
    if(mse == 0):  # MSE is zero means no noise is present in the signal . 
                  # Therefore PSNR have no importance. 
        return 100
    max_pixel = 255.0
    #psnr = 20 * (log10(max_pixel)) - 10*(log10((mse)))

    return psnr


def PSNR(original, compressed):

    #ori = original.astype(np.float64)/255
    #com = compressed.astype(np.float64)/255
    psnr = cv2.PSNR(original,compressed)
    return psnr 


#https://ourcodeworld.com/articles/read/991/how-to-calculate-the-structural-similarity-index-ssim-between-two-images-with-python
def SSIM(original, compressed): 
    # Convert the images to grayscale
    grayA = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
    grayB = cv2.cvtColor(compressed, cv2.COLOR_BGR2GRAY)

    # Compute the Structural Similarity Index (SSIM) between the two
    # images, ensuring that the difference image is returned
    (score, diff) = compare_ssim(grayA, grayB, full=True)
    diff = (diff * 255).astype("uint8") 
    # 6. You can print only the score if you want
    print(f"SSIM value is {score}")
    return score    

def cal_PSNR(origin,comp): 
     value = PSNR(origin, comp) 
     print(f"PSNR value is {value} dB") 
     return value

def cal_PSNRandSSIM(origin, comp):
    original = cv2.imread(origin) 
    compressed = cv2.imread(comp, 1)

    value_PSNR = cal_PSNR(original, compressed)
    value_SSIM = SSIM(original,compressed)
    return [value_PSNR,value_SSIM]



    

