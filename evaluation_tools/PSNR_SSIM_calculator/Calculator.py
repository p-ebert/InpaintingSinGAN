# -*- coding: utf-8 -*-
"""
Created on Tue Aug  4 14:31:20 2020

@author: lodado
"""
import os
import PSNR
import argparse
import graphing
import cv2

parser = argparse.ArgumentParser(description='test PSNR AND SSIM')

# path
parser.add_argument('--folders', default="./Input/savefolder/",  help="put folders here")
parser.add_argument('--HR', default="./Input/HR/", help="put HR(original) images here") #default images from CelebA

#arguments
parser.add_argument('--X', default="checkpoint", help="X axis")
parser.add_argument('--picofname', default="vail_pics", help="discription of pictures")
opt = parser.parse_args()

def search(txt):
    typeoffile = ['.txt','.jpg','.png','.bmp']
    print(txt)

    for str in typeoffile:
        if txt.find(str)!=-1:
            return False

    return True

def calALL():

    path = opt.folders

    ps = []
    ss = []



    list_PSNR =[]
    list_SSIM =[]

    file_list = os.listdir(path)

    f = open(path+"average.txt", 'w')

    f.write('average (PSNR,SSIM)\n')
    f.close()

    file_list = sorted(file_list, key = len)

    for Npath in file_list:

        newPath = path+Npath+'/'
        averPSNR = 0
        averSSIM = 0
        count = 0

        if not (search(newPath)):
            continue
        else:
            f = open(newPath+"PSNR.txt", 'w')
            f.close()

            f = open(newPath+"SSIM.txt", 'w')
            f.close()

            print("*folder* = "+newPath)

            newfile_list = os.listdir(newPath)



            list_PSNR =[]
            list_SSIM =[]

            for i in newfile_list:

                if(i[-9:]=='epoch.jpg'):
                    os.remove(newPath+'epoch.jpg')
                    continue

                if(i[-3:]=='txt' or i[-8:]=='SSIM.jpg' or i[-8:]=='PSNR.jpg'):
                    continue

                img = i
                LR = newPath + img
                print('open '+ LR)

                HR = opt.HR
                HR = HR + img
                print('open '+HR)

                #biimg = cv2.imread(LR)
                #biimg = cv2.resize(biimg, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)
                #cv2.imwrite(opt.HR+'../bicubic/'+i ,biimg)

                #LR = opt.HR+'../bicubic/'+i
                #print(LR)
                try:
                    one, two = PSNR.cal_PSNRandSSIM(HR, LR)
                except:
                    print('skip')
                    continue
                print(f"{one} is PSNR, {two} is SSIM")

                f = open(newPath+"PSNR.txt", 'a')
                f.write(img+f" PSNR is {one}\n")
                f.close()

                f = open(newPath+"SSIM.txt", 'a')
                f.write(img+f" SSIM is {two}\n")
                f.close()

                list_PSNR.append(one)
                list_SSIM.append(two)

                #well.. im not good at python..
                averPSNR += one
                averSSIM += two

                count +=1
                print('=============================')

            graphing.graph(list_PSNR, [], newPath,'PSNR',opt.picofname)
            graphing.graph(list_SSIM, [], newPath,'SSIM',opt.picofname)


            if(count==0):
                averPSNR = 0
                averSSIM = 0
            else:
                averPSNR /= count
                averSSIM /= count

            print("{} is average PSNR, {} is average SSIM".format(averPSNR,averSSIM))

            ps.append(averPSNR)
            ss.append(averSSIM)

            graphing.graph(ps, [], path,'PSNR',opt.X)
            graphing.graph(ss, [], path,'SSIM',opt.X)

            f = open(path+"average.txt", 'a')
            f.write("{} : {} is average PSNR, {} is average SSIM\n".format(Npath,averPSNR,averSSIM))
            f.close()

if __name__ == '__main__':
    calALL()
