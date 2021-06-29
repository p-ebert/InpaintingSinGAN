# -*- coding: utf-8 -*-
"""
Created on Tue Aug  4 17:59:00 2020

@author: lodado
"""

import os
import matplotlib.pyplot as plt


def graph(X=[], Y=[], savefolder ='./', typeof='PSNR', Xlab='epoches'):
    plt.figure()
    
    
    

    
    plt.title('graph')
    
    coloring = ''
    if(typeof=='PSNR'):
        coloring = 'blue'
    else:
        coloring = 'red'
   
    plt.xlabel(Xlab, labelpad=int(1))
    plt.ylabel(typeof)
    plt.plot(Y,label=typeof, color='black', marker='o', 
             markersize =3, markerfacecolor='black')
    
    plt.plot(X, label=typeof, color=coloring, marker='o', 
             markersize =3, markerfacecolor=coloring)
    
    plt.savefig(savefolder+typeof+'.jpg')
    plt.legend()
    plt.close()
