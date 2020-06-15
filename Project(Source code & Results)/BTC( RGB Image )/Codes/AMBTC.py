"""
Created on Thu Jun  4 16:49:28 2020
@author: Krishna
"""
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import pandas as pd
import math
import seaborn as sns


def AMBTC(img_name,mi,mq):
    
    path = 'D:\\standard_test_images\\'+img_name

    img = cv2.imread(path)
    img1 = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    r,c,d = img1.shape
    
    #converting RGB to NTSC YIQ channel
    Y = np.zeros((r,c), dtype = np.uint8)
    I = np.zeros((r,c), dtype = np.uint8)
    Q = np.zeros((r,c), dtype = np.uint8)
    for i in range(r):
        for j in range(c):
            R = img1[i,j,0]
            G = img1[i,j,1]
            B = img1[i,j,2]
            #RGB -> YIQ
            Y[i,j] = int((0.299*R) + (0.587*G) + (0.114*B))
            I[i,j] = int((0.596*R) - (0.275*G) - (0.321*B))
            Q[i,j] = int((0.212*R) - (0.523*G) + (0.311*B))
    
    plt.imshow(img1[:,:,0])
    plt.show()
    plt.imshow(img1[:,:,1])
    plt.show()
    plt.imshow(img1[:,:,2])
    plt.show()        
    plt.imshow(Y)
    plt.show()
    plt.imshow(I)
    plt.show()
    plt.imshow(Q)
    plt.show()
    #merging individual channels into one        
    yiq = cv2.merge((Y,I,Q))
    plt.imshow(yiq)
    plt.show()
    
    #defining AMBTC compression function
    def color_compress(channel,m,r,c,op_arr):
        x = 0
        y = 0
        m2 = m**2
        while x < r:
            while y < c:
                mean = 0
                High = 0
                Low = 0
                q = 0
                #calculating mean(first moment)
                for i in range(x,x+m):
                    for j in range(y,y+m):
                        mean += channel[i,j]/m2
                
                #calculating absolute moments
                for i in range(x,x+m):
                    for j in range(y,y+m):
                        if channel[i,j] >= mean:
                            High += channel[i,j]
                            q += 1
                        else:                
                            Low += channel[i,j]
                #when all pixel values in a window are same
                if q == m2:
                    y += m
                    continue
                h = High/q
                l = Low/((m**2)-q)
                
                #assigning the low and high values to bitmap(Low to 0 and High to 1)
                for i in range(x,x+m):
                    for j in range(y,y+m):
                        if channel[i,j] >= mean:
                            op_arr[i,j] = h
                        else:
                            op_arr[i,j] = l
                        
                y += m
            y = 0
            x += m    
                        
        return op_arr
    
    IO = color_compress(I,mi,r,c,np.zeros((r,c),dtype = 'uint8'))
    QO = color_compress(Q,mq,r,c,np.zeros((r,c),dtype = 'uint8'))
    
    plt.imshow(IO)
    plt.show()
    plt.imshow(QO)
    plt.show()
    #merging compressed Y,I,Q channels
    yiqo = cv2.merge((Y,IO,QO))
    plt.imshow(yiqo)
    plt.show()
    
    #converting compressed YIQ to RGB image
    R = np.zeros((r,c),dtype = 'uint8')
    G = np.zeros((r,c),dtype = 'uint8')
    B = np.zeros((r,c),dtype = 'uint8')
    for i in range(r):
        for j in range(c):
            Y1 = Y[i,j]
            IO1 = IO[i,j]
            QO1 = QO[i,j]
            #YIQ -> RGB
            R[i,j] = int((Y1) + (0.956*IO1) + (0.619*QO1))
            G[i,j] = int((Y1) - (0.272*IO1) - (0.647*QO1))
            B[i,j] = int((Y1) - (1.106*IO1) + (1.703*QO1))
    #merging individual channels into one  
    plt.imshow(R)
    plt.show()
    plt.imshow(G)
    plt.show()
    plt.imshow(B)
    plt.show()      
    RGBO = cv2.merge((R,G,B))
    plt.imshow(RGBO)
    plt.show()
    
    '''#Displaying results
    img_temp = [img1,RGBO,yiq,yiqo]               
    titles = ['original color image','output color image','yiq input image','yiq output image']
    for i in range(len(img_temp)):
        plt.imshow(img_temp[i])
        plt.xticks([])
        plt.yticks([])
        plt.title(titles[i])
        plt.show()
    
    RGBO2 = cv2.cvtColor(RGBO,cv2.COLOR_RGB2BGR)
    #saving the images to a directory
    cv2.imwrite('C:\\Users\\DELL\\Desktop\\Project_codes\\AMBTC_'+img_name+str(mi)+str(mq)+'.png',RGBO2)'''
        
if __name__ == '__main__':
    AMBTC('4.1.05.tiff',4,8)