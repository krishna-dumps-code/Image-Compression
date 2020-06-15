"""
Created on Sun May 31 20:00:16 2020
"""
import numpy as np
import cv2
import matplotlib.pyplot as plt
import math

img_name = '7.1.06.tiff'
path = 'D:\\standard_test_images\\'+img_name

img1 = cv2.imread(path)
gray1 = cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
img_arr = np.array(gray1)
r,c = np.shape(img_arr)
op_arr = np.zeros((r,c))
bitmap = np.zeros((r,c))
m = int(input('ENTER WINDOW SIZE(e.g:4/8/16/..)'))

def btc(img_arr,m,r,c,bitmap,op_arr):
    x = 0
    y = 0
    q = 0
    while x < r:
        while y < c:
            total = 0
            std = 0
            for i in range(x,x+m):
                for j in range(y,y+m):
                    total += img_arr[i][j]
            mean = total/(m**2)
            for i in range(x,x+m):
                for j in range(y,y+m):
                    if img_arr[i][j] >= mean:
                        q += 1
                    std += (mean - img_arr[i][j])**2
            std = (std**0.5)/m
            low = mean - ((q/(m**2-q))**0.5)*std
            high = mean + (((m**2-q)/q)**0.5)*std
            for i in range(x,x+m):
                for j in range(y,y+m):
                    if img_arr[i][j] >= mean:
                        bitmap[i][j] = 1
                        op_arr[i][j] = high
                    else:
                        op_arr[i][j] = low
                        
            y += m
        y = 0
        x += m    
    bitmap = bitmap.astype('int32')
    return bitmap, op_arr

a,b = btc(img_arr,m,r,c,bitmap,op_arr)

b1 = b.astype('int32')
img_temp = [gray1,b,a]               
titles = ['original gray image','output image','bitmap image']
for i in range(len(img_temp)):
    plt.imshow(img_temp[i],cmap = 'gray')
    plt.xticks([])
    plt.yticks([])
    plt.title(titles[i])
    plt.show()
cv2.imwrite('C:\\Users\\DELL\\Desktop\\Project_codes\\BTC_desert'+str(m)+'.png',b)
#cv2.imwrite('C:\\Users\\DELL\\Desktop\\Project_codes\\BTC_cameraman(rounded)'+str(m)+'.png',b1)
cv2.imwrite('C:\\Users\\DELL\\Desktop\\Project_codes\\BTC_desert_bitmap'+str(m)+'.png',a)

#Computing MSE and PSNR (image quality parameters)

i = 0
j = 0
sq_error = 0
while i < r:
    while j < c:
        sq_error += (gray1[i][j] - b[i][j])**2
        j += 1
    j = 0
    i += 1
    
MSE = sq_error/(r*c)

PSNR = 10*math.log10(255*255/MSE)

print(MSE,PSNR)


