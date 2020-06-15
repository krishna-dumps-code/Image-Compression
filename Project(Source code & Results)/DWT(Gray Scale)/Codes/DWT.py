import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import pandas as pd
import pywt
import pywt.data
import seaborn as sns

img_name = 'lena_color_512.tif'
path = 'D:\\standard_test_images\\'+img_name
img = cv2.imread(path)
img1 = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
plt.imshow(img1,cmap='gray')
plt.show()

C = pywt.wavedec2(img1,'db5',mode='periodization',level=2)
idwt = pywt.waverec2(C,'db5',mode='periodization')
idwt = idwt.astype('uint8')
plt.imshow(idwt,cmap='gray')
plt.show()

A2 = C[0]
(H1,V1,D1) = C[-1]
(H2,V2,D2) = C[-2]

plt.figure(figsize=(8,8))
plt.subplot(2,2,1)
plt.imshow(A2,cmap='gray')
plt.subplot(2,2,2)
plt.imshow(H2,cmap='gray')
plt.subplot(2,2,3)
plt.imshow(V2,cmap='gray')
plt.subplot(2,2,4)
plt.imshow(D2,cmap='gray')
plt.show()

arr,c_slices = pywt.coeffs_to_array(C)
plt.imshow(arr,cmap='gray')
plt.show()
'''coeff = pywt.dwt2(img1,'db3',mode='periodization')
A,(H,V,D) = coeff

img2 = pywt.idwt2(coeff,'db3',mode='periodization')
img2 = img2.astype('uint8')
plt.imshow(img2,cmap='gray')
plt.show()
A = A.astype('int32')
plt.subplot(2,2,1)
plt.imshow(A,cmap='gray')
plt.subplot(2,2,2)
plt.imshow(H,cmap='gray')
plt.subplot(2,2,3)
plt.imshow(V,cmap='gray')
plt.subplot(2,2,4)
plt.imshow(D,cmap='gray')
plt.show()

#A = A.astype('uint8')
coeff2 = pywt.dwt2(A,'db3',mode='periodization')
A1,(H1,V1,D1) = coeff2

img3 = pywt.idwt2(coeff2,'db3',mode='periodization')
img3 = img3.astype('int32')
plt.imshow(img3,cmap='gray')
plt.show()

plt.subplot(2,2,1)
plt.imshow(A1,cmap='gray')
plt.subplot(2,2,2)
plt.imshow(H1,cmap='gray')
plt.subplot(2,2,3)
plt.imshow(V1,cmap='gray')
plt.subplot(2,2,4)
plt.imshow(D1,cmap='gray')
plt.show()'''
