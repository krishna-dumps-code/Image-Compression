import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import pandas as pd
from math import cos,sqrt,pi
import seaborn as sns

def DCT_main(img_name,m,q_level):

    path = 'D:\\standard_test_images\\'+img_name
    img1 = cv2.imread(path)
    #conversion to graylevel image
    gray1 = cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
    img_arr = np.float64(gray1)
    plt.imshow(img_arr,cmap='gray')
    plt.show()
    r,c = img_arr.shape
    img_arr1 = img_arr - 128
    dct2 = cv2.dct(img_arr1)
    dct2 = dct2.astype('int32')
    qnt_arr = np.zeros((r,c),np.int32)
    deqnt_arr = np.zeros((r,c),np.int32)

    def matrix(q_matrix,q_level):
        if q_level == 50:
            return q_matrix
        elif q_level < 50:
            q = 50/q_level
            for i in range(8):
                for j in range(8):
                    q_matrix[i,j] = q_matrix[i,j]*q
                    if q_matrix[i,j] > 255:
                        q_matrix[i,j] = 255
        else:
            q = (100-q_level)/50
            for i in range(8):
                for j in range(8):
                    q_matrix[i,j] = q_matrix[i,j]*q
        return q_matrix

    def quantization(dct_mask,q_matrix,qnt_mask,deqnt_mask):                    
        for i in range(8):
            for j in range(8):
                qnt_mask[i,j] = dct_mask[i,j]/q_matrix[i,j]
            
        deqnt_mask = qnt_mask*q_matrix
    
        return qnt_mask,deqnt_mask

    x1 = 0
    y1 = 0
    q_mat = np.array([[16,11,10,16,24,40,51,61],[12,12,14,19,26,58,60,55],
                         [14,13,16,24,40,57,69,56],[14,17,22,29,51,87,80,62],
                         [18,22,37,56,68,109,103,77],[24,35,55,64,81,104,113,92],
                         [49,64,78,87,103,121,120,101],[72,92,95,98,112,100,103,99]],dtype=np.int32)
    q_matrix = matrix(q_mat,q_level)
    print(q_matrix)
    while x1 < r:
        x2 = x1 + m
        while y1 < c:
            y2 = y1 + m
            qnt,deqnt = quantization(dct2[x1:x2,y1:y2],q_matrix,np.zeros((m,m),dtype=np.int32),np.zeros((m,m),dtype=np.int32))
            qnt_arr[x1:x2,y1:y2] = qnt
            deqnt_arr[x1:x2,y1:y2] = deqnt
            y1 += m
        y1 = 0
        x1 += m

    qnt_arr = qnt_arr.astype('int32')
    idct = cv2.idct(deqnt_arr.astype('float32'))
    idct = idct + 128
    idct = idct.astype('int32')
    dct2 = dct2.astype('int32')
    img_temp = [dct2,qnt_arr,deqnt_arr,idct]               
    titles = ['dct image','quantized image','dequantized image','output image']
    for i in range(len(img_temp)):
        plt.imshow(img_temp[i],cmap = 'gray')
        plt.xticks([])
        plt.yticks([])
        plt.title(titles[i])
        plt.show()
        
        #saving the images to a directory
        cv2.imwrite('C:\\Users\\DELL\\Desktop\\Project_codes\\DCTwithCV_'+img_name+titles[i]+str(m)+str(q_level)+'.png',img_temp[i])
    
    #Computing MSE and PSNR (image quality parameters)
    i = 0
    j = 0
    sq_error = 0
    while i < r:
        while j < c:
            sq_error += (gray1[i][j] - idct[i][j])**2
            j += 1
        j = 0
        i += 1
    
    MSE = sq_error/(r*c)

    PSNR = 10*np.log10(255*255/MSE)
    
    #calculating compression ratio
    original_img_size = os.path.getsize('D:\\standard_test_images\\'+img_name)
    compressed_img_size = os.path.getsize('C:\\Users\\DELL\\Desktop\\Project_codes\\DCTwithCV_'+img_name+'output image'+str(m)+str(q_level)+'.png')
    CR = original_img_size/compressed_img_size
    
    return MSE, PSNR, CR

if __name__ == '__main__':
    img_name = input('Enter image name:')
    q_values = []
    MSE_values = []
    PSNR_values = []
    CR_values = []
    m = 8
    for _ in range(int(input('On how many quality values you want to perform compression: '))):
        q_level = int(input('Enter quality value(e.g. 20/50/80/..): '))
        q_values.append(q_level)
        mse,psnr,cr = DCT_main(img_name,m,q_level)
        MSE_values.append(mse)
        PSNR_values.append(psnr)
        CR_values.append(cr)
    print(q_values)
    print(MSE_values)
    print(PSNR_values)
    
    #placing all values in a pandas dataframe and saving it
    df = pd.DataFrame({'Quality(%)':q_values,'MSE':MSE_values,'PSNR':PSNR_values,'Compression Ratio':CR_values})
    df.to_csv('C:\\Users\\DELL\\Desktop\\Project_codes\\DCT'+img_name+'.csv')
    
    #plotting windowm size vs MSE, PSNR, CR and saving the plots
    mse_plot = sns.lineplot(x = 'Quality(%)',y = 'MSE',color = 'aqua',marker = 'o',data = df)
    fig1 = mse_plot.get_figure()
    plt.title('Mean square error plot')
    plt.show()
    fig1.savefig('MSE'+img_name+'.png')
    
    psnr_plot = sns.lineplot(x = 'Quality(%)',y = 'PSNR',color = 'red',marker = 'o',data = df)
    fig2 = psnr_plot.get_figure()
    plt.title('Peak signal to noise ratio plot')
    plt.show()
    fig2.savefig('PSNR'+img_name+'.png')
    
    cr_plot = sns.lineplot(x = 'Quality(%)',y = 'Compression Ratio',color = 'green',marker = 'o',data = df)
    fig3 = cr_plot.get_figure()
    plt.title('Compression ratio plot')
    plt.show()
    fig3.savefig('CR'+img_name+'.png')  
    
    sns.lineplot(x = 'Quality(%)',y = 'Compression Ratio',color = 'orange',marker = 'o',data = df)
    plt.legend(labels=['CR'])
    ax2 = plt.twinx()
    comp = sns.lineplot(x = 'Quality(%)',y = 'PSNR',color = 'purple',marker = 'o',data = df)
    fig4 = comp.get_figure()
    plt.legend(labels=['PSNR'])
    plt.title('CR & PSNR comparision plot')
    plt.show()
    fig4.savefig('CR & PSNR'+img_name+'.png')