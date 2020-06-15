import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import pandas as pd
from math import cos,sqrt,pi
import seaborn as sns

#main function
def DCT_color(img_name,m,q_level):

    path = 'D:\\standard_test_images\\'+img_name
    img1 = cv2.imread(path)
    img2 = cv2.cvtColor(img1,cv2.COLOR_BGR2RGB)
    plt.imshow(img2)
    plt.show()  
    #converting BGR channels to YCRCB channels
    ycrcb = cv2.cvtColor(img1,cv2.COLOR_BGR2YCR_CB)
    y = ycrcb[:,:,0]
    cr1 =  ycrcb[:,:,1]
    cb1 =  ycrcb[:,:,2]
    #sub sampling cr & cb channels to half
    h,w = cr1.shape
    '''cr = cv2.resize(cr1,(h//2,w//2))
    cb = cv2.resize(cb1,(h//2,w//2))'''
    y_arr = np.float32(y)
    cr_arr = np.float32(cr1)
    cb_arr = np.float32(cb1)
    q_y = np.array([[16,11,10,16,24,40,51,61],[12,12,14,19,26,58,60,55],
                         [14,13,16,24,40,57,69,56],[14,17,22,29,51,87,80,62],
                         [18,22,37,56,68,109,103,77],[24,35,55,64,81,104,113,92],
                         [49,64,78,87,103,121,120,101],[72,92,95,98,112,100,103,99]],dtype=np.int32)
    q_c = np.array([[17,18,24,47,99,99,99,99],[18,21,26,66,99,99,99,99],
                         [24,26,56,99,99,99,99,99],[47,66,99,99,99,99,99,99],
                         [99,99,99,99,99,99,99,99],[99,99,99,99,99,99,99,99],
                         [99,99,99,99,99,99,99,99],[99,99,99,99,99,99,99,99]],dtype=np.int32)
    
    #compression function
    def DCT_main(img_name,img_arr,m,q_matrix,q_level):
        r,c = img_arr.shape
        img_arr = img_arr - 128
        dct2 = cv2.dct(img_arr)
        #dct2 = dct2.astype('int32')
        qnt_arr = np.zeros((r,c),dtype=np.int32)
        deqnt_arr = np.zeros((r,c),dtype=np.int32)

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
        q_mat = matrix(q_matrix,q_level)
        print(q_mat)
        while x1 < r:
            x2 = x1 + m
            while y1 < c:
                y2 = y1 + m
                qnt,deqnt = quantization(dct2[x1:x2,y1:y2],q_mat,np.zeros((m,m),dtype=np.int32),np.zeros((m,m),dtype=np.int32))
                qnt_arr[x1:x2,y1:y2] = qnt
                deqnt_arr[x1:x2,y1:y2] = deqnt
                y1 += m
            y1 = 0
            x1 += m

        #qnt_arr = qnt_arr.astype('int32')
        idct = cv2.idct(deqnt_arr.astype('float32'))
        idct = idct + 128.0
        idct = idct.astype('uint8')
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
            cv2.imwrite('C:\\Users\\DELL\\Desktop\\Project_codes\\DCT_color'+img_name+titles[i]+str(m)+str(q_level)+'.png',img_temp[i])
        
        return idct
    
    y_idct = DCT_main(img_name,y_arr,m,q_y,q_level)
    cr_idct = DCT_main(img_name,cr_arr,m,q_c,q_level)
    cb_idct = DCT_main(img_name,cb_arr,m,q_c,q_level)
    
    #converting CR & CB to original size
    '''cr_idct = cv2.resize(cr_idct,(int(h),int(w)))
    cb_idct = cv2.resize(cb_idct,(int(h),int(w)))
    print(cr_idct.shape)'''
    
    #Computing MSE and PSNR (image quality parameters)
    i = 0
    j = 0
    sq_error_y = 0
    sq_error_cr = 0
    sq_error_cb = 0
    while i < h:
        while j < w:
            sq_error_y += ((y[i][j] - y_idct[i][j])**2)/(h*w)
            sq_error_cr += ((cr1[i][j] - cr_idct[i][j])**2)/(h*w)
            sq_error_cb += ((cb1[i][j] - cb_idct[i][j])**2)/(h*w)
            j += 1
        j = 0
        i += 1
    
    MSE = sq_error_y+sq_error_cr+sq_error_cb

    PSNR = 10*np.log10(255*765/MSE)
    
    #combining y,cr,cb channels
    ycrcbo = cv2.merge((y_idct,cr_idct,cb_idct))
    #ycrcbo = ycrcbo.astype('float32')
    plt.imshow(ycrcbo)
    plt.title('ycrcb output image')
    plt.show()
    bgro = cv2.cvtColor(ycrcbo,cv2.COLOR_YCR_CB2BGR)
    rgbo = cv2.cvtColor(ycrcbo,cv2.COLOR_YCR_CB2RGB)
    #saving final images
    plt.imshow(rgbo)
    plt.title('compressed image')
    plt.show()
    cv2.imwrite('C:\\Users\\DELL\\Desktop\\Project_codes\\DCT_color'+img_name+'compressed image'+str(m)+str(q_level)+'.png',bgro)
    cv2.imwrite('C:\\Users\\DELL\\Desktop\\Project_codes\\DCT_color'+img_name+'coutput ycbcr image'+str(m)+str(q_level)+'.png',ycrcbo)
    #calculating compression ratio
    original_img_size = os.path.getsize('D:\\standard_test_images\\'+img_name)
    compressed_img_size = os.path.getsize('C:\\Users\\DELL\\Desktop\\Project_codes\\DCT_color'+img_name+'compressed image'+str(m)+str(q_level)+'.png')
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
        mse,psnr,cr = DCT_color(img_name,m,q_level)
        MSE_values.append(mse)
        PSNR_values.append(psnr)
        CR_values.append(cr)
    print(q_values)
    print(MSE_values)
    print(PSNR_values)
    
    #placing all values in a pandas dataframe and saving it
    df = pd.DataFrame({'Quality(%)':q_values,'MSE':MSE_values,'PSNR':PSNR_values,'Compression Ratio':CR_values})
    df.to_csv('C:\\Users\\DELL\\Desktop\\Project_codes\\DCT_color'+img_name+'.csv')
    
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

