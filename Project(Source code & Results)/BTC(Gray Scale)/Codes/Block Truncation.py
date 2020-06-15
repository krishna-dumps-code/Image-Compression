"""
Created on Sun May 31 20:00:16 2020
@author: Krishna
"""
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import pandas as pd
import math
import seaborn as sns

def BTC(img_name,m):
    
    path = 'D:\\standard_test_images\\'+img_name

    img1 = cv2.imread(path)
    #conversion to graylevel image
    gray1 = cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
    img_arr = np.array(gray1)
    r,c = np.shape(img_arr)
    op_arr = np.zeros((r,c))
    bitmap = np.zeros((r,c),dtype = 'uint8')
   
    #defining block truncation compression function
    def compress(img_arr,m,r,c,bitmap,op_arr):
        x = 0
        y = 0
        m2 = m**2
        while x < r:
            while y < c:
                mean = 0
                std = 0
                q = 0
                #calculating mean(first moment)
                for i in range(x,x+m):
                    for j in range(y,y+m):
                        mean += img_arr[i][j]/m2
                
                #calculating standard deviation(second moment)
                for i in range(x,x+m):
                    for j in range(y,y+m):
                        if img_arr[i][j] >= mean:
                            q += 1
                            std += (img_arr[i][j]-mean)**2
                std = (std/m2)**0.5
                #when all gray values in a window are same
                if q == m2:
                    y += m
                    continue
                #calculating Low & High values for output image
                low = mean - ((q/(m2-q))**0.5)*std
                high = mean + (((m2-q)/q)**0.5)*std
    
                #assigning the low and high values to bitmap(Low to 0 and High to 1)
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
        
        return bitmap, op_arr
    
    #calling compress function inside BTC function
    a,b = compress(img_arr,m,r,c,bitmap,op_arr)

    #using matplotlib to display images
    img_temp = [gray1,b,a]               
    titles = ['original gray image','output image','bitmap image']
    for i in range(len(img_temp)):
        plt.imshow(img_temp[i],cmap = 'gray')
        plt.xticks([])
        plt.yticks([])
        plt.title(titles[i])
        plt.show()
        
    #saving the images to a directory
    cv2.imwrite('C:\\Users\\DELL\\Desktop\\Project_codes\\BTC_'+img_name+str(m)+'.png',b)
    #cv2.imwrite('C:\\Users\\DELL\\Desktop\\Project_codes\\BTC_'+img_name+'bitmap'+str(m)+'.png',a)

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
    
    #calculating compression ratio
    original_img_size = os.path.getsize('D:\\standard_test_images\\'+img_name)
    compressed_img_size = os.path.getsize('C:\\Users\\DELL\\Desktop\\Project_codes\\BTC_'+img_name+str(m)+'.png')
    CR = original_img_size/compressed_img_size
    
    return MSE, PSNR, CR


if __name__ == '__main__':
    img_name = input('Enter image name:')
    Window_sizes = []
    MSE_values = []
    PSNR_values = []
    CR_values = []
    for _ in range(int(input('Enter no. of window sizes you want to operate on:'))):
        m = int(input('Enter window size(e.g. 2/4/8/..)'))
        Window_sizes.append(m)
        mse,psnr,cr = BTC(img_name,m)
        MSE_values.append(mse)
        PSNR_values.append(psnr)
        CR_values.append(cr)
    print(Window_sizes)
    print(MSE_values)
    print(PSNR_values)
    
    #placing all values in a pandas dataframe and saving it
    df = pd.DataFrame({'Window Size':Window_sizes,'MSE':MSE_values,'PSNR':PSNR_values,'Compression Ratio':CR_values})
    df.to_csv('C:\\Users\\DELL\\Desktop\\Project_codes\\'+img_name+'.csv')
    
    #plotting windowm size vs MSE, PSNR, CR and saving the plots
    mse_plot = sns.lineplot(x = 'Window Size',y = 'MSE',color = 'blue',marker = 'o',data = df)
    fig1 = mse_plot.get_figure()
    plt.title('Mean square error plot')
    plt.show()
    fig1.savefig('MSE'+img_name+'.png')
    
    psnr_plot = sns.lineplot(x = 'Window Size',y = 'PSNR',color = 'red',marker = 'o',data = df)
    fig2 = psnr_plot.get_figure()
    plt.title('Peak signal to noise ratio plot')
    plt.show()
    fig2.savefig('PSNR'+img_name+'.png')
    
    cr_plot = sns.lineplot(x = 'Window Size',y = 'Compression Ratio',color = 'green',marker = 'o',data = df)
    fig3 = cr_plot.get_figure()
    plt.title('Compression ratio plot')
    plt.show()
    fig3.savefig('CR'+img_name+'.png')  
    
    sns.lineplot(x = 'Window Size',y = 'Compression Ratio',color = 'orange',marker = 'o',data = df)
    plt.legend(labels=['CR'])
    ax2 = plt.twinx()
    comp = sns.lineplot(x = 'Window Size',y = 'PSNR',color = 'aqua',marker = 'o',data = df)
    fig4 = comp.get_figure()
    plt.legend(labels=['PSNR'])
    plt.title('CR & PSNR comparision plot')
    plt.show()
    fig4.savefig('CR & PSNR'+img_name+'.png')