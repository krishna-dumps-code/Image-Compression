import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import pandas as pd
import math
import seaborn as sns


def color_BTC(img_name,m):
    
    path = 'D:\\standard_test_images\\'+img_name

    img = cv2.imread(path)
    img1 = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    r,c,d = img1.shape
    
    R = img1[:,:,0]
    G = img1[:,:,1]
    B = img1[:,:,2]
    
    #defining BTC compression function
    def color_compress(channel,m,r,c,op_arr):
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
                        mean += channel[i][j]/m2
                
                #calculating absolute moments
                for i in range(x,x+m):
                    for j in range(y,y+m):
                        if channel[i][j] >= mean:
                            q += 1
                            std += (channel[i][j]-mean)**2
                std = (std/m2)**0.5
                #when all pixel values in a window are same
                if q == m2:
                    y += m
                    continue
                #calculating Low & High values for output image
                low = mean - ((q/(m2-q))**0.5)*std
                high = mean + (((m2-q)/q)**0.5)*std
    
                #assigning the low and high values to bitmap(Low to 0 and High to 1)
                for i in range(x,x+m):
                    for j in range(y,y+m):
                        if channel[i][j] >= mean:
                            op_arr[i][j] = high
                        else:
                            op_arr[i][j] = low
                
                        
                y += m
            y = 0
            x += m    
                        
        return op_arr
    
    RO = color_compress(R,m,r,c,np.zeros((r,c),dtype = 'uint8'))
    GO = color_compress(G,m,r,c,np.zeros((r,c),dtype = 'uint8'))
    BO = color_compress(B,m,r,c,np.zeros((r,c),dtype = 'uint8'))
    
    #merging compressed RGB channels
    RGBO = cv2.merge((RO,GO,BO))
    
    #Displaying results
    img_temp = [img1,RGBO]               
    titles = ['original color image','output color image']
    for i in range(len(img_temp)):
        plt.imshow(img_temp[i])
        plt.xticks([])
        plt.yticks([])
        plt.title(titles[i])
        plt.show()
    
    RGBO2 = cv2.cvtColor(RGBO,cv2.COLOR_RGB2BGR)
    #saving the images to a directory
    cv2.imwrite('C:\\Users\\DELL\\Desktop\\Project_codes\\color_BTC_'+img_name+str(m)+'.png',RGBO2)
    
    i = 0
    j = 0
    sq_error = 0
    while i < r:
        while j < c:
            sq_error += (R[i][j] - RO[i][j])**2
            sq_error += (G[i][j] - GO[i][j])**2
            sq_error += (B[i][j] - BO[i][j])**2
            j += 1
        j = 0
        i += 1
    
    MSE = sq_error/(3*r*c)

    PSNR = np.log10(255*765/abs(MSE))
    
    original_img_size = os.path.getsize('D:\\standard_test_images\\'+img_name)
    compressed_img_size = os.path.getsize('C:\\Users\\DELL\\Desktop\\Project_codes\\color_BTC_'+img_name+str(m)+'.png')
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
        mse,psnr,cr = color_BTC(img_name,m)
        MSE_values.append(abs(mse))
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
