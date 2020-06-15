import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import pandas as pd
import pywt
import pywt.data
import seaborn as sns

#main DWT fuction
def DWT(img_name,th):
    path = 'D:\\standard_test_images\\'+img_name
    img1 = cv2.imread(path)
    gray1 = cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
    h,w = gray1.shape
    h = int(h)
    w = int(w)
    plt.imshow(gray1,cmap='gray')
    plt.show()
    
    coeff = pywt.dwt2(gray1,'db3',mode='periodization')
    A,(H,V,D) = coeff
    
    img_temp = [A,H,V,D]
    titles = ['Approximated Coefficients(LL)','Horizontal detailed coefficients(LH)',
              'Vertical detailed coefficients(HL)','Diagonal detailed coefficients(HH)']
    for i in range(len(img_temp)):
        plt.figure(figsize=(3,3))
        plt.imshow(img_temp[i],cmap = 'gray')
        plt.xticks([])
        plt.yticks([])
        plt.title(titles[i])
        plt.show()
        cv2.imwrite('C:\\Users\\DELL\\Desktop\\Project_codes\\DWT_'+img_name+' '+titles[i]+'.png',img_temp[i])
    
    #hard thresholding of detailed coefficients for compression
    def threshold(img_arr,th):
        r,c = img_arr.shape
        
        i = 0
        j = 0
        while i < r:
            while j < c:
                if img_arr[i,j] < th:
                    img_arr[i,j] = 0
                j += 1
            j = 0
            i += 1
        return img_arr
            
    H1 = threshold(H,th)
    V1 = threshold(V,th)
    D1 = threshold(D,th)  
    
    th_coeff = A,(H1,V1,D1)
    idwt = pywt.idwt2(th_coeff,'db3',mode='periodization')
    idwt = idwt.astype('uint8')
    plt.imshow(idwt,cmap='gray')
    plt.show()          
    cv2.imwrite('C:\\Users\\DELL\\Desktop\\Project_codes\\DWT_'+'Output DWT'+img_name+str(th)+'.png',idwt)            
    
    x = 0
    y = 0
    MSE = 0
    while x < h:
        while y < w:
            MSE += ((gray1[x][y] - idwt[x][y])**2)/(h*w)
            y += 1
        y = 0
        x += 1

    PSNR = 10*np.log10(255*255/MSE)
    
    #calculating compression ratio
    original_img_size = os.path.getsize('D:\\standard_test_images\\'+img_name)
    compressed_img_size = os.path.getsize('C:\\Users\\DELL\\Desktop\\Project_codes\\DWT_'+'Output DWT'+img_name+str(th)+'.png')
    CR = original_img_size/compressed_img_size
    
    return MSE, PSNR, CR


if __name__ == '__main__':
    th_values = []
    MSE_values = []
    PSNR_values = []
    CR_values = []
    img_name = input('Enter image name: ')
    n = int(input('Enter no. of threshold values you want to use: '))
    for _ in range(n):
        th = int(input('Enter Threshold Value: '))
        th_values.append(th)
        mse,psnr,cr = DWT(img_name,th)
        MSE_values.append(mse)
        PSNR_values.append(psnr)
        CR_values.append(cr)
    print(th_values)
    print(MSE_values)
    print(PSNR_values)
    
    #placing all values in a pandas dataframe and saving it
    df = pd.DataFrame({'Threshold':th_values,'MSE':MSE_values,'PSNR':PSNR_values,'Compression Ratio':CR_values})
    df.to_csv('C:\\Users\\DELL\\Desktop\\Project_codes\\DWT'+img_name+'.csv')
    
    #plotting windowm size vs MSE, PSNR, CR and saving the plots
    mse_plot = sns.lineplot(x = 'Threshold',y = 'MSE',color = 'aqua',marker = 'o',data = df)
    fig1 = mse_plot.get_figure()
    plt.title('Mean square error plot')
    plt.show()
    fig1.savefig('DWT_MSE'+img_name+'.png')
    
    psnr_plot = sns.lineplot(x = 'Threshold',y = 'PSNR',color = 'red',marker = 'o',data = df)
    fig2 = psnr_plot.get_figure()
    plt.title('Peak signal to noise ratio plot')
    plt.show()
    fig2.savefig('DWT_PSNR'+img_name+'.png')
    
    cr_plot = sns.lineplot(x = 'Threshold',y = 'Compression Ratio',color = 'green',marker = 'o',data = df)
    fig3 = cr_plot.get_figure()
    plt.title('Compression ratio plot')
    plt.show()
    fig3.savefig('DWT_CR'+img_name+'.png')  
    
    sns.lineplot(x = 'Threshold',y = 'Compression Ratio',color = 'orange',marker = 'o',data = df)
    plt.legend(labels=['CR'])
    ax2 = plt.twinx()
    comp = sns.lineplot(x = 'Threshold',y = 'PSNR',color = 'purple',marker = 'o',data = df)
    fig4 = comp.get_figure()
    plt.legend(labels=['PSNR'])
    plt.title('CR & PSNR comparision plot')
    plt.show()
    fig4.savefig('DWT_CR & PSNR'+img_name+'.png')
    
