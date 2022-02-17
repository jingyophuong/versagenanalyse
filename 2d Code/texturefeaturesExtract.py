#from Phuong T. Pham - Begin: 08.10.2021
from mailbox import MH
from statistics import median_high
import cv2
import numpy as np
import os
import glob
import mahotas as mt
import matplotlib.pyplot as plt
import string
import pandas as pd




def extract_haralick_features_manuell(image):
    return
#define a extract haralick features function
def extract_haralick_features(image, path = ''):
    if(path != ''): 
        image = cv2.imread(path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    features = mt.features.haralick(image)

    haralick_features_mean = features.mean(axis=0)
    return haralick_features_mean


#define a extract LBP features function
def extract_LBP(image):
    feat = mt.features.lbp(image=image, points=(2*8), radius=2, ignore_zeros="True")
    feat = np.reshape(cv2.calcHist([np.float32(feat)], [0], None, [256], [0, 256]),(256))
    return feat

#devide a image in 5x4 grid images and calculating the HF for each image
def features_of_grid(image, axis = 0):
    height = image.shape[0]
    width = image.shape[1]    

    x_step = int(width / 5)
    y_step = int(height / 4)

    x = 0
    y= 0
    
    all_features = []
    
    for j in range(4):
        if j < 3: 
            h = y_step 
        else:
            h = height - y
        for i in range(5):
            if i < 4: 
                w = x_step 
            else:
                w = width - x
        
            crop_img = image[y:y+h, x:x+w] 
            
            x += w
            if axis == 0:
                if i == 0 and j == 0:
                    all_features = extract_haralick_features(crop_img)
                elif j == 0 and i == 1:
                    hfc = extract_haralick_features(crop_img)
                    all_features = np.stack((all_features, hfc), axis=-1)
                else:
                    hfc = []
                    hfc.append(extract_haralick_features(crop_img))
                    hfc = np.asarray(hfc)
                    all_features = np.concatenate((all_features, hfc.T), axis=1)
            else:
                all_features.append(extract_haralick_features(crop_img))
        y += h
        x= 0
            
    return all_features


def extract_HF_of_a_probe(image1_path, image2_path, axis = 0, grid  = True):
    image = cv2.imread(image1_path)
    image2 = cv2.imread(image2_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(image2, cv2.COLOR_RGB2GRAY)
    if grid: 
        if axis == 0:
            f = np.column_stack((features_of_grid(gray, axis=axis), features_of_grid(gray2, axis= axis)))
        else:
            f = np.row_stack((features_of_grid(gray, axis=axis), features_of_grid(gray2, axis= axis)))
        
        return f
    
    f= pd.DataFrame()
    f = f.append(features_with_kernel(gray), ignore_index=True)
    f = f.append(features_with_kernel(gray2), ignore_index=True)
    return f


def extract_HF_mean_of_a_probe(grid_features):
    return np.mean(grid_features, axis= 1)

def extract_HF_mean_of_a_probe(image1_path, image2_path, grid = False):
    features = extract_HF_of_a_probe(image1_path, image2_path, grid = grid)
    return np.mean(features).values

def get_labels_of_grid():
    g = string.ascii_uppercase
    g = g[:20]
    g = list(g)
    k = string.ascii_lowercase
    k = k[:20]
    k = list(k)
    k = np.asarray(k)
    k = np.split(k, 4)
    k = np.flipud(k)
    k = k.flatten()
    g = np.stack((g, k))
    apb = g.flatten()
    return apb

def features_with_kernel(image, white = True):
    kernel = [100,100]
    step = 5
    #kernel run though image
    height = image.shape[0]
    width = image.shape[1]
    x = 0
    y = 0
    m_height = kernel[0]
    m_width = kernel[1]
    feature_names=["Angular Second Moment", "Contrast", "Correlation", "Sum of Squares: Variance", "Inverse Difference Moment", "Sum Average", 
                            "Sum Variance", "Sum Entropy", "Entropy", "Difference Variance", "Difference Entropy", "Info. Measure of Correlation 1", "Info. Measure of Correlation 2"]
  
    features = pd.DataFrame(columns = feature_names)
    while (x < width):
        if(x+m_width >= width): 
            m_width = width - x 
        while (y < height):
            if(y + m_height >= height):
                m_height = height - y
            m_img = image[y : y + m_height- 1, x : x+m_width-1]
            if white: 
                r_img = 255 - m_img
                zerocount = np.count_nonzero(r_img)
            else: 
                zerocount = np.count_nonzero(m_img)
            if(zerocount > m_height * m_width * 0.9):
            #features = features.append(extract_haralick_features(m_img))
                feature = pd.Series(extract_haralick_features(m_img), index= feature_names)
                features = features.append(feature, ignore_index=True)
            y += step
            #print(x, y, m_width , m_height)
        x += step
        y = 0
    return features.mean()

def put_image_above_a_image(image, qimage1):
# Now create a mask of logo and create its inverse mask also
    img2gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    ret, mask = cv2.threshold(img2gray, 10, 255, cv2.THRESH_BINARY)
    mask_inv = cv2.bitwise_not(mask)
# Now black-out the area of logo in ROI
    img1_bg = cv2.bitwise_and(qimage1,qimage1,mask = mask_inv)
# Take only region of logo from logo image.
    img2_fg = cv2.bitwise_and(image,image,mask = mask)
# Put logo in ROI and modify the main image
    dst = cv2.add(img1_bg,img2_fg)
    return dst
    
def round_image_mani(image):
  
    o_width = image.shape[1]
    o_height = image.shape[0]
    
    b_width = o_width *2 -1
    b_height = o_height * 2 -1

    Mitte = [int(o_width / 2), int(o_height / 2)]
    x0c = o_width - Mitte[0] -1
    y0c = o_height - Mitte[1] -1
    Mitte = np.add([x0c, y0c], Mitte)
    b_image = np.zeros([b_height,b_width, 3],dtype=np.uint8)
    b_image[y0c: y0c + o_height, x0c : x0c + o_width , :] = image
    #1/4 
    qimage = b_image[0 : 0 + o_height, 0 : 0 + o_width]

    image = put_image_above_a_image(image, qimage)
    qimage = b_image[0 : 0 + o_height, Mitte[0] : Mitte[0] + o_width]
   
    image = put_image_above_a_image(image, qimage)
    qimage = b_image[Mitte[1] : Mitte[1] + o_height, 0 : 0 + o_width]
  
    image = put_image_above_a_image(image, qimage)
    qimage = b_image[Mitte[1] : Mitte[1] + o_height,  Mitte[0] : Mitte[0] + o_width]
   
    image = put_image_above_a_image(image, qimage)

    return image

if __name__ == "__main__":
    #fs = extract_HF_mean_of_a_probe(r"Klebeverbindungen_Daten\2D-MakroImages\Betamate 1496V\ProbeR2_1.png",  r'Klebeverbindungen_Daten\2D-MakroImages\Betamate 1496V\ProbeR2_2.png')
    #fs = np.mean(fs, axis = 1)
    image = cv2.imread(r"Klebeverbindungen_Daten\Test\1.png")
    image2 = cv2.imread(r"Klebeverbindungen_Daten\Test\2.png")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(image2, cv2.COLOR_RGB2GRAY)

    image3 = cv2.imread(r"Klebeverbindungen_Daten\Test\ProbeR1_3.png")
    gray3 = cv2.cvtColor(image3, cv2.COLOR_BGR2GRAY)

    image4 = cv2.imread(r"Klebeverbindungen_Daten\Test\ProbeR1_4.png")
    gray4 = cv2.cvtColor(image4, cv2.COLOR_BGR2GRAY)

    fig,axs  = plt.subplots(1,2, figsize=(10,10))
    #axs[0].imshow(gray)
    #axs[1].imshow(gray2)
    #axs[2].imshow(gray3)
    #axs[3].imshow(gray4)

    # features_kernel = pd.DataFrame()
    # features_kernel = features_kernel.append(features_with_kernel(gray), ignore_index=True)
    # features_kernel =features_kernel.append(features_with_kernel(gray2), ignore_index=True)
    # features_kernel =features_kernel.append(features_with_kernel(gray3, white= False), ignore_index=True)
    # features_kernel =features_kernel.append(features_with_kernel(gray4), ignore_index=True)
    # print(features_kernel)
    columns=["Angular Second Moment", "Contrast", "Correlation", "Sum of Squares: Variance", "Inverse Difference Moment", "Sum Average", 
                            "Sum Variance", "Sum Entropy", "Entropy", "Difference Variance", "Difference Entropy", "Info. Measure of Correlation 1", "Info. Measure of Correlation 2"] 
   
    features_grid = pd.DataFrame()

    features_grid = features_grid.append(pd.Series(extract_haralick_features(gray), index= columns), ignore_index=True)
    features_grid =features_grid.append(pd.Series(extract_haralick_features(gray2), index= columns), ignore_index=True)
    features_grid =features_grid.append(pd.Series(extract_haralick_features(gray3), index= columns), ignore_index=True)
    features_grid =features_grid.append(pd.Series(extract_haralick_features(gray4), index= columns), ignore_index=True)
    print(features_grid)
    #print(fs)
    
    # plt.show()
    #features_with_kernel(gray)
    image5 = cv2.imread(r"Klebeverbindungen_Daten\Test\ProbeR4_1.png")
    axs[0].imshow(image5)

    image5 = round_image_mani(image5)
    axs[1].imshow(image5)
    plt.show()
