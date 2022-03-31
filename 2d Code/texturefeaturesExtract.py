#from Phuong T. Pham - Begin: 08.10.2021
from statistics import median_high
import cv2
from cv2 import GaussianBlur
import numpy as np
import mahotas as mt
import matplotlib.pyplot as plt
import string
import pandas as pd
#from torch import convolution
import haralick
import math
from scipy import ndimage as ndi

 
##################################################HARALICK - FEATURES##################################################################################
##################################################HARALICK - FEATURES##################################################################################
##################################################HARALICK - FEATURES##################################################################################
##################################################HARALICK - FEATURES##################################################################################

#define a extract haralick features function
def extract_haralick_features(image, path = '', round_object = False, underground = 0):
    if(path != ''): 
        image = cv2.imread(path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        #print(extract_LBP(image))
    if round_object:
        h = haralick.Haralick(image, round_object, underground)
        features = h.props()
    else: 
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
    
def extract_HF_std_of_a_probe(image1_path, image2_path, grid = False):
    features = extract_HF_of_a_probe(image1_path, image2_path, grid = grid)
    return np.std(features).values

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


##################################################GRADIENT - GABORFILTER##################################################################################
##################################################GRADIENT - GABORFILTER##################################################################################
##################################################GRADIENT - GABORFILTER##################################################################################
##################################################GRADIENT - GABORFILTER##################################################################################
##################################################GRADIENT - GABORFILTER##################################################################################

def convolution(image, kernel, average=False, verbose=False):
    if len(image.shape) == 3:
        print("Found 3 Channels : {}".format(image.shape))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        print("Converted to Gray Channel. Size : {}".format(image.shape))
    else:
        print("Image Shape : {}".format(image.shape))

    print("Kernel Shape : {}".format(kernel.shape))

    if verbose:
        plt.imshow(image, cmap='gray')
        plt.title("Image")
        plt.show()

    image_row, image_col = image.shape
    kernel_row, kernel_col = kernel.shape

    output = np.zeros(image.shape)

    pad_height = int((kernel_row - 1) / 2)
    pad_width = int((kernel_col - 1) / 2)

    padded_image = np.zeros((image_row + (2 * pad_height), image_col + (2 * pad_width)))

    padded_image[pad_height:padded_image.shape[0] - pad_height, pad_width:padded_image.shape[1] - pad_width] = image

    if verbose:
        plt.imshow(padded_image, cmap='gray')
        plt.title("Padded Image")
        plt.show()

    for row in range(image_row):
        for col in range(image_col):
            output[row, col] = np.sum(kernel * padded_image[row:row + kernel_row, col:col + kernel_col])
            if average:
                output[row, col] /= kernel.shape[0] * kernel.shape[1]

    print("Output Image size : {}".format(output.shape))

    if verbose:
        plt.imshow(output, cmap='gray')
        plt.title("Output Image using {}X{} Kernel".format(kernel_row, kernel_col))
        plt.show()

    return output

def dnorm(x, mu, sd):
    return 1 / (np.sqrt(2 * np.pi) * sd) * np.e ** (-np.power((x - mu) / sd, 2) / 2)


def gaussian_kernel(size, sigma=1, verbose=False):
    kernel_1D = np.linspace(-(size // 2), size // 2, size)
    for i in range(size):
        kernel_1D[i] = dnorm(kernel_1D[i], 0, sigma)
    kernel_2D = np.outer(kernel_1D.T, kernel_1D.T)

    kernel_2D *= 1.0 / kernel_2D.max()

    if verbose:
        plt.imshow(kernel_2D, interpolation='none', cmap='gray')
        plt.title("Kernel ( {}X{} )".format(size, size))
        plt.show()

    return kernel_2D


def gaussian_blur(image, kernel_size, verbose=False):
    kernel = gaussian_kernel(kernel_size, sigma=math.sqrt(kernel_size), verbose=verbose)
    return convolution(image, kernel, average=True, verbose=verbose)

def egde_detection(image, filter, show = False):
    img_x = convolution(image, filter, show)

    if show:
        plt.imshow(img_x, cmap = 'gray')
        plt.title("Horizontal Edge")
        plt.show()
    
    img_y = convolution(image, np.flip(filter.T, axis=0), show)

    if show:
        plt.imshow(img_y, cmap = 'gray')
        plt.title("Vertical Edge")
        plt.show()
    
    gradient = np.sqrt(np.square(img_x) + np.square(img_y))
    gradient += 255.0 / gradient.max() #normalize

    if show:
        plt.imshow(gradient, cmap='gray')
        plt.title('gradient')
        plt.show()
    return gradient


def build_filters():
    filters = []
    ksize = 9
    for theta in np.arange(0, np.pi, np.pi / 8):
        for lamda in np.arange(0, np.pi, np.pi/4): 
            kern = cv2.getGaborKernel((ksize, ksize), 1.0, theta, lamda, 0.5, 0, ktype=cv2.CV_32F)
            kern /= 1.5*kern.sum()
            filters.append(kern)
    return filters

def gabor_process(img, filters):
    accum = np.zeros_like(img)
    for kern in filters:
        fimg = cv2.filter2D(img, -1, kern)
        np.maximum(accum, fimg, accum)
    return accum


def getGaborFeatures(path, sobel_filter, gabor_kernels):
    image = cv2.imread(path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    res = []
    edge_img = egde_detection(image, sobel_filter, False)
    for i in range(len(gabor_kernels)):
        res1 = gabor_process(edge_img, gabor_kernels[i])
        res.append(np.asarray(res1))
    return res
if __name__ == "__main__":
  
    image1 = cv2.imread(r"Klebeverbindungen_Daten\2D-MakroImages\Betamate 1496V\ProbeE1_1.png")
    gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)

    # print(extract_haralick_features(gray1, round_object=True, underground=255))

    # image2 = cv2.imread(r"Klebeverbindungen_Daten\Test\ProbeR1_2.png")
    # gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

    filter = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    


    kernels = []
    res = []
    label = []

    filters = build_filters()
    filters = np.asarray(filters)

    




