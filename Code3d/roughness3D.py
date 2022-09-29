#Author: Phuong Pham 
#Datum: from 11.2021 - 

import os
from random import random
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

def show_data(data):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection = '3d')
    ax.scatter(data['x'], data.y, data.z, zdir='z', c='red')
    plt.show()

# calculating Ra, Rz, Rmax, Rq for whole surface

########################################################ROUGHNESS##########################################################################
def cal_Roughness_params(data, path = ''):
    if(path != ''):
        data =  pd.read_csv(path,sep=';', names=['x', 'y', 'z'])
    xyz_minmax = []
    xyz_minmax.append([data.z.min(), data.z.max()])
    print(xyz_minmax)
    Roughness_Params = pd.DataFrame(columns=['Sa','Sq', 'Ssk' , 'Sku', 'Sp', 'Sv', 'Sz'])
    # cal abs
    data.__setitem__('absz', abs(data.z))
    # cal ^2
    data.__setitem__('z2', data.z * data.z)
    # cal ^3
    data.__setitem__('z3', data.z2 * data.z)
    # cal ^4
    data.__setitem__('z4', data.z3 * data.z)
    # find heighest and lowest z
    data = data.sort_values(by=['z'])
    valleys = abs(data.z.head(5))
    data = data.sort_values(by=['z'], ascending = False)
    summits = abs(data.z.head(5))
    Sv = valleys.sum()/5
    Sp = summits.sum()/5
    Sz = Sv + Sp
    Sa = data['absz'].mean()
    Sq = np.sqrt(data['z2'].mean())
    Ssk = 1/ (pow(Sq, 3)) * data.z3.mean()
    Sku = 1/ (pow(Sq, 4)) * data.z4.mean()

    Roughness_Params = Roughness_Params.append({'Sa' :  Sa , 'Sq' : Sq,
                                                'Sz': Sz, 'Ssk': Ssk, 'Sku': Sku , 'Sv': Sv, 'Sp' : Sq}, ignore_index= True)

    #print(data)
    return Roughness_Params


def cal_Roughness_params_sampling(depthmap, roi):
    #Roughness_Params = pd.DataFrame(columns=['Sa', 'Sq', 'Sz', 'Ssk', 'Ssu'])
    y_roi = roi[0]
    x_roi = roi[1]
    dm_shape = depthmap.shape
    max_y = dm_shape[0]-1
    max_x = dm_shape[1]-1
    n_y_step = int(max_y / (y_roi -1))
    if(max_y % (y_roi -1) == 0): n_y_step -=1
    n_x_step = int(max_x / (x_roi -1))
    if(max_x % (x_roi -1) == 0): n_x_step -=1
    print('Number of segments = ' , n_y_step*n_x_step)
    x0 = 0
    y0 = 0
    MW = []
    STABW = []
    for i in range(n_x_step):
        y0 = 0
        x0 = (x_roi -1)* i
        for j in range(n_y_step):
            y0 = (y_roi -1)* j
            data = depthmap[y0:y0+y_roi ,x0:x0+x_roi]
            data = np.array(data).astype(int)
            #print(x0, y0)
            m = data.mean()
            MW.append(m)
            abw_data = np.sqrt(((data -m)**2).mean())
            
            STABW.append(abw_data)

            #print(m, abw_data)


    return [MW, STABW]



##################################################WAVENESS############################################################################



###################################################GAUSS-FILTER#######################################################################


def convolution(image, kernel, average=False, verbose=False):
    
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

def gaussfilter(z, dx, dy, lambdac_x, lambdac_y):

    const = np.sqrt(np.log(2) / np.pi)
    xs = np.arange(-lambdac_x, lambdac_x-dx, dx)
    ys = np.arange(-lambdac_y, lambdac_y-dy, dy)
    mx = len(xs)
    my = len(ys)
    S = np.zeros((my, mx))
    for i, y_i in enumerate(ys):  
        for j, x_j in enumerate(xs):
            S[i,j] = (1/(const**2 * lambdac_x * lambdac_y))* np.exp(-np.pi*(x_j/const /lambdac_x)**2 - np.pi * (y_i / const / lambdac_y)**2)
    S = S/np.sum(np.sum(S))
    w = convolution(z, S)
    r = z -w
    return (r,w)

def test_gaussfilter(): 
    # dx = 0.1
    # dy = 0.1
    # nx = 128
    # ny = 128
    # x = np.arange(0, nx -1, 1) * dx
    # y = np.arange(0, ny -1, 1) * dy
    # X, Y = np.meshgrid(x,y)
    # Z = f(X)
    # Z = Z - np.mean(np.mean(Z))

    # lambdacx = 6.4
    # lambdacy =6.4

    # res = gaussfilter(Z,dx,dy,lambdacx, lambdacy)

  


    # fig = plt.figure(figsize=(4,4))

    # fig = plt.figure()
    # ax = plt.axes(projection='3d')
    # ax.contour3D(X, Y, res[1], 50, cmap='binary')
    # ax.set_xlabel('x')
    # ax.set_ylabel('y')
    # ax.set_zlabel('z')
    # plt.show()


    xyz_datei_path = r"Klebeverbindungen_Daten\3d\RelativData\ProbeE1_1.txt"

    data = data = pd.read_csv(xyz_datei_path, sep=';', names=['y', 'x', 'z'])
    x = data.x.values
    y = data.y.values
    z = data.z.values
    fig = plt.figure(figsize=(4,4))

    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.scatter(x, y, z,  c=z, cmap='viridis', linewidth=0.5)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    plt.show()


###################################################CURVATURE###################################################

def curvature(data, path = ''):
    if(path != ''):
        data =  pd.read_csv(path,sep=';', names=['x', 'y', 'z'])
    data = data.to_numpy()

    #point cloud to mesh

    # points is a 3D numpy array (n_points, 3) coordinates of a sphere
    cloud = pv.PolyData(data)
    #cloud.plot()


    volume = cloud.delaunay_3d(alpha=27.)
    shell = volume.extract_geometry()

    #grid = pv.StructuredGrid(shell)
    
    curva = shell.curvature(curv_type='gaussian')
    shell.plot(scalars=curva)
    print(curva)

def f(X): 
    return -1/64 *X*X + X
def r(X):
    return np.random.rand() 
def w(X): 
    return 2*np.sin(2* np.pi* X /6.4)
def get_roughness_of_a_probe(file1, file2):
    data1 =  pd.read_csv(file1 ,sep=';', names=['x', 'y', 'z'])
    data2 =  pd.read_csv(file2 ,sep=';', names=['x', 'y', 'z'])

    r1 = cal_Roughness_params(data1)
    r2 = cal_Roughness_params(data2)
    r = pd.DataFrame()
    r =r.append(r1, ignore_index=True)
    r =r.append(r2, ignore_index=True)
    r.to_csv('r3de6.csv', sep = ';', decimal=',', float_format='%.3f')
    print(r)
    return np.concatenate((r.mean().values, r.std().values))

if __name__ == '__main__':
    path1 =  "Klebeverbindungen_Daten\AP5-3D Punktwolken\BM\RelativData\ProbeE6_1-rel.txt"
    path2 = "Klebeverbindungen_Daten\AP5-3D Punktwolken\BM\RelativData\ProbeE6_2-rel.txt"
    r = get_roughness_of_a_probe(path1, path2)
    print(r)
    #print(cal_Roughness_params(0,path1))
    #print(cal_Roughness_params(0,path2))
