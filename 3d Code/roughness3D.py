#Author: Phuong Pham 
#Datum: from 11.2021 - 

import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt 


def show_data(data):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection = '3d')
    ax.scatter(data['x'], data.y, data.z, zdir='z', c='red')
    plt.show()

# calculating Ra, Rz, Rmax, Rq for whole surface

########################################################ROUGHNESS##########################################################################
def cal_Roughness_params(data):
    Roughness_Params = pd.DataFrame(columns=['Sa', 'Sq', 'Sz', 'Ssk', 'Ssu'])
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

    Sz = (valleys.sum() + summits.sum()) / 5
    Sa = data['absz'].mean()
    Sq = np.sqrt(data['z2'].mean())
    Ssk = 1/ (pow(Sq, 3)) * data.z3.mean()
    Ssu = 1/ (pow(Sq, 4)) * data.z4.mean()

    Roughness_Params = Roughness_Params.append({'Sa' :  Sa , 'Sq' : Sq,
                                                'Sz': Sz, 'Ssk': Ssk, 'Ssu': Ssu }, ignore_index= True)

    #print(data)
    return Roughness_Params

def cal_Roughness_params(path):
    data = pd.read_csv(path,sep=';', names=['x', 'y', 'z'])
    Roughness_Params = pd.DataFrame(columns=['Sa', 'Sq', 'Sz', 'Ssk', 'Ssu'])
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

    Sz = (valleys.sum() + summits.sum()) / 5
    Sa = data['absz'].mean()
    Sq = np.sqrt(data['z2'].mean())
    Ssk = 1/ (pow(Sq, 3)) * data.z3.mean()
    Ssu = 1/ (pow(Sq, 4)) * data.z4.mean()

    Roughness_Params = Roughness_Params.append({'Sa' :  Sa , 'Sq' : Sq,
                                                'Sz': Sz, 'Ssk': Ssk, 'Ssu': Ssu }, ignore_index= True)

    #print(data)
    return Roughness_Params

#calculating Ra, Rz, Rmax, Rq for sampling surfaces
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

