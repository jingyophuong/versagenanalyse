#Author: Phuong Pham 
#Datum: from 11.2021 - 

#from scipy import signal
from turtle import st
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt
###############################################################VISUALIZATION###############################################################################
###############################################################VISUALIZATION###############################################################################
###############################################################VISUALIZATION###############################################################################
###############################################################VISUALIZATION###############################################################################

pi = 3.14159265359
def show_a_profil(data, y):
    dataofy = data.loc[data['y'] == y]
    dataofy = dataofy.sort_values(by=['x'])
    x = dataofy.x
    z = dataofy.z
    w = dataofy.zw
    r = dataofy.zr
    plt.plot(x, z,  'o-' ,label = 'P-Profil')
    plt.plot(x, w,  'o-', label = 'W-Profil')
    plt.plot(x, r,  'o-', label = 'R-Profil')
    
    plt.legend()
    title = 'Profil of surface by y = ' + str(y)
    plt.title(title)
    plt.show()
##################################################Gauss-Filter##############################################################################
##################################################Gauss-Filter##############################################################################
##################################################Gauss-Filter##############################################################################
##################################################Gauss-Filter##############################################################################
##################################################Gauss-Filter##############################################################################

def test_gaussfilter(): 
    n = 8000

    dx= 0.001

    x = np.arange(n) * dx

    z = 2 * np.sin(2 * pi * x/16) + 0.1 * np.random.random(n)
  #  z = signal.detrend(z)
    print(z)
    lambdac = 0.8
    #dx = 0.08

    gaussfilter(data = z, x = x, dx = dx, lambdac = lambdac)
    

def gaussfilter(data, x, dx, lambdac):
    const = np.sqrt(np.log(2)/2/pi/pi)
    n = len(data)
    w = np.zeros(n)
    for k in range(n):
        p = np.arange(n)
        S=(1/np.sqrt(2*pi)/const/lambdac)*np.exp(-0.5*((k -p)*dx/const / lambdac)**2)
        smod = S / S.sum()
        w[k]= np.sum( np.multiply(smod, data))
    r = data - w
    #w = np.convolve(smod, data)
    return [r, w]

        

def apply_gaussfilter_for_surface(data, lambdac):
    n_y_data = data.groupby('y').mean()
    n_y = n_y_data.index
    filtered_data = pd.DataFrame(columns=['x', 'y', 'z', 'zr', 'zw'])
    for y in n_y:
                    
        dataofy = data.loc[data['y'] == y]
        
        dataofy = dataofy.sort_values(by=['x'])
        dataofy =  dataofy.reset_index(drop = True)
        dataofy['dx'] = np.abs(dataofy['x'].diff())
        dataofy['dx'][0] = 11
        
        dx = dataofy['dx'].mean() * 0.01
        
        x = np.array(dataofy['x'])
        z = np.array(dataofy.z)
        
        
        profils = gaussfilter(data=z, x = x , dx= dx, lambdac=2)
        
        dataofy['zr'] = profils[0]
        dataofy['zw'] = profils[1]
        filtered_data = filtered_data.append(dataofy, ignore_index = True)
   # print(filtered_data)
    return filtered_data

 
###############################################################ROUGHNESS###############################################################################
###############################################################ROUGHNESS###############################################################################
###############################################################ROUGHNESS###############################################################################
###############################################################ROUGHNESS###############################################################################
###############################################################ROUGHNESS###############################################################################



def compute_Rsm(profil,dx,  HeightThreshold, SpacingThreshold): 
     #compute Rsm: as the average distance between zero crossings of the profile
        #find all zero crossing
    r = profil.zr
    n = len(r)
    zerocrossing = []
    for j in range(n-1):
        if(r[j] * r[j+1] < 0): 
            zerocrossing.append(j)
        #make sure the first zero is a low to high crossing
    if len(zerocrossing) == 0: 
        return 0
    if(r[zerocrossing[0]] > 0 and r[zerocrossing[0] +1] <0): 
        zerocrossing = zerocrossing[1: len(zerocrossing)]
    
        #compute Rsm
    Left = 0
    Right = Left + 2
    NextToRight = Left +3
    Smi =[]
    while NextToRight < len(zerocrossing):
        a = max(r[zerocrossing[Left] : zerocrossing[Right]])
        b = min(r[zerocrossing[Left] : zerocrossing[Right]])
        c = max(r[zerocrossing[Right] : zerocrossing[NextToRight]])
        
        if(a > HeightThreshold and b < -1*HeightThreshold and c > HeightThreshold):
            LeftInterpolated = zerocrossing[Left] *dx - dx * r [zerocrossing[Left]] / (r[zerocrossing[Left] +1] - r[zerocrossing[Left]])
            RightInterpolated = zerocrossing[Right] *dx- dx * r[zerocrossing[Right]] / (r[zerocrossing[Right] +1] - r[zerocrossing[Right]])
            temp = RightInterpolated - LeftInterpolated
            if temp > SpacingThreshold:
                Smi.append(temp)
                Left = Right
                Right = Left+2
                NextToRight = Left +3
            else: 
                Right += 2
                NextToRight += 2
        else:
            Right += 2
            NextToRight += 2
    #return np.mean(Smi)
    if len(Smi) ==0: 
        return 0
    else:
        return np.mean(Smi)
#calculating Ra, Rz, Rmax, Rq of a profil
#Ra: Arithmetischer Mittenrauwert
#Rz: Gemittelte Rautiefe
#Rmax: Maximale Rautiefe
#Rq: Quadratischer Mittenrauwert
#Rsm Mittlerer Rillenabstand
#Rsk: Schiefe
#Rku: Kurtosis/Steilheit
def cal_roughness_params_for_a_profil(data, y): 
    Roughness_Params = pd.DataFrame(columns=['y', 'Ra', 'Rq', 'Rz', 'Rmax',  'Rsm', 'Rsk', 'Rku'])
   
    profil = data.loc[data['y'] == y]
    profil = profil.sort_values(by=['x'])
    if profil.empty:
        return
    
    profil = profil.reset_index(drop= True)
    mean =  profil['zr'].mean() 
    profil['zr'] = profil['zr'] - mean #shift profil to zero mean
    xmax = profil['x'].max()
    xmin = profil['x'].min()
    dis = xmax - xmin +1 

    step = int(dis/6)
    if xmin > step or step < 48: 
        return
    f_limit = int(step /2) + xmin
    r_a_sum = 0
    r_z_sum = 0
    r_z_n = []
    r_q_sum = 0
    r_sm_sum = 0
    r_sk_sum = 0
    r_ku_sum = 0
    for i in range(5):
        
        filter1 = profil['x'] >= f_limit + i* step 
        filter2 =  profil['x'] <= f_limit + (i+1)*step
        single_mess_section = profil.loc[filter1 & filter2]
        if single_mess_section.empty:
            return
        single_mess_section =  single_mess_section.reset_index(drop = True)
        #cal Ra
        single_mess_section.__setitem__('absz', abs(single_mess_section.zr))
        single_ra = single_mess_section.groupby('y').mean()
        r_a_sum += single_ra['absz'].values[0] 

        #cal Rz
        Rz = single_mess_section['zr'].max() - single_mess_section['zr'].min()
        r_z_sum += Rz

        #find Rmax (or Rz1max)
        r_z_n.append(Rz)

        #cal Rq
        single_mess_section.__setitem__('z2', single_mess_section.zr * single_mess_section.zr)
        rq = single_mess_section.groupby('y').mean()['z2'].values[0]
        r_q_sum += np.sqrt(rq)
        
        # cal Rsm
        dx = 0.12#single_mess_section.dx.mean()
        SpacingThreshold = 0.01 * len(single_mess_section)* dx #the spacing between the two zero crossing is greater than a threshold (typical value is 1% of the sampling length)
                
        HeightThreshold = 0.1 * Rz # the maximum profile height between the two zero crossings is greater than a threshold (typical value is 10% of Rz)
        
    
        r_sm_sum += compute_Rsm(profil=single_mess_section, dx = dx, HeightThreshold = HeightThreshold, SpacingThreshold = SpacingThreshold )
        
        #compute Rsk
        single_mess_section.__setitem__('z3', single_mess_section.z2 * single_mess_section.zr)
        z3_mean = single_mess_section.groupby('y').mean()['z3'].values[0]
        r_sk = (1/ pow(rq,3)) * z3_mean
        r_sk_sum += r_sk
        
        #compute Rku
        
        single_mess_section.__setitem__('z4', single_mess_section.z3 * single_mess_section.zr)
        z4_mean = single_mess_section.groupby('y').mean()['z4'].values[0]
        r_ku = (1/ pow(rq,4)) * z4_mean
        r_ku_sum += r_ku
        
    


    Roughness_Params = Roughness_Params.append({'y' : y, 'Ra' : r_a_sum / 5, 'Rz': r_z_sum / 5, 
                                                'Rmax': np.max(r_z_n), 'Rq': r_q_sum / 5, 'Rsm': r_sm_sum/5,
                                               'Rsk': r_sk_sum / 5, 'Rku': r_ku_sum /5} , ignore_index= True)
   
    return Roughness_Params


def cal_Roughness_params(data): 
    n_y_data = data.groupby('y').mean()
    n_y = n_y_data.index
    
    Roughness_Params = pd.DataFrame(columns=['y', 'Ra', 'Rq', 'Rz', 'Rmax',  'Rsm', 'Rsk', 'Rku'])
  
    for y in n_y:
        r_of_a_profil = cal_roughness_params_for_a_profil(data=data, y = y)
        if not r_of_a_profil is None:
            if not r_of_a_profil.empty:
                Roughness_Params = Roughness_Params.append(r_of_a_profil, ignore_index= True)
    return Roughness_Params


##################################################WAVENESS##################################################################################
##################################################WAVENESS##################################################################################
##################################################WAVENESS##################################################################################
##################################################WAVENESS##################################################################################
##################################################WAVENESS##################################################################################


#cal WDt, WDSm, WDc
#WDc: Mittlere Profilelementhöhe
#WDSm: Dominante Wellenlänge 
#WDt: Maximale Profilelementdifferenz
def cal_waveness_for_a_profil(data, y):
    Waveness_Params = pd.DataFrame(columns = ['WDt', 'WDsm', 'WDc'])
    profil = data.loc[data['y'] == y]
    profil = profil.sort_values(by=['x'])
    if profil.empty:
        return
     
    profil = profil.reset_index(drop= True)
    mean =  profil.zw.mean() 
    profil.zw = profil.zw - mean #shift profil to zero mean
    xmax = profil['x'].max()
    xmin = profil['x'].min()
    dis = xmax - xmin +1 
    step = int(dis/6)
    if xmin > step: 
        return
    f_limit = int(step /2)
    
    wc_sum = 0
    for i in range(5):
        
        filter1 = profil['x'] >= f_limit + i* step 
        filter2 =  profil['x'] <= f_limit + (i+1)*step
        single_mess_section = profil.loc[filter1 & filter2]
        single_mess_section =  single_mess_section.reset_index(drop = True)
       
        #cal Wc
        Wct= single_mess_section['zr'].max() - single_mess_section['zr'].min()
        wc_sum += Wct

    
    #compute WDt
    max_w = profil.zw.max()
    min_w = profil.zw.min()
    wDt = abs(max_w - min_w)
    #Compute WDc
    wdc  = wc_sum / 5
    #Compute WDSm
    dx = 0.12#single_mess_section.dx.mean()
    SpacingThreshold = 0.01 * len(profil)* dx #the spacing between the two zero crossing is greater than a threshold (typical value is 1% of the sampling length)
                
    HeightThreshold = 0.1 *  wDt # the maximum profile height between the two zero crossings is greater than a threshold (typical value is 10% of Rz)
        
    wdsm = compute_Rsm(profil, dx, HeightThreshold, SpacingThreshold)
    
    Waveness_Params = Waveness_Params.append({'WDt' : wDt, 'WDsm':wdsm, 'WDc':wdc}, ignore_index=True)
    return Waveness_Params

def cal_waveness(data):
    Waveness_Params = pd.DataFrame(columns = ['WDt', 'WDsm', 'WDc'])
    n_y_data = data.groupby('y').mean()
    n_y = n_y_data.index

    for y in n_y:
        w_of_a_profil = cal_waveness_for_a_profil(data=data, y = y)
        Waveness_Params = Waveness_Params.append(w_of_a_profil, ignore_index= True)
    return Waveness_Params

############################################################TOPOGRAPHY#############################################################################
############################################################TOPOGRAPHY#############################################################################
############################################################TOPOGRAPHY#############################################################################
############################################################TOPOGRAPHY#############################################################################
############################################################TOPOGRAPHY#############################################################################
#compute all possible parameter of each profile of surface
def compute_topography_of_all_profiles(xyz_datei_path):
    
    data = pd.read_csv(xyz_datei_path, sep=';', names=['x', 'y', 'z'])
    xyz_minmax =[]
    for name_col in data:
        xyz_minmax.append([data[name_col].min(), data[name_col].max()])

    data['x'] -= xyz_minmax[0][0]
    data['y'] -= xyz_minmax[1][0]

    data = apply_gaussfilter_for_surface(data = data, lambdac = 2)

    #show_a_profil(data, 500)
    r = cal_Roughness_params(data)
    w = cal_waveness(data)
    topo = pd.concat([r,w], axis=1)
    return topo

#compute mean and standard deviation of each parameter  
def compute_topography_of_surface(xyz_datei_path):

    data = pd.read_csv(xyz_datei_path, sep=';', names=['x', 'y', 'z'])
    xyz_minmax =[]
    for name_col in data:
        xyz_minmax.append([data[name_col].min(), data[name_col].max()])

    data['x'] -= xyz_minmax[0][0]
    data['y'] -= xyz_minmax[1][0]

    data = apply_gaussfilter_for_surface(data = data, lambdac = 2)

    #show_a_profil(data, 500)
    r = cal_Roughness_params(data)
    w = cal_waveness(data)
    topo = pd.concat([r,w], axis=1)

    topo = topo.set_index('y')
    surface_topo = pd.DataFrame()
    for col_name in topo.columns: 
        m = topo[col_name].mean()
        std = topo[col_name].std()
        cov = std/m
        surface_topo['m_'+col_name]= [m]
        surface_topo['std_'+col_name] = [std]
        surface_topo['cov' + col_name] = [cov]
    return surface_topo
##############################################################MAIN#################################################################################
##############################################################MAIN#################################################################################
##############################################################MAIN#################################################################################
##############################################################MAIN#################################################################################
##############################################################MAIN#################################################################################
##############################################################MAIN#################################################################################

#t = compute_topography_of_surface(r'Klebeverbindungen_Daten\2D-MakroImages\Klebeverbindungen\ProbeE1_1_pointcloud-rel.txt')
#print(t)
#plt.plot(r.y, r.Ra, label='Ra')
#plt.plot(r.y, r.Rq, label='Rq')
#plt.plot(r.y, r.Rz, label = 'Rz')
#plt.plot(r.y, r.Rmax, label = 'Rmax')
#plt.plot(r.y, r.Rv, label)
#plt.plot(r.y, r.Rsm, label = 'Rsm')
#plt.plot(r.y, r.Rsk, label = 'Rsk')
#plt.plot(r.y, r.Rku, label = 'Rku')
#plt.legend()
#cal_roughness_params_for_a_profil(data , 500)