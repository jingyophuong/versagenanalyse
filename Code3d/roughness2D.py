#Author: Phuong Pham 
#Datum: from 11.2021 - 
import hmac
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt
import cv2
import scipy.ndimage.filters as filters
###############################################################VISUALIZATION###############################################################################
###############################################################VISUALIZATION###############################################################################
###############################################################VISUALIZATION###############################################################################
###############################################################VISUALIZATION###############################################################################
def show_a_profil_(profil, y, filtered = True, h = False):
    
    profil = profil.sort_values(by=['x'])
    x = profil.x
    z = profil.z
  
    if not filtered: 
        res = gaussfilter(z, dx = 40)
        w = res[1]
        r = res[0]
    else: 
        w = profil.zw
        r = profil.zr
   
    plt.plot(x, z,  '-' ,label = 'P-Profil')
    plt.plot(x, w,  '-', label = 'W-Profil')
    plt.plot(x, r,  '-', label = 'R-Profil')
    title = 'Profil of surface by y = ' + str(y)
    if h: 
        title = 'Profil of surface by x = ' + str(y)
    plt.title(title)
    plt.legend()
    plt.show()

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

    z = 2 * np.sin(2 * np.pi * x/16) + 0.1 * np.random.random(n)
  #  z = signal.detrend(z)
    print(z)
    lambdac = 0.8
    #dx = 0.08

    gaussfilter(data = z, x = x, dx = dx, lambdac = lambdac)
    

def gaussfilter(profil,  dx):
   
    # zmax = np.max(profil) - np.min(profil)
  
    # factor = 1
    # if(zmax > 200):
    #     factor = int(np.log10(zmax)) +1
    #     zmax = zmax /(10**factor)
    # lambdac = 0
    # if zmax > 0.025 and zmax < 0.1:
    #     lambdac = 0.08
    # elif zmax >= 0.1 and zmax < 0.5:
    #     lambdac = 0.25
    # elif zmax >= 0.5 and zmax < 10:
    #     lambdac = 0.8
    # elif zmax >= 10 and zmax < 50:
    #     lambdac = 2.5
    # elif zmax >=50 and zmax < 200:
    #     lambdac = 8
    # lambdac *= (10**factor)
    # zmax *= (10**factor)

    #Ra = np.mean(np.abs(profil))
    dx /=1000
    # factor = 0
    # print(Ra)
    # if(Ra > 80):
    #     Ra /=1000
    #     factor = 3
    # lambdac = 0
    # if Ra > 0.006 and Ra < 0.02:
    #     lambdac = 0.08
    # elif Ra >= 0.02 and Ra < 0.1:
    #     lambdac = 0.25
    # elif Ra >= 0.1 and Ra < 2:
    #     lambdac = 0.8
    # elif Ra >= 2 and Ra < 10:
    #     lambdac = 2.5
    # elif Ra >=10 and Ra < 80:
    #     lambdac = 8
    # lambdac *= (10**factor)
    # Ra *= (10**factor)
    lambdac = 0.8
    # const = np.sqrt(np.log(2)/2/np.pi/np.pi)
   
    # xg = np.arange(-lambdac, lambdac, dx, dtype=float)
  
    # S =(1/(const *lambdac)) * np.exp(-np.pi*(xg/(const*lambdac))**2)
    
    # S = S/np.sum(S)
   
    # w = np.convolve(S, profil, 'same')
    #print(lambdac, dx)
    # r = profil -   w
    # return [r, w]
    
        #zero oder gaussian regression
    const = np.sqrt(np.log(2)/2/np.pi/np.pi)
    w = np.zeros(len(profil), dtype= float)
    for k in range(len(profil)): 
        p = np.arange(len(profil))  
        S = (1/np.sqrt(2*np.pi) / const / lambdac) * np.exp(-0.5 * ((k-p)*dx/const/lambdac)**2)
        Smod = S/sum(S)
        w[k] = sum(Smod * profil)
    r = profil - w
    return [r, w]
    
        

def apply_gaussfilter_for_all_profiles(data, dx, cut_prop = 5):
    n_y_data = data.groupby('y').mean()
    n_y = n_y_data.index
    filtered_data = pd.DataFrame(columns=['x', 'y', 'z', 'zr', 'zw'])

    cut_out = int(len(n_y) * cut_prop / 100)
    min_index = cut_out
    max_index = len(n_y) - cut_out -1
    n_y = n_y[min_index : max_index]

    for y in n_y:
               
        dataofy = data.loc[data['y'] == y]
        if(len(dataofy) < 10):
            continue
        dataofy = dataofy.sort_values(by=['x'])
        dataofy =  dataofy.reset_index(drop = True)
        dxs= np.abs(dataofy['x'].diff())
        
        dx = np.mean(dxs) * 0.01
        
        x = np.array(dataofy['x'])
        z = np.array(dataofy.z)
        
        
        profils = gaussfilter(data=z, x = x , dx= dx)
        #print(dataofy)
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



def compute_Rsm(r,dx,  HeightThreshold, SpacingThreshold): 
     #compute Rsm: as the average distance between zero crossings of the profile
        #find all zero crossing
    r = np.asarray(r)
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
# def cal_roughness_params_for_a_profil(data, y, dx): 
#     Roughness_Params = pd.DataFrame(columns=['y', 'Ra', 'Rq', 'Rz', 'Rmax',  'Rsm', 'Rsk', 'Rku'])
   
#     profil = data.loc[data['y'] == y]
#     profil = profil.sort_values(by=['x'])
#     if profil.empty:
#         return
    
#     profil = profil.reset_index(drop= True)
#     mean =  profil['zr'].mean() 
#     profil['zr'] = profil['zr'] - mean #shift profil to zero mean
#     xmax = profil['x'].max()
#     xmin = profil['x'].min()
#     dis = xmax - xmin +1 

#     step = int(dis/6)
#     if xmin > step or step < 48: 
#         return
#     f_limit = int(step /2) + xmin
#     r_a_sum = 0
#     r_z_sum = 0
#     r_z_n = []
#     r_q_sum = 0
#     r_sm_sum = 0
#     r_sk_sum = 0
#     r_ku_sum = 0
#     for i in range(5):
        
#         filter1 = profil['x'] >= f_limit + i* step 
#         filter2 =  profil['x'] <= f_limit + (i+1)*step
#         single_mess_section = profil.loc[filter1 & filter2]
#         if single_mess_section.empty:
#             return
#         single_mess_section =  single_mess_section.reset_index(drop = True)
#         #cal Ra
#         single_mess_section.__setitem__('absz', abs(single_mess_section.zr))
#         single_ra = single_mess_section.groupby('y').mean()
#         r_a_sum += single_ra['absz'].values[0] 

#         #cal Rz
#         Rz = single_mess_section['zr'].max() - single_mess_section['zr'].min()
#         r_z_sum += Rz

#         #find Rmax (or Rz1max)
#         r_z_n.append(Rz)

#         #cal Rq
#         single_mess_section.__setitem__('z2', single_mess_section.zr * single_mess_section.zr)
#         rq = single_mess_section.groupby('y').mean()['z2'].values[0]
#         r_q_sum += np.sqrt(rq)
        
#         # cal Rsm
#         dx = 0.01*dx#single_mess_section.dx.mean()
#         SpacingThreshold = 0.01 * len(single_mess_section)* dx #the spacing between the two zero crossing is greater than a threshold (typical value is 1% of the sampling length)
                
#         HeightThreshold = 0.1 * Rz # the maximum profile height between the two zero crossings is greater than a threshold (typical value is 10% of Rz)
        
    
#         r_sm_sum += compute_Rsm(profil=single_mess_section, dx = dx, HeightThreshold = HeightThreshold, SpacingThreshold = SpacingThreshold )
        
#         #compute Rsk
#         single_mess_section.__setitem__('z3', single_mess_section.z2 * single_mess_section.zr)
#         z3_mean = single_mess_section.groupby('y').mean()['z3'].values[0]
#         r_sk = (1/ pow(rq,3)) * z3_mean
#         r_sk_sum += r_sk
        
#         #compute Rku
        
#         single_mess_section.__setitem__('z4', single_mess_section.z3 * single_mess_section.zr)
#         z4_mean = single_mess_section.groupby('y').mean()['z4'].values[0]
#         r_ku = (1/ pow(rq,4)) * z4_mean
#         r_ku_sum += r_ku
        
    


#     Roughness_Params = Roughness_Params.append({'y' : y, 'Ra' : r_a_sum / 5, 'Rz': r_z_sum / 5, 
#                                                 'Rmax': np.max(r_z_n), 'Rq': r_q_sum / 5, 'Rsm': r_sm_sum/5,
#                                                'Rsk': r_sk_sum / 5, 'Rku': r_ku_sum /5} , ignore_index= True)
   
#     return Roughness_Params

def cal_roughness_params_for_a_profil(profil, dx, y): 
    Roughness_Params = pd.DataFrame(columns=['y', 'Ra', 'Rq', 'Rz', 'Rmax',  'Rsm', 'Rsk', 'Rku'])
   
    #profil = data.loc[data['y'] == y]
    profil = profil.sort_values(by=['x'])
    profil = profil.reset_index()
    #print(profil)
    if profil.empty:
        return
    if len(profil) < 50:
        return
    
    profil = profil.reset_index(drop= True)
    
    xmax = profil['x'].max()
    xmin = profil['x'].min()
    dis = xmax - xmin +1 

    step = int(dis/6)
    if step < dx: 
        return
    f_limit = int(step /2) + xmin
    b_limit = xmax - int(step /2)
    filter1 = profil['x'] >= f_limit
    filter2 =  profil['x'] <= b_limit
    single_mess_section = profil.loc[filter1 & filter2]
    #mean =  single_mess_section['zr'].mean() 
    #single_mess_section['zr'] = single_mess_section['zr'].apply(lambda x : x-mean) #shift profil to zero mea
    #print(single_mess_section)
    if single_mess_section.empty:
        return
    single_mess_section =  single_mess_section.reset_index(drop = True)
    #cal Ra
    single_mess_section.__setitem__('absz', abs(single_mess_section.zr))
    single_ra = single_mess_section.groupby('y').mean().absz.values[0]
  
    #cal Rz
    #Rz = single_mess_section['zr'].max() - single_mess_section['zr'].min()
   
    r_z_n = []
    for i in range(5):
        l = single_mess_section.x.min() + i*(single_mess_section.x.max() - single_mess_section.x.min()) / 5
        r = single_mess_section.x.min() + (i+1)*(single_mess_section.x.max() - single_mess_section.x.min()) / 5
        l_filter = single_mess_section.x >= l
        r_filter = single_mess_section.x <= r
        section = single_mess_section[l_filter & r_filter]
        r_z_n.append(section.zr.max() - section.zr.min())
    
    Rz = np.mean(r_z_n)
    #cal Rq
    single_mess_section.__setitem__('z2', single_mess_section.zr * single_mess_section.zr)
    rq = single_mess_section.groupby('y').mean()['z2'].values[0]
    
    rq = np.sqrt(rq)
    # cal Rsm
    #dxs= np.abs(single_mess_section['x'].diff())
        
    #dx = np.mean(dxs) * 0.01
    #dx = 0.01*dx#single_mess_section.dx.mean()
    SpacingThreshold = 0.01 * len(single_mess_section)* dx #the spacing between the two zero crossing is greater than a threshold (typical value is 1% of the sampling length)
            
    HeightThreshold = 0.1 * Rz # the maximum profile height between the two zero crossings is greater than a threshold (typical value is 10% of Rz)
    r_sm = compute_Rsm(single_mess_section.zr, dx = dx, HeightThreshold = HeightThreshold, SpacingThreshold = SpacingThreshold )
    
    #compute Rsk
    single_mess_section.__setitem__('z3', single_mess_section.z2 * single_mess_section.zr)
    z3_mean = single_mess_section.groupby('y').mean()['z3'].values[0]
    r_sk = (1/ pow(rq,3)) * z3_mean
    #compute Rku
    
    single_mess_section.__setitem__('z4', single_mess_section.z3 * single_mess_section.zr)
    #print(single_mess_section)
    z4_mean = single_mess_section.groupby('y').mean()['z4'].values[0]
    r_ku = (1/ pow(rq,4)) * z4_mean
    Roughness_Params = Roughness_Params.append({'y' : y, 'Ra' : single_ra, 'Rz': Rz, 
                                                'Rmax': np.max(r_z_n), 'Rq': rq, 'Rsm': r_sm,
                                               'Rsk': r_sk, 'Rku': r_ku} , ignore_index= True)
    #print(single_mess_section)
    return Roughness_Params


def cal_Roughness_params(data, dx): 
    n_y_data = data.groupby('y').mean()
    n_y = n_y_data.index
    
    Roughness_Params = pd.DataFrame(columns=['y', 'Ra', 'Rq', 'Rz', 'Rmax',  'Rsm', 'Rsk', 'Rku'])
  
    for y in n_y:
        r_of_a_profil = cal_roughness_params_for_a_profil(data=data, y = y, dx = dx)
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
# def cal_waveness_for_a_profil(data, y):
#     Waveness_Params = pd.DataFrame(columns = ['WDt', 'WDsm', 'WDc'])
#     profil = data.loc[data['y'] == y]
#     profil = profil.sort_values(by=['x'])
#     if profil.empty:
#         return
     
#     profil = profil.reset_index(drop= True)
#     mean =  profil.zw.mean() 
#     profil.zw = profil.zw - mean #shift profil to zero mean
#     xmax = profil['x'].max()
#     xmin = profil['x'].min()
#     dis = xmax - xmin +1 
#     step = int(dis/6)
#     if xmin > step: 
#         return
#     f_limit = int(step /2)
    
#     wc_sum = 0
#     for i in range(5):
        
#         filter1 = profil['x'] >= f_limit + i* step 
#         filter2 =  profil['x'] <= f_limit + (i+1)*step
#         single_mess_section = profil.loc[filter1 & filter2]
#         single_mess_section =  single_mess_section.reset_index(drop = True)
       
#         #cal Wc
#         Wct= single_mess_section['zr'].max() - single_mess_section['zr'].min()
#         wc_sum += Wct

    
#     #compute WDt
#     max_w = profil.zw.max()
#     min_w = profil.zw.min()
#     wDt = abs(max_w - min_w)
#     #Compute WDc
#     wdc  = wc_sum / 5
#     #Compute WDSm
#     dx = 0.12#single_mess_section.dx.mean()
#     SpacingThreshold = 0.01 * len(profil)* dx #the spacing between the two zero crossing is greater than a threshold (typical value is 1% of the sampling length)
                
#     HeightThreshold = 0.1 *  wDt # the maximum profile height between the two zero crossings is greater than a threshold (typical value is 10% of Rz)
        
#     wdsm = compute_Rsm(profil, dx, HeightThreshold, SpacingThreshold)
    
#     Waveness_Params = Waveness_Params.append({'WDt' : wDt, 'WDsm':wdsm, 'WDc':wdc}, ignore_index=True)
#     return Waveness_Params


def cal_waveness_for_a_profil(profil, dx):
    Waveness_Params = pd.DataFrame(columns = ['WDt', 'WDsm', 'WDc'])
    
    profil = profil.sort_values(by=['x'])
    if profil.empty:
        return
     
    profil = profil.reset_index()
    mean =  profil.zw.mean() 
    profil.zw = profil.zw - mean #shift profil to zero mean
    xmax = profil['x'].max()
    xmin = profil['x'].min()
    dis = xmax - xmin +1
    step = int(dis/6)
    if step < dx: 
        return
    f_limit = int(step /2) + xmin
    b_limit = xmax - int(step /2)
    filter1 = profil['x'] >= f_limit
    filter2 =  profil['x'] <= b_limit
    profil = profil.loc[filter1 & filter2]
    
    if(profil.empty):
        return
    Wct= profil['zw'].max() - profil['zw'].min()
    #compute WDt
    max_w = profil.zw.max()
    min_w = profil.zw.min()
    wDt = abs(max_w - min_w)

    #Compute WDSm
    
    SpacingThreshold = 0.01 * len(profil)* dx #the spacing between the two zero crossing is greater than a threshold (typical value is 1% of the sampling length)
                
    HeightThreshold = 0.1 *  wDt # the maximum profile height between the two zero crossings is greater than a threshold (typical value is 10% of Rz)
        
    wdsm = compute_Rsm(profil.zw, dx, HeightThreshold, SpacingThreshold)
    
    Waveness_Params = Waveness_Params.append({'WDt' : wDt, 'WDsm':wdsm, 'WDc':Wct}, ignore_index=True)
    return Waveness_Params

def cal_waveness(data):
    Waveness_Params = pd.DataFrame(columns = ['WDt', 'WDsm', 'WDc'])
    n_y_data = data.groupby('y').mean()
    n_y = n_y_data.index

    for y in n_y:
        w_of_a_profil = cal_waveness_for_a_profil(data=data, y = y)
        Waveness_Params = Waveness_Params.append(w_of_a_profil, ignore_index= True)
    return Waveness_Params

def cal_steigung_for_a_profil(profil):
    z = profil.z
   
    diff_z = np.diff(z)
  
    sum_diff_z = np.sum(diff_z)
    delta_x = np.sum(np.diff(profil.x))
   
    return sum_diff_z/delta_x

############################################################TOPOGRAPHY#############################################################################
############################################################TOPOGRAPHY#############################################################################
############################################################TOPOGRAPHY#############################################################################
############################################################TOPOGRAPHY#############################################################################
############################################################TOPOGRAPHY#############################################################################


#compute mean and standard deviation of each parameter  

def  compute_topography_of_surface_h(xyz_datei_path, stats = False):

    data = pd.read_csv(xyz_datei_path, comment = '#', sep=';', names=['x', 'y', 'z'])
  
    surface_topo = pd.DataFrame()
    yy = data.groupby('y').sum()
    yy = yy.index
  
    cut_out = int(len(yy) * 2.5 / 100)
    min_index = cut_out
    max_index = len(yy) - cut_out -1
    yy = yy[min_index : max_index]
    data = data.set_index('y')
  
    for y in yy: 
        profil = data.loc[y]
        if profil.empty:
            continue
        if len(profil) < 50:
            continue
        dx = np.diff(profil.x)
        dx = np.abs(np.mean(dx))
        res = gaussfilter(profil.z, dx = dx)
        #profil['zr'] = res[0]
        #profil['zw'] = res[1]
        profil = profil.assign(zr=res[0])
        profil = profil.assign(zw = res[1])
        r = cal_roughness_params_for_a_profil(profil = profil, dx = dx, y = y)
        w = cal_waveness_for_a_profil(profil = profil, dx = dx)
        if r.empty or w.empty:
            continue
        else: 
            rw = pd.concat([r,w], axis = 1)
            rw['gradient'] = cal_steigung_for_a_profil(profil=profil)
        #print(r)
            surface_topo = surface_topo.append(rw)

    if stats: 
        topo = pd.DataFrame()
        for col_name in surface_topo.columns: 
            m = surface_topo[col_name].mean()
            std = surface_topo[col_name].std()
            cov = std/m
            topo['m_'+col_name]= [m]
            topo['std_'+col_name] = [std]
            #surface_topo['cov' + col_name] = [cov]
        return topo
    #surface_topo = surface_topo.drop(['y'], axis = 1)
    return surface_topo

def compute_topography_of_a_probe_h(xyz_datei_path1, xyz_datei_path2):
    t1 = compute_topography_of_surface_h(xyz_datei_path1)
    t2 = compute_topography_of_surface_h(xyz_datei_path2)
    #t = pd.DataFrame()
    #t = t.append(t1.mean(), ignore_index=True)
    #t = t.append(t2.mean(), ignore_index= True)
    t = t1.append(t2) 
    t = t.drop(['y'], axis = 1)
    surface_topo = pd.DataFrame()
    for col_name in t.columns: 
        m = t[col_name].mean()
        #m = np.mean([t1[col_name].mean(), t2[col_name].mean()])
        #std = np.std([t1[col_name].mean(), t2[col_name].mean()])
        std = t[col_name].std()
        #diff_mean=np.abs(t1[col_name].mean() - t2[col_name].mean())
        #diff_std = np.abs(t1[col_name].std() - t2[col_name].std())
        #cov = std/m
        rmax = t[col_name].max()
        rmin = t[col_name].min()
        surface_topo['m_'+col_name]= [m]
        surface_topo['std_'+col_name] = [std]
        #surface_topo['dm_'+col_name]= [diff_mean]
        #surface_topo['dstd_'+col_name] = [diff_std]
        #surface_topo['max_'+ col_name] = [rmax]
        #surface_topo['min_'+ col_name] = [rmin]
    #print(surface_topo)
    return surface_topo

    
def compute_topography_of_a_probe_hv(xyz_datei_path1, xyz_datei_path2):
    #t1_h = compute_topography_of_surface_h(xyz_datei_path1).set_index('y')
    t1_v = compute_topography_of_surface_v(xyz_datei_path1).set_index('y')
    cols = t1_h.columns
    print(cols)
    #dyy =np.diff(t1_h.index)
    dxx =np.diff(t1_v.index)
    dy = dyy[0]
    #dx = dxx[0]
    t1 = pd.DataFrame()
    for col in cols:
      
        res_h = ((t1_h[col].sum() * 2 - t1_h[col].iloc[0] - t1_h[col].iloc[-1]) * 0.5 *dy) / np.sum(dyy)
        #res_v = ((t1_v[col].sum() * 2 - t1_v[col].iloc[0] - t1_v[col].iloc[-1]) * 0.5 *dx) / np.sum(dxx)
        t1[col + 'h'] = [res_h]
        #t1[col + 'v'] = [res_v]
    
    #t2 = compute_topography_of_surface(xyz_datei_path2)
    t2_h = compute_topography_of_surface_h(xyz_datei_path2).set_index('y')
    #t2_v = compute_topography_of_surface_v(xyz_datei_path2).set_index('y')
    cols = t2_h.columns
    dyy = np.diff(t2_h.index)
    #dxx = np.diff(t2_v.index)
    dy = dyy[0]
    #dx = dxx[0]
    t2 = pd.DataFrame()
    for col in cols:
        res_h = ((t2_h[col].sum() * 2 - t2_h[col].iloc[0] - t2_h[col].iloc[-1]) * 0.5 *dy) / np.sum(dyy)
        #res_v = ((t2_v[col].sum() * 2 - t2_v[col].iloc[0] - t2_v[col].iloc[-1]) * 0.5 *dx) / np.sum(dxx)
        t2[col + 'h'] = [res_h]
        #t2[col + 'v'] = [res_v]

    t = t1.append(t2) 
    surface_topo = pd.DataFrame()
    for col_name in t.columns: 
        m = t[col_name].mean()
        #m = np.mean([t1[col_name].mean(), t2[col_name].mean()])
        #std = np.std([t1[col_name].mean(), t2[col_name].mean()])
        std = t[col_name].std()
        diff_mean=np.abs(t1[col_name].mean() - t2[col_name].mean())
        #diff_std = np.abs(t1[col_name].std() - t2[col_name].std())
        #cov = std/m
        rmax = t[col_name].max()
        rmin = t[col_name].min()
        #surface_topo['m_'+col_name]= [m]
        surface_topo['std_'+col_name] = [std]
        surface_topo['dm_'+col_name]= [diff_mean]
        #surface_topo['dstd_'+col_name] = [diff_std]
        #surface_topo['max_'+ col_name] = [rmax]
        #surface_topo['min_'+ col_name] = [rmin]
    print(surface_topo)
    return surface_topo


def compute_topography_of_surface_v(xyz_datei_path, stats = False):

    data = pd.read_csv(xyz_datei_path, comment = '#', sep=';', names=['y', 'x', 'z'])
    
    surface_topo = pd.DataFrame()
    yy = data.groupby('y').sum()
    yy = yy.index
    cut_out = int(len(yy) * 1.5/ 100)
    min_index = cut_out
    max_index = len(yy) - cut_out -1
    yy = yy[min_index : max_index]
    data = data.set_index('y')
    for y in yy: 
        profil = data.loc[y]
        if profil.empty:
            continue
        if len(profil) < 50:
            continue
        dx = np.diff(profil.x)
        dx = np.abs(np.mean(dx))
        res = gaussfilter(profil.z, dx = dx)
        #profil['zr'] = res[0]
        #profil['zw'] = res[1]
        profil = profil.assign(zr=res[0])
        profil = profil.assign(zw = res[1])
        
        r = cal_roughness_params_for_a_profil(profil = profil, dx = dx, y = y)
        w = cal_waveness_for_a_profil(profil = profil, dx = dx)
        
        if r.empty or w.empty:
            continue
        else: 
            rw = pd.concat([r,w], axis = 1)
            rw['gradient'] = cal_steigung_for_a_profil(profil=profil)
            surface_topo = surface_topo.append(rw)

    if stats: 
        topo = pd.DataFrame()
        for col_name in surface_topo.columns: 
            m = surface_topo[col_name].mean()
            std = surface_topo[col_name].std()
            cov = std/m
            topo['m_'+col_name]= [m]
            topo['std_'+col_name] = [std]
            #surface_topo['cov' + col_name] = [cov]
        return topo
    #surface_topo = surface_topo.drop(['y'], axis = 1)
    
    return surface_topo

def compute_topography_of_a_probe_v(xyz_datei_path1, xyz_datei_path2):
    t1 = compute_topography_of_surface_v(xyz_datei_path1)
    t2 = compute_topography_of_surface_v(xyz_datei_path2)

    t = t1.append(t2) 
    t = t.drop(['y'], axis = 1)
    surface_topo = pd.DataFrame()
    for col_name in t.columns: 
        m = t[col_name].mean()
        std = t[col_name].std()
        cov = std/m
        rmax = t[col_name].max()
        rmin = t[col_name].min()
        surface_topo['m_'+col_name]= [m]
        surface_topo['std_'+col_name] = [std]
        #surface_topo['max_'+ col_name] = [rmax]
        #surface_topo['min_'+ col_name] = [rmin]
    #print(surface_topo)
    return surface_topo



def compute_histogram_topography_of_a_probe_h(xyz_datei_path1, xyz_datei_path2):
    t1 = compute_topography_of_surface_h(xyz_datei_path1)
    t2 = compute_topography_of_surface_h(xyz_datei_path2)
    #t = pd.DataFrame()
    #t = t.append(t1.mean(), ignore_index=True)
    #t = t.append(t2.mean(), ignore_index= True)
    t = t1.append(t2) 
    t = t.drop(['y'], axis = 1)
    #surface_topo = pd.DataFrame()
    range = pd.DataFrame({'Ra': [0,300],
                        'Rq': [0,300], 
                        'Rz': [0,500], 
                        'Rmax': [0,1000],
                        'Rsm':[0,5000],
                        'Rsk': [-100, 100], 
                        'Rku' : [0,300], 
                        'WDt': [0,3000], 
                        'WDsm': [0,10000], 
                        'WDc': [0,3000]
                        })
    bins = 256
    num = len(t)
    data = []
    for col_name in t.columns: 
        o_rang = int(range[col_name].loc[0])
        u_rang = int(range[col_name].loc[1])
        if u_rang - o_rang <= 500: 
            bins = int((u_rang - o_rang) / 5)
        else: 
            bins = int((u_rang - o_rang) / 50)
        hist =  np.reshape(cv2.calcHist([np.float32(t[col_name].values)], [0], None, [bins], [o_rang, u_rang]),(bins))/num * 100
        data = np.hstack([data, hist])
        #print(hist)
    return data
##############################################################MAIN#################################################################################
##############################################################MAIN#################################################################################
##############################################################MAIN#################################################################################
##############################################################MAIN#################################################################################
##############################################################MAIN#################################################################################
##############################################################MAIN#################################################################################
if __name__ == '__main__':
   

    #path1 = "Klebeverbindungen_Daten\AP5-3D Punktwolken\BM-TEST\FinalPC\ProbeR26_1.txt"
    #path2 = "Klebeverbindungen_Daten\AP5-3D Punktwolken\BM-TEST\FinalPC\ProbeE22_2.txt"
    path1 = "Klebeverbindungen_Daten\AP5-3D Punktwolken\BM\FinalPC\ProbeE6_1.txt"
    path2 = "Klebeverbindungen_Daten\AP5-3D Punktwolken\BM\FinalPC\ProbeE6_2.txt"
    #path1 =  "Klebeverbindungen_Daten\AP5-3D Punktwolken\SK\FinalPC\ProbeR9_1.txt"
    #r = compute_histogram_topography_of_a_probe_h(path1, path2)
    #print(r)
    #r.to_csv('r.csv', sep = ';', decimal=',', float_format='%.3f')
    #print(r)

    # t1 = compute_topography_of_surface_h(path1)
    # t1.to_csv('t1E6.csv', sep = ';', decimal=',', float_format='%.3f')
    # print(t1)
    # t2 = compute_topography_of_surface_h(path2)
    # t2.to_csv('t2E6.csv', sep = ';', decimal=',', float_format='%.3f')
    #print(t2)
    data = pd.read_csv(path1, comment = '#', sep=';', names=['x', 'y', 'z'])
    
    yy = data.groupby('y').sum().index
    data = data.set_index('y')
    y = yy[200]
    profil = data.loc[y]
    dx = np.diff(profil.x)
    dx = np.abs(np.mean(dx))
   
    res = gaussfilter(profil.z, dx = dx)
            
    profil = profil.assign(zr=res[0])
    profil = profil.assign(zw = res[1])
    #show_a_profil_(profil, y )
    a = cal_steigung_for_a_profil(profil=profil)
    print(a)
    # print(compute_topography_of_surface(path1))
    # path3 = r'J:\MA\versagenanalyse\Klebeverbindungen_Daten\AP5-3D Punktwolken\BM\RelativDataH\ProbeE1_1-relH.txt'
    # path4 = r'J:\MA\versagenanalyse\Klebeverbindungen_Daten\AP5-3D Punktwolken\BM\RelativDataH\ProbeE1_2-relH.txt'   
    # #r = compute_topography_of_a_probe_h(path3, path4)

    # #print(r)

    # data = pd.read_csv(path3, comment = '#', sep=';', names=['y', 'x', 'z'])
  
    # yy = data.groupby('y').sum()
    # yy = yy.index
    # print(yy)
    # data = data.set_index('y')
    # y = yy[300]
    # profil = data.loc[y]
    # print(profil)
    # dx = np.diff(profil.x)
    
    # dx = np.abs(np.mean(dx))
    # print(dx)
    # res = gaussfilter(profil.z, dx = dx)
    # profil['zr'] = res[0]
    # profil['zw'] = res[1]
    # r = cal_roughness_params_for_a_profil(profil, dx = dx, y= y)
    # w = cal_waveness_for_a_profil(profil = profil, dx = dx)
    # rw = pd.concat([r,w], axis = 1)
    # print(rw)
    # show_a_profil_(profil, y = y, h = True)