#Author: Phuong Pham 
#Datum: from 11.2021 - 


from os import sep
import os
import re
import numpy as np
from numpy.core.defchararray import index
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
from pandas.core.construction import array
import configparser


def distance_to_fitplane_cal(plane_normalvector, plane_position, point):
    qsum = 0.0
    dsum = 0.0
    for i in range(len(plane_normalvector)):
        qsum += plane_normalvector[i] * plane_normalvector[i]
        dsum += (point[i] - plane_position[i]) * plane_normalvector[i]

    return dsum / np.sqrt(qsum)


##################################################################################################################################################
##################################################################################################################################################
###################################################################Interpolation##################################################################
##################################################################################################################################################
##################################################################################################################################################


#read pointcloud as dataframe
def interpolation(data):
    print("data row count = ", data.shape)
    xyz_minmax = []
    for name_col in data:
        column = data[name_col]
        xyz_minmax.append([column.min(), column.max()])

    #set x,y of image
    data['x'] = (data['x'] - xyz_minmax[0][0]) 
    data['y'] = (data['y'] - xyz_minmax[1][0]) 
    data['z'] = (data['z'] - xyz_minmax[2][0]) 

    xyz_minmax.clear()
    for name_col in data:
        column = data[name_col]
        xyz_minmax.append([column.min(), column.max()])

    print(xyz_minmax)
    data = data.sort_values(by=['y', 'x'], ignore_index = True)

    n_y_data = data.groupby('y').sum()
    n_y = n_y_data.index


    in_data =  pd.DataFrame(columns=['x', 'y', 'z'])
    xx = pd.DataFrame({'xx':np.arange(xyz_minmax[0][1])})
    xx = xx.set_index('xx')
    for ny in n_y: 
        data_y = data.loc[data['y'] == ny]
        data_y = data_y.set_index('x')
        data_y = pd.concat([data_y, xx], axis= 1)
        data_y['y'] = data_y['y'].apply(lambda a: ny )
        data_y = data_y.interpolate(method ='linear', limit_direction ='forward')
        data_y.index.name = 'x'
        data_y = data_y.reset_index()
        in_data =  in_data.append(data_y)

    print(in_data)
    data = in_data
    data = data.sort_values(by=['x', 'y'], ignore_index = True)
    print(data)
    in_data =  pd.DataFrame(columns=['x', 'y', 'z'])
    yy = pd.DataFrame({'yy': np.arange(xyz_minmax[1][1])})
    yy = yy.set_index('yy')

    for nx in range(int(xyz_minmax[0][1])):
        data_x = data.loc[data['x'] == nx]
        data_x = data_x.set_index('y')
        data_x = pd.concat([data_x, yy], axis= 1)
        data_x['x'] = data_x['x'].apply(lambda a: nx )
        data_x = data_x.interpolate(method='linear', limit_direction= 'forward')
        data_x.index.name = 'y'
        data_x = data_x.reset_index()
        in_data = in_data.append(data_x)
    
    in_data = in_data.fillna(0)

    return in_data


##################################################################################################################################################
##################################################################################################################################################
###################################################################Visualization##################################################################
##################################################################################################################################################
##################################################################################################################################################

def from_pc_to_image(data_path):
    #read pointcloud as dataframe
    # data = pd.read_csv('ProbeE11_1_pointcloud.txt', sep=";", comment='#', header=None, names=['x', 'y', 'z'], index_col= None)
    # print("data row count = ", data.shape)
    xyz_minmax = []

    # for name_col in data:
    #     column = data[name_col]
    #     xyz_minmax.append([column.min(), column.max()])
    # print(xyz_minmax)

    data = pd.read_csv(data_path, sep=";", comment='#', header=None, names=['x', 'y', 'z'], index_col= None)

    #set x,y of image
    #data['x'] = (data['x'] - xyz_minmax[0][0]) 
    #data['y'] = (data['y'] - xyz_minmax[1][0]) 
    #data['z'] = (data['z'] - xyz_minmax[2][0]) 
    #xyz_minmax.clear()
    for name_col in data:
        column = data[name_col]
        xyz_minmax.append([column.min(), column.max()])
    print(xyz_minmax)
    #get width, height of image
    widthI = int(xyz_minmax[0][1] - xyz_minmax[0][0] +1)
    heightI = int(xyz_minmax[1][1] - xyz_minmax[1][0] +1)
    #normalize z value in greyvalue 0 - 255

    norscaler = (xyz_minmax[2][1] - xyz_minmax[2][0] )/255
    data['z'] = (data['z'] - xyz_minmax[2][0]) / norscaler
    #data['z'] [data['z'] < 0] = 0

    print(data.head(5))
    print(data.shape)

    dataI = np.zeros((heightI, widthI, 3), dtype=np.uint8)
    #dataI += [255,255,255]
    for index, row in data.iterrows():
        dataI[int(row['y']), int(row['x'])] = [row['z'], 0,0]

    #data.to_csv( 'out.csv',header = None, sep = ";", index = False)

    print(dataI.shape)

    image = Image.fromarray(dataI, "RGB")
    image.save("out12b.png")

    image.show()

def from_pc_to_image(data):
    #read pointcloud as dataframe
    # data = pd.read_csv('ProbeE11_1_pointcloud.txt', sep=";", comment='#', header=None, names=['x', 'y', 'z'], index_col= None)
    # print("data row count = ", data.shape)
    xyz_minmax = []

    for name_col in data:
        column = data[name_col]
        xyz_minmax.append([column.min(), column.max()])
    print(xyz_minmax)

    #data = pd.read_csv('interpolation_data12b.csv', sep=";", comment='#', header=None, names=['x', 'y', 'z'], index_col= None)

    #set x,y of image
    data['x'] = (data['x'] - xyz_minmax[0][0])/12
    data['y'] = (data['y'] - xyz_minmax[1][0])/20
    data['z'] = (data['z'] - xyz_minmax[2][0]) 
    xyz_minmax.clear()
    for name_col in data:
        column = data[name_col]
        xyz_minmax.append([column.min(), column.max()])
    print(xyz_minmax)
    #get width, height of image
    widthI = int(xyz_minmax[0][1] - xyz_minmax[0][0] +1)
    heightI = int(xyz_minmax[1][1] - xyz_minmax[1][0] +1)
    #normalize z value in greyvalue 0 - 255

    norscaler = (xyz_minmax[2][1] - xyz_minmax[2][0] )/255
    data['z'] = (data['z'] - xyz_minmax[2][0]) / norscaler
    #data['z'] [data['z'] < 0] = 0

    print(data.head(5))
    print(data.shape)

    dataI = np.zeros((heightI, widthI, 3), dtype=np.uint8)
    #dataI += [255,255,255]
    for index, row in data.iterrows():
        dataI[int(row['y']), int(row['x'])] = [row['z'], 0,0]

    #data.to_csv( 'out.csv',header = None, sep = ";", index = False)

    print(dataI.shape)

    image = Image.fromarray(dataI, "RGB").convert('L')
    return image 

def read_plane_configdata(configdata_path):
    config = configparser.ConfigParser()
    config.read(configdata_path)
    normalvector = [float(config['Plane']['nx']), float(config['Plane']['ny']), float(config['Plane']['nz'])]
    position =  [float(config['Plane']['px']), float(config['Plane']['py']), float(config['Plane']['pz'])]
    res = [normalvector, position]
    return res

def relativ_z_with_plane_cal(data, plane_normalvector, plane_position):
    data['z'] = distance_to_fitplane_cal(plane_normalvector, plane_position, [data['x'], data['y'], data['z']])
    return data



def cal_all_absdata_to_reldata(directory):     
    # assign directory
   
    # iterate over files in
    # that directory
    for filename in os.listdir(directory):
        f = os.path.join(directory, filename)
        # checking if it is a file
        if os.path.isfile(f):
            name = filename.split('.')
            planefilename = 'Plane/' + name[0]+ '-Plane.txt'
            data = pd.read_csv(f, sep=";", comment='#', header=None, names=['x', 'y', 'z'], index_col= None)
            planePath  = os.path.join(directory, planefilename)
            if os.path.isfile(planePath):
                planeinfo = read_plane_configdata(planePath)
                data = relativ_z_with_plane_cal(data=data, plane_normalvector= planeinfo[0], plane_position= planeinfo[1])
                os.path.join(directory, 'RelativData')
                data.to_csv( directory + 'RelativData/' +name[0] + "-rel.txt", sep = ';', index = False, header = None)

# path = "ProbeE11_2_pointcloud.txt"
# plane_path = 'plane112.txt'
# plane_info = read_plane_configdata(plane_path)
# print(plane_info)
# data = pd.read_csv(path, sep=";", comment='#', header=None, names=['x', 'y', 'z'], index_col= None)
# print(data)
# data = relativ_z_with_plane_cal(data, plane_info[0], plane_info[1])
# print(data)
# #data = interpolation(data=data)
# data.to_csv('ProbeE11_2_pointcloud-rel.txt', sep=";", index = False, header = None)
# #data = pd.read_csv('ProbeE11_1-rel.txt', sep=';', comment='#', header=None, names=['x', 'y', 'z'], index_col= None)
# image = from_pc_to_image(data=data)
# image.save("ProbeE11_2_depthmap-12-20.png")
# image.show()

def from_pc_to_image(data, zmax, zmin):
    xyz_minmax = []

    for name_col in data:
        column = data[name_col]
        xyz_minmax.append([column.min(), column.max()])
    #print(xyz_minmax)
    #set x,y of image
    data['x'] = (data['x'] - xyz_minmax[0][0])/12
    data['y'] = (data['y'] - xyz_minmax[1][0])/20
    #data['z'] = (data['z'] - xyz_minmax[2][0]) 
    xyz_minmax.clear()
    for name_col in data:
        column = data[name_col]
        xyz_minmax.append([column.min(), column.max()])
    #get width, height of image
    widthI = int(xyz_minmax[0][1] - xyz_minmax[0][0] +1)
    heightI = int(xyz_minmax[1][1] - xyz_minmax[1][0] +1)
    #normalize z value in greyvalue 0 - 255

    norscaler = (zmax - zmin)/255
    print(zmax, zmin, norscaler)
    data['z'] = (data['z'] - zmin) / norscaler
   
    dataI = np.zeros((heightI, widthI, 3), dtype=np.uint8)
    for row in data.iterrows():
        dataI[int(row['y']), int(row['x'])] = [row['z'], 0,0]
    #print(dataI)
  
    image = Image.fromarray(dataI, "RGB").convert('L')
   
    return image 

def getzminmax():
    directory = 'J:/MA/versagenanalyse/Klebeverbindungen_Daten/3d/RelativData'
        # iterate over files in
        # that directory
    zmin = 0
    zmax = 0
    for filename in os.listdir(directory):
        f = os.path.join(directory, filename)
        # checking if it is a file
        all_data = pd.DataFrame(columns=['x', 'y', 'z'])
        
        if os.path.isfile(f):
            data = pd.read_csv(f, sep=";", comment='#', header=None, names=['x', 'y', 'z'], index_col= None)
            n_z_min = data['z'].min()
            n_z_max = data['z'].max()
            #print(n_z_min, n_z_max)
            if(n_z_min < zmin): 
                zmin = n_z_min
            if(n_z_max > zmax): zmax = n_z_max

    return [zmin, zmax]

def GrayscaleImagesFromPC(zmax, zmin):
    directory = 'J:/MA/versagenanalyse/Klebeverbindungen_Daten/3d/'
        # iterate over files in
        # that directory
    for filename in os.listdir(directory+ 'RelativData/'):
        f = os.path.join(directory + 'RelativData/', filename)
        # checking if it is a file
        all_data = pd.DataFrame(columns=['x', 'y', 'z'])
      
        if os.path.isfile(f):
            name = filename.split('.')[0]
            data = pd.read_csv(f, sep=";", comment='#', header=None, names=['x', 'y', 'z'], index_col= None)
            image = from_pc_to_image(data=data, zmax= zmax, zmin= zmin)
            os.path.join(directory, 'GreyscaleImagesFromPC')
            image.save(directory + 'GreyscaleImagesFromPC/'+ name + '-depthmap.png')
   


directory = 'J:/MA/versagenanalyse/Klebeverbindungen_Daten/3d/'
cal_all_absdata_to_reldata(directory=directory)
zminmax = getzminmax()
print(zminmax)
GrayscaleImagesFromPC(zmin=zminmax[0], zmax=zminmax[1])
