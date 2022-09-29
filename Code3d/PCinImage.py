#Author: Phuong Pham 
#Datum: from 11.2021 - 


import imghdr
from msilib.schema import Directory
from os import sep
import os
import re
import cv2
from cv2 import namedWindow
import numpy as np
from numpy.core.defchararray import index
import pandas as pd
from PIL import Image, ImageOps
import matplotlib.pyplot as plt
from pandas.core.construction import array
import configparser
import roughness2D



def distance_to_fitplane_cal(plane_normalvector, plane_position, point):

    qsum = 0.0
    dsum = 0.0
    if(plane_normalvector[2] < 0): 
        plane_normalvector = np.multiply(plane_normalvector, -1)
        print(plane_normalvector)
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
  

    #print(xyz_minmax)
    data = data.sort_values(by=['y', 'x'], ignore_index = True)

    n_y_data = data.groupby('y').sum()
    n_y = n_y_data.index


    in_data =  pd.DataFrame(columns=['x', 'y', 'z'])
    #xx = pd.DataFrame({'xx':np.arange(xyz_minmax[0][1])})
    #xx = xx.set_index('xx')

    for ny in n_y: 
        data_y = data.loc[data['y'] == ny]
        data_y = data_y.set_index('x')
      
        if not(data_y.index.is_unique):
            data_y = data_y[~data_y.index.duplicated(keep='first')]
          
        xx = pd.DataFrame({'xx': np.arange(np.min(data_y.index),np.max(data_y.index+1))})
        xx = xx.set_index('xx')
    
        data_y = pd.concat([data_y, xx], axis= 1)
        data_y['y'] = data_y['y'].apply(lambda a: ny )
       
        data_y = data_y.interpolate(method ='linear', limit_direction ='forward')
        data_y.index.name = 'x'
        data_y = data_y.reset_index()
        in_data =  in_data.append(data_y)
    in_data = in_data.dropna()
    #print(in_data)
    data = in_data
    #in_data.clear_data()
    data = data.sort_values(by=['x', 'y'], ignore_index = True)
    #print(data)
    #in_data =  pd.DataFrame(columns=['x', 'y', 'z'])
    
    n_x_data = data.groupby('x').sum()
    n_x = n_x_data.index
    #print(n_x)
    for nx in n_x:
        data_x = data.loc[data['x'] == nx]
        data_x = data_x.set_index('y')
        yy = pd.DataFrame({'yy': np.arange(np.min(data_x.index),np.max(data_x.index+1), 10)})
        yy = yy.set_index('yy')
        data_x = pd.concat([data_x, yy], axis= 1)
        data_x['x'] = data_x['x'].apply(lambda a: nx )
        data_x = data_x.interpolate(method='linear', limit_direction= 'forward')
        #print(data_x)
        data_x.index.name = 'y'
        data_x = data_x.reset_index()
        in_data = in_data.append(data_x)
    
    in_data = in_data.dropna()
    #print(in_data)
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
    data['z'] = distance_to_fitplane_cal(plane_normalvector, plane_position, [data['x'],data['y'], data['z']])
    return data



def cal_all_absdata_to_reldata(absdata_dir, plane_dir):     
    # assign directory
   
    # iterate over files in
    # that directory
    files = os.listdir(absdata_dir)
    print(files)
    
    for filename in files:
        if('Probe' in filename):
            f = os.path.join(absdata_dir, filename)
            # checking if it is a file
            if os.path.isfile(f):
                name = filename.split('.')
                planefilename = name[0]+ '-Plane.txt'
                data = pd.read_csv(f, sep=";", comment='#', header=None, names=['x', 'y', 'z'], index_col= None).astype(float)
                planePath  = os.path.join(plane_dir, planefilename)
                if os.path.isfile(planePath):
                    planeinfo = read_plane_configdata(planePath)
                    print(planeinfo)
                    data = relativ_z_with_plane_cal(data=data, plane_normalvector= planeinfo[0], plane_position= planeinfo[1])
                    
                    os.path.join(absdata_dir + "../", 'RelativData')
                    data.to_csv( plane_dir + "../" + 'RelativData/' +name[0] + "-rel.txt", sep = ';', index = False, header = None)



def from_pc_to_image(data, zmax, zmin):
    xyz_minmax = []

    for name_col in data:
        column = data[name_col]
        xyz_minmax.append([column.min(), column.max()])
    #print(xyz_minmax)
    #set x,y  of image
    
    dx = 40
    #dy = 100
    dy = 45
    #dy = 55
    # dy = data.groupby(['y']).sum().sort_index()
    # dx = data.loc[data.y == dy.index[100]].x
    # dx = int(np.abs(np.max(np.diff(dx))))
    # dy = int(np.abs(np.max(np.diff(dy.index))))
    # print(dx,dy)
    #dx = int(np.mean(np.diff(dx)) + 0.5)
    #dy = int(np.mean( np.diff(dy.index))+0.5)
    #print(dx, dy)

    data['x'] = ((data['x'] - xyz_minmax[0][0])/dx).astype('int')
    data['y'] = ((data['y'] - xyz_minmax[1][0])/dy).astype('int')
    #data['z'] = (data['z'] - xyz_minmax[2][0]) 

    data = interpolation(data)
    xyz_minmax.clear()
    for name_col in data:
        column = data[name_col]
        xyz_minmax.append([column.min(), column.max()])
    #get width, height of image
    widthI = int(xyz_minmax[0][1] - xyz_minmax[0][0] +1)
    heightI = int(xyz_minmax[1][1] - xyz_minmax[1][0] +1)
    #normalize z value in greyvalue 0 - 255

    norscaler = (zmax - zmin)/65535
    #print(zmax, zmin, norscaler)
    data['z'] = (data['z'] - zmin) / norscaler
    #print(np.max(data.z))
    #plt.hist(np.asarray(data.z), bins = 256)
    #plt.show()
    dataI = np.zeros((heightI, widthI), dtype=np.uint16)
    for index, row in data.iterrows():
        y = int(row['y'])
        x = int(row['x'])
        dataI [y,x] = row['z']
        #dataI[y,x] = [row['z'], 0,0]
    #print(dataI)
  
    image = Image.fromarray(dataI, 'I;16')
   
    return image 


def get_pointcloud_in_horizontal(data, step):
    minx = int(data.x.min())
    maxx = int(data.x.max())
    print(minx, maxx)
    res = pd.DataFrame(columns=['x', 'y', 'z'])
    for i in range(minx,maxx,step):
        xx = data.loc[data['x'] == i]
        res = res.append(xx)
    return res

def get_all_horizontal_pointcloud(directory, step =11):
    for filename in os.listdir(directory):
        f = os.path.join(directory , filename)
    
        if os.path.isfile(f):
            print(f)
            name = filename.split('.')[0]
            data = pd.read_csv(f, sep=";", comment='#', header=None, names=['x', 'y', 'z'], index_col= None)
            data = interpolation(data = data)
            #print(f)
            data = get_pointcloud_in_horizontal(data, step)
            os.path.join(directory + "../", 'HorizontalProfiles')
            data.to_csv(directory + "../"+'HorizontalProfiles/'+ name + '.txt', sep = ';', index = False, header = None)


def test_cal_abstorel():
    path1 = "Klebeverbindungen_Daten\AP5-3D Punktwolken\BM\Data\ProbeE12_1.csv"
    #path2 = "Klebeverbindungen_Daten\AP5-3D Punktwolken\BM\RelativData\ProbeE12_1-rel.txt"
    data1 = pd.read_csv(path1, sep=";", comment='#', header=None, names=['x', 'y', 'z'], index_col= None).astype(float)
    #data2 =  pd.read_csv(path2 ,sep=';', names=['x', 'y', 'z'])
    
    planePath  = "Klebeverbindungen_Daten\AP5-3D Punktwolken\BM\Plane\ProbeE12_1-Plane.txt"
    if os.path.isfile(planePath):
        planeinfo = read_plane_configdata(planePath)
        print(planeinfo)
        data = relativ_z_with_plane_cal(data=data1, plane_normalvector= planeinfo[0], plane_position= planeinfo[1])
        data.to_csv('teste12.txt', sep = ';', index = False, header = None)

def GrayscaleImagesFromPC(zmax, zmin, directory):
    
        # iterate over files in
        # that directory
    for filename in os.listdir(directory+ 'RelativData/'):

        f = os.path.join(directory + 'RelativData/', filename)
        # checking if it is a file
        all_data = pd.DataFrame(columns=['x', 'y', 'z'])
      
        if os.path.isfile(f): #and 'S7_1' in f:
            #if('S7_2' in f or 'S12' in f or 'S18_2' in f or 'S21_2' in f):
            name = filename.split('.')[0]
            data = pd.read_csv(f, sep=";", comment='#', header=None, names=['x', 'y', 'z'], index_col= None)
            image = from_pc_to_image(data=data, zmax= zmax, zmin= zmin)
            #image = crop_minAreaRect(np.array(image))
            os.path.join(directory, 'GrayscaleImagesFromPC')
            image.save(directory + 'GrayscaleImagesFromPC/'+ name + '-depthmap.png')

def crop_minAreaRectForAllImage(directory):
       # that directory
    for filename in os.listdir(directory):
        print(filename)
        if os.path.isfile(directory + filename): #and 'S7_1' in filename:
            img = cv2.imread(directory+filename, -1)
            #print(img)
            img = crop_minAreaRect(img)
            cv2.imwrite(directory + '../Final/' + filename, img)

def getzminmax(directory):
   
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
            print(filename, n_z_min, n_z_max, n_z_max - n_z_min)
            n_z_max -= n_z_min
            
            if(n_z_min < zmin): 
                zmin = n_z_min
            if(n_z_max > zmax): zmax = n_z_max

    #return [zmin, zmax]
    return [0, zmax]

def add_border(img,  border):
    
    if isinstance(border, int) or isinstance(border, tuple):
        bimg = ImageOps.expand(img, border=border)
    else:
        raise RuntimeError('border is not an integer or tuple!')

    return bimg
    #bimg.save(output_image)
def noise_reduction_with_histogram(data, bins = 256, per= 2):
    noise = int(len(data)/ (bins*100) * per)
   
    z = np.array(data.z)
    hist, bins = np.histogram(z, bins = bins)
  
    hist[hist<=noise] = 0
    hist_0_d = np.where(hist == 0)[0]
    #plt.bar(x = bins[0:-1], height = hist, width=10)
    #plt.show()
    z_d = np.digitize(z, bins)    
    for hist_0_i in hist_0_d:
        z[z_d==hist_0_i] = float('nan')
    data['z'] = z
    print(data)
    data= data.dropna()
    return data

def noise_reduction_for_all_files(directory):
    for filename in os.listdir(directory):
        f = os.path.join(directory, filename)
        if os.path.isfile(f):
            data = pd.read_csv(f, sep=";", comment='#', header=None, names=['x', 'y', 'z'], index_col= None)
            nr_data = noise_reduction_with_histogram(data)
            os.path.join(directory + "../", 'NoiseReduced')
            nr_data.to_csv(directory + "../"+'NoiseReduced/'+ filename, sep = ';', index = False, header = None)
        else:
            print('directory dont exist')



def crop_minAreaRect(img):
    row, col = img.shape[:2]
    bottom = img[row-2:row, 0:col]
    mean = cv2.mean(bottom)[0]

    bordersize = 30
    img = cv2.copyMakeBorder(
        img,
        top=bordersize,
        bottom=bordersize,
        left=bordersize,
        right=bordersize,
        borderType=cv2.BORDER_CONSTANT,
        value=0
    )
    #img2  = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY )
    img2 = (img/256).astype('uint8')
    kernel = np.ones((21,21),np.uint8)
    img2 = cv2.morphologyEx(img2, cv2.MORPH_CLOSE, kernel)
    #print(img2)
    plt.imshow(img2)
    #plt.show()
    t, img2 = cv2.threshold(img2,0,255,cv2.THRESH_OTSU)
    plt.imshow(img2)
    #plt.show()
    contours, hierarchy = cv2.findContours(img2, cv2.RETR_TREE  , cv2.CHAIN_APPROX_SIMPLE)
    print("num of contours: {}".format(len(contours)))

    mult = 1  # I wanted to show an area slightly larger than my min rectangle set this to one if you don't
    #img_box = cv2.cvtColor(img.copy(), cv2.COLOR_GRAY2BGR)
    #for cnt in contours:
    cnt = contours[0]
    rect = cv2.minAreaRect(cnt)
    #print(rect)
    box = cv2.boxPoints(rect)
    box = np.int0(box)
    #cv2.drawContours(img, [box], 0, (0,255,0), 2) # this was mostly for debugging you may omit

    W = rect[1][0]
    H = rect[1][1]

    Xs = [i[0] for i in box]
    Ys = [i[1] for i in box]
    x1 = min(Xs)
    x2 = max(Xs)
    y1 = min(Ys)
    y2 = max(Ys)

    rotated = False
    angle = rect[2]
    print(angle)
    if angle > 45:
        angle +=270
        rotated = True

    center = (int((x1+x2)/2), int((y1+y2)/2))
    size = (int(mult*(x2-x1)),int(mult*(y2-y1)))
    #cv2.circle(img, center, 10, (0,255,0), -1) #again this was mostly for debugging purposes

    M = cv2.getRotationMatrix2D((size[0]/2, size[1]/2), angle, 1.0)

    #cropped = cv2.getRectSubPix(img, size, center)    
    cropped =  img[y1:y1+size[1], x1:x1+size[0]]
    cropped = cv2.warpAffine(cropped, M, size)

    croppedW = W if not rotated else H 
    croppedH = H if not rotated else W

    #croppedRotated = cv2.getRectSubPix(cropped, (int(croppedW*mult), int(croppedH*mult)), (size[0]/2, size[1]/2))
    x = int(size[0] / 2 - croppedW/2)
    y = int(size[1]/2 - croppedH /2)
    print(x,y)
    croopedRotated = cropped[y: y + int(croppedH), x: x + int(croppedW)]

    
    
    return croopedRotated

def deapthmapTOPointCloud(input_dir, dy, dx, output_dir):

    for filename in os.listdir(input_dir):
       
        f = os.path.join(input_dir, filename)
        data_h = pd.DataFrame()
        if os.path.isfile(f): #and 'S7_1' in f:
            print(f)
            #if ('R6' in f):
            img = cv2.imread(f, -1)
            img = img.astype('float')
            img[img == 0] = 'nan'
            z_h = np.zeros((img.shape), dtype = float)
            norscaler = 65535/5000
            z_h= img / norscaler -2500
        
        
            for i in range(z_h.shape[0]):
                h = pd.DataFrame(columns=['x', 'y', 'z'])
                y = (np.zeros([z_h.shape[1]]) + dy )* i
                x = np.arange(z_h.shape[1])* dx
                z = z_h[i]
                h.x = x
                h.y = y
                h.z = z
                data_h = data_h.append(h)
            data_h = data_h.dropna()

            name =   filename.split('-')[0]
            data_h.to_csv(output_dir + name + '.txt', sep=';', header = None, index = False)
    

if __name__ == '__main__1':
    f = ''
    img = cv2.imread(f, -1)
    img = img.astype('float')
    img[img == 0] = 'nan'
    z_h = np.zeros((img.shape), dtype = float)
    norscaler = 65535/5000
    z_h= img / norscaler -2500
    z_v = z_h.T
    print(z_h.shape, z_v.shape)


    for i in range(z_h.shape[0]):
        h = pd.DataFrame(columns=['x', 'y', 'z'])
        y = (np.zeros([z_h.shape[1]]) + dy )* i
        x = np.arange(z_h.shape[1])* dx
        z = z_h[i]
        h.x = x
        h.y = y
        h.z = z
        data_h = data_h.append(h)
    data_h = data_h.dropna()

    name =   filename.split('-')[0]
    data_h.to_csv(output_dir + name + '.txt', sep=';', header = None, index = False)
if __name__ == '__main__':

    # n  = 'E_2'

    # img = cv2.imread('Klebeverbindungen_Daten\AP5-3D Punktwolken\BM\GrayscaleImagesFromPC\Probe' + n + '-rel-depthmap.png')
        
    # cropimg = crop_minAreaRect(img)
    # plt.imshow(cropimg)
    # plt.show()
    # cv2.imwrite('Klebeverbindungen_Daten\AP5-3D Punktwolken\BM\Final\Probe' + n + '-rel-depthmap.png', cropimg)
    # dir = r'Klebeverbindungen_Daten/AP5-3D Punktwolken/SK-TEST/'
    # absdir = dir +'Data/'
    # planedir = dir + 'Plane/'
    # reldir = dir + 'RelativData/'
    # #ndir =  r'Klebeverbindungen_Daten/AP5-3D Punktwolken/BM/NoiseReduced/'
    # imdir = dir + 'GrayscaleImagesFromPC/'
    # outdir =  dir + 'FinalPC/'
    # #cal_all_absdata_to_reldata(absdata_dir=absdir, plane_dir=planedir)
    # #GrayscaleImagesFromPC(zmin = -2500, zmax = 2500, directory=dir)
    # #crop_minAreaRectForAllImage(imdir)
    # deapthmapTOPointCloud(input_dir=dir + 'Final/', dy = 45, dx = 40, output_dir=outdir)

    
    #
    #get_all_horizontal_pointcloud(directory=r'Klebeverbindungen_Daten/AP5-3D Punktwolken/BM/RelativData/', step = 45)
    #noise_reduction_for_all_files(reldir)    
    

    #zminmax = getzminmax(dir)
    #print(zminmax)
    #data = pd.read_csv(r'Klebeverbindungen_Daten\AP5-3D Punktwolken\BM\RelativData\ProbeS17_1-rel.txt', sep=";", comment='#', header=None, names=['x', 'y', 'z'], index_col= None)
    #data.hist(column = 'z', bins = 255)
    # print(np.max(data.z) - np.min(data.z))
    #plt.show()
    # #data = interpolation(data = data)
    
   
    # im = cv2.imread(r'Klebeverbindungen_Daten\AP5-3D Punktwolken\BM\GrayscaleImagesFromPC\ProbeS17_1-rel-depthmap.png')
    # # calculate mean value from RGB channels and flatten to 1D array
    # vals = im.mean(axis=2).flatten()
    # # plot histogram with 255 bins
    # b, bins, patches = plt.hist(vals, 255)
    # plt.xlim([0,255])
    # plt.show()
    #plt.imshow(img)
    #plt.show()
   
    # directory = 'J:/MA/versagenanalyse/Klebeverbindungen_Daten/AP5-3D Punktwolken/BM/GrayscaleImagesFromPC'
    
    # maxshape = 512
    # for filename in os.listdir(directory):
    #     f = os.path.join(directory, filename)
    #     img = Image.open(f)
    #     w,h = img.size
    #     bx = int((maxshape - w ) / 2)
    #     by = int((maxshape - h) / 2)
    #     bimg = add_border(img,  border=(bx, by, maxshape - w -bx, maxshape - h - by))
    #     bimg.save(f)
    # print(maxshape)


    #get_all_horizontal_pointcloud(directory=dir, step = 35)
  
    


    absdir = r'Klebeverbindungen_Daten/AP5-3D Punktwolken/SK/Data/'
    planedir = r'Klebeverbindungen_Daten/AP5-3D Punktwolken/SK/Plane/'
    reldir = r'Klebeverbindungen_Daten/AP5-3D Punktwolken/SK/RelativData/'
    dir =  r'Klebeverbindungen_Daten/AP5-3D Punktwolken/SK/'
    hdir = r'Klebeverbindungen_Daten/AP5-3D Punktwolken/SK/HorizontalProfiles/'
    inDir = r'Klebeverbindungen_Daten/AP5-3D Punktwolken/SK/Interpolation/'
    imdir = 'Klebeverbindungen_Daten/AP5-3D Punktwolken/SK/'
    outdir =  'Klebeverbindungen_Daten/AP5-3D Punktwolken/SK/FinalPC/'
    #crop_minAreaRectForAllImage(imdir)
    #deapthmapTOPointCloud(input_dir=imdir, dy = 100, dx = 40, output_dir=outdir)
    # isdir = os.path.isdir( reldir)
    # if not isdir: 
    #     os.mkdir(reldir) 
    #cal_all_absdata_to_reldata(absdata_dir=absdir, plane_dir=planedir)
    # isdir = os.path.isdir(dir)
    # if not isdir: 
    #     os.makedirs(dir) 
    #noise_reduction_for_all_files(reldir)
    # isdir = os.path.isdir(hdir)    
    # if not isdir: 
    #     os.makedirs(hdir) 
    #get_all_horizontal_pointcloud(dir, step = 35)
    #data = pd.read_csv(reldir + 'ProbeE7_1-rel.txt', sep=";", comment='#', header=None, names=['x', 'y', 'z'], index_col= None)
    #indata = interpolation(data)
    #indata.to_csv(inDir+'ProbeE7_1.txt', sep=';')
    #img = from_pc_to_image(data=data, zmin= -2500, zmax= 2500)
    #img.save('ProbeE7_1-depthmap.png')
    #GrayscaleImagesFromPC(zmin = -2500, zmax = 2500, directory=dir)
    #plt.imshow(img)
    #plt.show()

    #dir  = r'Klebeverbindungen_Daten\2D-MakroImages\BM'
    # samples = pd.read_csv("Klebeverbindungen_Daten/Proben.csv" ,sep=';', error_bad_lines=False)
    # samples = samples.set_index("ID")
    # targets = []
    # #feature_names = ['Sa', 'Sq', 'Sz', 'Ssk', 'Ssu']
   
    # maxheight = 1027
    # for index, row in samples.iterrows():
    #     path1 = dir + "\Probe" + index + "_1.png"
    #     path2 = dir + "\Probe" + index + "_2.png"
    #     if(os.path.isfile(path1) and os.path.isfile(path2) and row['stress angle'] != 90 ):
    #         img1 = Image.open(path1)
    #         img2 = Image.open(path2)
            
    #         #if(maxheight < img1.size[1] + img2.size[1]):
    #         #   maxheight = img1.size[1] + img2.size[1]
    # #print(maxheight)
    #         im = Image.new(mode = 'RGB', size=(maxheight, maxheight))
    #         if np.max(img1.size) > maxheight:
    #             new_size = (maxheight, int(maxheight * np.min(img1.size)  / np.max(img1.size)))
    #             img1 = img1.resize(new_size)
    #         if np.max(img2.size) > maxheight:
    #             new_size = (maxheight, int(maxheight * np.min(img2.size)  / np.max(img2.size)))
    #             img2 = img2.resize(new_size)
    #         x1 = int((maxheight- img1.size[1] - img2.size[1] -5) /2)
    #         x2 = x1 + img1.size[1] + 5
    #         x3 = int((maxheight- img1.size[0]) /2)
    #         x4 = int((maxheight- img2.size[0])/2)

    #         im.paste(img1, ( x3, x1))
    #         im.paste(img2, ( x4, x2) )
    #         im.save(dir + '\\' + str(row['stress angle']) + '\\'+ index + '.png')

    # new_width = 846
    # for f in os.listdir(dir):
    #     if(os.path.isfile(dir + '\\'+ f)):
    #         print(f)
    #                 # Load the image
    #         img = cv2.imread(dir + '\\'+ f)

    #         # Add salt-and-pepper noise to the image.
    #         noise_img = random_noise(img, mode='s&p',amount=0.2)

    #         # The above function returns a floating-point image
    #         # on the range [0, 1], thus we changed it to 'uint8'
    #         # and from [0,255]
    #         noise_img = np.array(255*noise_img, dtype = 'uint8')

    #         # Display the noise image
    #         cv2.imwrite(dir + '\\..\\BM3_noise\\' +f , noise_img)

    # new_width = 868
    # for f in os.listdir(dir):
    #     if(os.path.isfile(dir + '\\'+ f)):
    #         print(f)

    #         img = Image.open(dir + '\\' + f)
    #         new_height = int(new_width * img.size[1] / img.size[0])
    #         new_img = img.resize((new_width, new_height))
    #         new_img.save(dir + '\\..\\SK2_klein\\' +f )
                

    file = 'Klebeverbindungen_Daten\Proben.csv'
    proben = pd.read_csv(file, sep=';')
    
    bm = proben[proben['adhasiv'] == 'BM']
    sk = proben[proben['adhasiv'] == 'SK']

    bmrt2 = bm[bm['stress rate'] == '2']
    bmrt2 = bmrt2[bmrt2['temperature'] == 'RT']
    bmrt2 = bmrt2[bmrt2['Proben-Kennzeichen'] != 'Proben bei uns nicht vorhanden']
    skrt2 = sk[sk['stress rate'] == '2']
    skrt2 = skrt2[skrt2['temperature'] == 'RT']
    skrt2 = skrt2[skrt2['Proben-Kennzeichen'] != 'Proben bei uns nicht vorhanden']
    print(bmrt2)
    print(skrt2)
    print(len(bmrt2)/34)
    print(len(skrt2)/28)
    #print(bm)