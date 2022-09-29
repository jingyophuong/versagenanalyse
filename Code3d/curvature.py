import pylab
import scipy
from mpl_toolkits.mplot3d import Axes3D
import scipy.ndimage as nd
import numpy as np
import pandas as pd
import cv2
from matplotlib import cm
from mpl_toolkits.axes_grid1 import make_axes_locatable

import matplotlib.pyplot as plt

def surface_curvature(X,Y,Z):

	(lr,lb)=X.shape

	print(lr)
	#print("awfshss-------------")
	print(lb)
#First Derivatives
	Xv,Xu=np.gradient(X)
	Yv,Yu=np.gradient(Y)
	Zv,Zu=np.gradient(Z)
#	print(Xu)

#Second Derivatives
	Xuv,Xuu=np.gradient(Xu)
	Yuv,Yuu=np.gradient(Yu)
	Zuv,Zuu=np.gradient(Zu)   

	Xvv,Xuv=np.gradient(Xv)
	Yvv,Yuv=np.gradient(Yv)
	Zvv,Zuv=np.gradient(Zv) 

#2D to 1D conversion 
#Reshape to 1D vectors
	Xu=np.reshape(Xu,lr*lb)
	Yu=np.reshape(Yu,lr*lb)
	Zu=np.reshape(Zu,lr*lb)
	Xv=np.reshape(Xv,lr*lb)
	Yv=np.reshape(Yv,lr*lb)
	Zv=np.reshape(Zv,lr*lb)
	Xuu=np.reshape(Xuu,lr*lb)
	Yuu=np.reshape(Yuu,lr*lb)
	Zuu=np.reshape(Zuu,lr*lb)
	Xuv=np.reshape(Xuv,lr*lb)
	Yuv=np.reshape(Yuv,lr*lb)
	Zuv=np.reshape(Zuv,lr*lb)
	Xvv=np.reshape(Xvv,lr*lb)
	Yvv=np.reshape(Yvv,lr*lb)
	Zvv=np.reshape(Zvv,lr*lb)

	Xu=np.c_[Xu, Yu, Zu]
	Xv=np.c_[Xv, Yv, Zv]
	Xuu=np.c_[Xuu, Yuu, Zuu]
	Xuv=np.c_[Xuv, Yuv, Zuv]
	Xvv=np.c_[Xvv, Yvv, Zvv]

#% First fundamental Coeffecients of the surface (E,F,G)
	
	E=np.einsum('ij,ij->i', Xu, Xu) 
	F=np.einsum('ij,ij->i', Xu, Xv) 
	G=np.einsum('ij,ij->i', Xv, Xv) 

	m=np.cross(Xu,Xv,axisa=1, axisb=1) 
	p=np.sqrt(np.einsum('ij,ij->i', m, m)) 
	n=m/np.c_[p,p,p]
# n is the normal
#% Second fundamental Coeffecients of the surface (L,M,N), (e,f,g)
	L= np.einsum('ij,ij->i', Xuu, n) #e
	M= np.einsum('ij,ij->i', Xuv, n) #f
	N= np.einsum('ij,ij->i', Xvv, n) #g

# Alternative formula for gaussian curvature in wiki 
# K = det(second fundamental) / det(first fundamental)
#% Gaussian Curvature
	K=(L*N-M**2)/(E*G-F**2)
	K=np.reshape(K,lr*lb)
#	print(K.size)
#wiki trace of (second fundamental)(first fundamental inverse)
#% Mean Curvature
	H = ((E*N + G*L - 2*F*M)/((E*G - F**2)))/2
	print(H.shape)
	H = np.reshape(H,lr*lb)
#	print(H.size)

#% Principle Curvatures
	Pmax = H + np.sqrt(H**2 - K)
	Pmin = H - np.sqrt(H**2 - K)
#[Pmax, Pmin]
	Principle = [Pmax,Pmin]
	return Principle


def fun(x,y):
	return x**2+y**2
#x = scipy.linspace(-1,1,20)
#y = scipy.linspace(-1,1,20)
#[x,y]=scipy.meshgrid(x,y)

#z = (x**3 +y**2 +x*y)
#print(x.shape)
#s = nd.gaussian_filter(z,10)
def cal_curvatures_of_a_surface(path1, pathcsv):
   # path1 = "Klebeverbindungen_Daten\AP5-3D Punktwolken\BM\Final\ProbeR1_1-rel-depthmap.png"
    #pathcsv = "Klebeverbindungen_Daten\AP5-3D Punktwolken\BM\FinalPC\ProbeE1_1-rel-depthmap.png"
    img = cv2.imread(path1, -1)
    h, w = img.shape
    roi_y = int(h*2.5/100)
    roi_x = int(w*1.5/100)
    roi_h = int(h - 2*roi_y)
    roi_w = int(w - 2*roi_x)
    img = img[roi_y:roi_y+roi_h, roi_x : roi_x + roi_w]
    h, w = img.shape

    # data = pd.read_csv(path1, comment = '#', sep=';', names=['x', 'y', 'z'])
    # x = data.x
    # y = data.y
    # z = data.z
    #img = cv2.imread(f, -1)

    img[img == 0] = 32767
    img = img.astype('float')
    z_h = np.zeros((img.shape), dtype = float)
    norscaler = 65535/5000
    z_h= img / norscaler -2500
    s = nd.gaussian_filter(z_h,10)

    data = pd.read_csv(pathcsv, comment = '#', sep=';', names=['x', 'y', 'z'])
    dx = np.diff(data.groupby(['x']).sum().index)[0]
    dy = np.diff(data.groupby(['y']).sum().index)[0]
    print(dx,dy)
    x = scipy.linspace(0, (w-1)*dx, w)
    y = scipy.linspace(0, (h-1)*dy, h)
    [x,y]=scipy.meshgrid(x,y)

    temp1 = surface_curvature(x,y,z_h)
    max_curvatures = temp1[0][temp1[0] != 0]
    min_curvatures = temp1[1][temp1[0] != 0]
    gauss_curvatures = 0.5*(max_curvatures - min_curvatures)
    mean_curvatures = gauss_curvatures**2 - (max_curvatures- gauss_curvatures)**2
    #print(max_curvatures, min_curvatures)
    # print("maximum curvatures")
    # print(temp1[0])
    # print("minimum curvatures")
    # print(temp1[1])
    # print("gauss curvatures")
    # print(gauss_curvatures)
    # #buckelfläche

    # print(np.count_nonzero(np.array(gauss_curvatures * 1000, dtype= int)> 0))
    # #abwickeln
    # print(np.count_nonzero(np.array(gauss_curvatures * 1000, dtype  = int ) == 0))
    # #hyper
    # print(np.count_nonzero(np.array(gauss_curvatures * 1000, dtype  = int ) < 0))

    # print("mean curvatures")
    # print(mean_curvatures)
    fig = pylab.figure()
    ax = Axes3D(fig)
    #print(np.mean(max_curvatures), np.mean(min_curvatures), np.mean(gauss_curvatures), np.mean(mean_curvatures))
    my_col = cm.jet(max_curvatures/np.amax(max_curvatures))
    ax.plot_surface(x,y,z_h)
    pylab.show()
    elliptical_points_count = np.count_nonzero(np.array(gauss_curvatures * 1000, dtype= int)> 0)

    return pd.DataFrame({'max_cur' : [np.mean(max_curvatures)], 
                        'min_cur' : [np.mean(min_curvatures)],
                        'gauss_cur': [np.mean(gauss_curvatures)], 
                        'mean_cur': [np.mean(mean_curvatures)], 
                        'elliptical points': [elliptical_points_count / len(gauss_curvatures)]})



def cal_curvatures_each_point_of_a_surface(path1, pathcsv):
   # path1 = "Klebeverbindungen_Daten\AP5-3D Punktwolken\BM\Final\ProbeR1_1-rel-depthmap.png"
    #pathcsv = "Klebeverbindungen_Daten\AP5-3D Punktwolken\BM\FinalPC\ProbeE1_1-rel-depthmap.png"
    img = cv2.imread(path1, -1)
    h, w = img.shape
    roi_y = int(h*2.5/100)
    roi_x = int(w*2.5/100)
    roi_h = int(h - 2*roi_y)
    roi_w = int(w - 2*roi_x)
    img = img[roi_y:roi_y+roi_h, roi_x : roi_x + roi_w]
    h, w = img.shape

    # data = pd.read_csv(path1, comment = '#', sep=';', names=['x', 'y', 'z'])
    # x = data.x
    # y = data.y
    # z = data.z
    #img = cv2.imread(f, -1)

    #img[img == 0] = 32767
    img = img.astype('float')
    z_h = np.zeros((img.shape), dtype = float)
    norscaler = 65535/5000
    z_h= img / norscaler -2500
    z_h[img==0] = 0
    #s = nd.gaussian_filter(z_h,10)
   
    data = pd.read_csv(pathcsv, comment = '#', sep=';', names=['x', 'y', 'z'])
    dx = np.diff(data.groupby(['x']).sum().index)[0]
    dy = np.diff(data.groupby(['y']).sum().index)[0]
    print('dx,dy=',dx,dy)
    print('h,w =', h,w)
    x = scipy.linspace(0, (w-1)*dx, w)
    y = scipy.linspace(0, (h-1)*dy, h)
    [x,y]=scipy.meshgrid(x,y)
    


    temp1 = surface_curvature(x,y,z_h)

   

    max_curvatures = temp1[0][temp1[0] != 0]
    min_curvatures = temp1[1][temp1[0] != 0]
    g = max_curvatures *min_curvatures
    
    g2 = np.sqrt(np.abs(g))
    g[g>0] = 1
    g[g<0] = -1
   
    gauss_curvatures = g*g2

    mean_curvatures = (max_curvatures+ min_curvatures)/2
    

    # cur = [max_curvatures, min_curvatures, gauss_curvatures, mean_curvatures]
    # names = ['Max curvature', 'Min curvature', 'Gaussian curvature', 'Mean curvature']
    # fig, axs = plt.subplots(2, 2)
    # for col in range(2):
    #     for row in range(2):
    #         ax = axs[row, col]
    #         ax.axes.xaxis.set_visible(False)
    #         ax.axes.yaxis.set_visible(False)
    #         i = 2*col +row 
    #         data = np.reshape(cur[i], (h,w))
    #         pcm = ax.imshow(data)
    #         divider = make_axes_locatable(ax)
    #         cax = divider.append_axes("right", size="5%", pad=0.05)
    #         fig.colorbar(pcm, cax=cax)
    #         ax.set_title(names[i], fontsize=18)
    # plt.show()




    print(np.min(max_curvatures), np.max(max_curvatures))
    print(np.min(min_curvatures), np.max(min_curvatures))
    print(np.min(gauss_curvatures), np.max(gauss_curvatures))
    print(np.min(mean_curvatures), np.max(mean_curvatures))

    



    bins = 800
    min_range = -0.1
    max_range = 0.1
    
    max_min_range = min_range #+ max_range/2
    max_max_range = max_range #+ max_range/2

    min_min_range = min_range #- max_range/2
    min_max_range = max_range #- max_range/2

    max_curvatures[max_curvatures > max_max_range] = max_max_range
    max_curvatures[max_curvatures < max_min_range] = max_min_range
    min_curvatures[min_curvatures > min_max_range] = min_max_range
    min_curvatures[min_curvatures < min_min_range] = min_min_range
    gauss_curvatures[gauss_curvatures > max_range] = max_range
    gauss_curvatures[gauss_curvatures < min_range] = min_range
    mean_curvatures[mean_curvatures > max_range] = max_range
    mean_curvatures[mean_curvatures < min_range] = min_range
    print('h = ', (0.2 * (len(gauss_curvatures)** (1/3)))/ (3.49 * np.std(gauss_curvatures)))
   
    h_max =  np.reshape(cv2.calcHist([np.float32(max_curvatures)], [0], None, [bins], [max_min_range,max_max_range]),(bins)) 
    h_min = np.reshape(cv2.calcHist([np.float32(min_curvatures)], [0], None, [bins], [min_min_range,min_max_range]),(bins))
    h_mean =  np.reshape(cv2.calcHist([np.float32(mean_curvatures)], [0], None, [bins], [min_range,max_range]),(bins)) 
    h_gauss =  np.reshape(cv2.calcHist([np.float32(gauss_curvatures)], [0], None, [bins],[min_range,max_range]),(bins))
    
    
    # h_max =  np.reshape(cv2.calcHist([np.float32(max_curvatures)], [0], None, [bins], [min_range,max_range]),(bins)) / len(max_curvatures)
    # h_min = np.reshape(cv2.calcHist([np.float32(min_curvatures)], [0], None, [bins], [min_range,max_range]),(bins)) / len(max_curvatures)
    # h_mean =  np.reshape(cv2.calcHist([np.float32(mean_curvatures)], [0], None, [bins], [min_range,max_range]),(bins)) / len(max_curvatures)
    # h_gauss =  np.reshape(cv2.calcHist([np.float32(gauss_curvatures)], [0], None, [bins],[min_range,max_range]),(bins)) / len(max_curvatures)
    
    
    #return (max_curvatures, min_curvatures, gauss_curvatures, mean_curvatures)
    return gauss_curvatures
    #return min_curvatures
    #return  h_max, h_min, h_gauss, h_mean,len(max_curvatures)
    #return np.hstack((h_max , h_min, h_gauss, h_mean))   
    #print(h_max)


if __name__== '__main__1':
    path1 = "Klebeverbindungen_Daten\AP5-3D Punktwolken\BM\Final\ProbeE_1-rel-depthmap.png"
    pathcsv = "Klebeverbindungen_Daten\AP5-3D Punktwolken\BM\FinalPC\ProbeE_1.txt"

    c  = cal_curvatures_each_point_of_a_surface(path1, pathcsv)
  
    #image = np.reshape(c)

if __name__ == '__main__':
    path1 = "Klebeverbindungen_Daten\AP5-3D Punktwolken\BM\Final\ProbeE1_1-rel-depthmap.png"
    pathcsv = "Klebeverbindungen_Daten\AP5-3D Punktwolken\BM\FinalPC\ProbeE1_1.txt"

    c  = cal_curvatures_each_point_of_a_surface(path1, pathcsv)
    
    bins = 800
    countsc, binsmax = np.histogram(c, bins = bins, range=[-0.1, 0.1])
    
    fig, (ax0, ax1, ax2) = plt.subplots(nrows=1, ncols=3, sharex= True,
                                    figsize=(12, 6))
    ax0.hist(binsmax[:-1],bins, weights=countsc, orientation = 'vertical')
    ax0.set_title('Surface 1 (absolut)',)
    #plt.show()
    path2 = "Klebeverbindungen_Daten\AP5-3D Punktwolken\BM\Final\ProbeS17_2-rel-depthmap.png"
    pathcsv2 = "Klebeverbindungen_Daten\AP5-3D Punktwolken\BM\FinalPC\ProbeS17_2.txt"

    k  = cal_curvatures_each_point_of_a_surface(path2, pathcsv2)
    
    countsk, binsmax = np.histogram(k, bins = bins, range=[-0.1, 0.1])
   
    ax1.hist(binsmax[:-1],bins, weights=countsk ,  orientation = 'vertical')
    ax1.set_title('Surface 2 (absolut)')
    #.show()

      
    ax2.hist(binsmax[:-1],bins, weights=(countsc+countsk) /(len(c)+ len(k)), orientation = 'vertical') 
    ax2.set_title('Sample Result (relativ)')
    #plt.title ('Gaussian curvature')
    plt.show()

    