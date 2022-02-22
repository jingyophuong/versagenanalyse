#from Phuong Pham - 01.2022 
from torch import scatter
import roughness3D
import  roughness2D
import os
import pandas as pd
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt 


##############################################################CLASSIFICATION-WITH-ROUGHNESS3D#################################################################################
##############################################################CLASSIFICATION-WITH-ROUGHNESS3D#################################################################################
##############################################################CLASSIFICATION-WITH-ROUGHNESS3D#################################################################################
##############################################################CLASSIFICATION-WITH-ROUGHNESS3D#################################################################################
##############################################################CLASSIFICATION-WITH-ROUGHNESS3D#################################################################################
def show_data(data, indexes):
    fig = plt.figure()
    data_nr = len(data.columns)
    axsNr = int(data_nr /2 +1)
    fig,axs = plt.subplots(2, axsNr, figsize = (20,15))

    i = 0
    indexes = np.array(indexes) 
    indexes = indexes.astype(int)
    #colormap = np.array(['r', 'g', 'c','b'])
    for col in data.columns:
        x = int(i/axsNr)
        y = int(i % axsNr)
        scatter = axs[x,y].scatter(np.arange(len(data)), data[col] , c = indexes)
        axs[x,y].legend(*scatter.legend_elements(), title = 'classes')
        i = i+1
        
        axs[x,y].set_title(col)
    plt.show()
 


def classify_with_roughness3d():
    # #1.Step: loading the general information of all samples

    samples = pd.read_csv("Klebeverbindungen_Daten/Proben.csv" ,sep=';', error_bad_lines=False)
    samples = samples.set_index("ID")
    targets = []
    feature_names = ['Sa', 'Sq', 'Sz', 'Ssk', 'Ssu']
    data = pd.DataFrame()
    #if os.path.isfile('Klebeverbindungen_Daten/roughness3d.csv'):
    #    data = pd.read_csv('Klebeverbindungen_Daten/roughness3d.csv', sep = ",")


    if(not(os.path.isfile('Klebeverbindungen_Daten/roughness3d.csv')) or data.empty ):
    
        for index, row in samples.iterrows():
            path1 = "Klebeverbindungen_Daten/3d/RelativData/Probe" + index + "_1.txt"
            path2 = "Klebeverbindungen_Daten/3d/RelativData/Probe" + index + "_2.txt"
            if(os.path.isfile(path1) and os.path.isfile(path2)):
                roughness_probe = roughness3D.cal_Roughness_params(path1)
                roughness_probe =roughness_probe.append(roughness3D.cal_Roughness_params(path2), ignore_index = True)
                #roughness_std = roughness_probe.std()
                roughness_mean = roughness_probe.mean().rename(str.lower, axis='columns')
                #topo = pd.concat([roughness_std, roughness_mean])

            #HF_features = np.append(HF_features, row['stress angle'])
                data = data.append(roughness_mean, ignore_index=True)
                targets.append(row['stress angle'])
        
        data.to_csv('Klebeverbindungen_Daten/roughness3d.csv', index=False)

    X = data.values
    print(data)
    show_data(data, targets)
    #print(X)
    pca = PCA(n_components=2)
    principalComponent = pca.fit_transform(X)
    principalDf = pd.DataFrame(data = principalComponent, index = targets, columns=["P1", "P2"])

    #print(principalDf)

    fig  = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(1,1,1)
    ax.set_xlabel('P1', fontsize = 15)
    ax.set_ylabel('P2', fontsize = 15)
    ax.set_title('Correlation between texture features and stress angle', fontsize = 20)

    colors = ['r','b', 'c']
    ta = [0,60,90]
    for target, color in zip(ta, colors):
        ax.scatter(principalDf.loc[target, "P1"], principalDf.loc[target, "P2"], c = color, s = 50)

    #ax.legend(ta)
    ax.grid()
    plt.legend(ta, title = "stress angle")
    plt.show()


##############################################################CLASSIFICATION-WITH-ROUGHNESS2D#################################################################################
##############################################################CLASSIFICATION-WITH-ROUGHNESS2D#################################################################################
##############################################################CLASSIFICATION-WITH-ROUGHNESS2D#################################################################################
##############################################################CLASSIFICATION-WITH-ROUGHNESS2D#################################################################################
##############################################################CLASSIFICATION-WITH-ROUGHNESS2D#################################################################################
##############################################################CLASSIFICATION-WITH-ROUGHNESS2D#################################################################################

def classify_with_roughness2d():
    # #1.Step: loading the general information of all samples

    samples = pd.read_csv("Klebeverbindungen_Daten/Proben.csv" ,sep=';', error_bad_lines=False)
    samples = samples.set_index("ID")
    targets = []
    #feature_names = ['Sa', 'Sq', 'Sz', 'Ssk', 'Ssu']
    data = pd.DataFrame()
    #if os.path.isfile('Klebeverbindungen_Daten/roughness3d.csv'):
    #    data = pd.read_csv('Klebeverbindungen_Daten/roughness3d.csv', sep = ",")


    if(not(os.path.isfile('Klebeverbindungen_Daten/roughness3d.csv')) or data.empty ):
    
        for index, row in samples.iterrows():
            path1 = "Klebeverbindungen_Daten/3d/RelativData/Probe" + index + "_1.txt"
            path2 = "Klebeverbindungen_Daten/3d/RelativData/Probe" + index + "_2.txt"
            if(os.path.isfile(path1) and os.path.isfile(path2)):
                roughness_probe = roughness2D.compute_topography_of_surface(path1)
                roughness_probe = roughness_probe.append(roughness2D.compute_topography_of_surface(path2), ignore_index = True)
                roughness = roughness_probe.mean()
            #HF_features = np.append(HF_features, row['stress angle'])
                data = data.append(roughness, ignore_index=True)
                targets.append(row['stress angle'])
        
        data.to_csv('Klebeverbindungen_Daten/roughness2d.csv', index=False)

    print(data)
    X = data.values

    #print(X)
    pca = PCA(n_components=2)
    principalComponent = pca.fit_transform(X)
    principalDf = pd.DataFrame(data = principalComponent, index = targets, columns=["P1", "P2"])

    #print(principalDf)

    fig  = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(1,1,1)
    ax.set_xlabel('P1', fontsize = 15)
    ax.set_ylabel('P2', fontsize = 15)
    ax.set_title('Correlation between topography features and stress angle', fontsize = 20)

    colors = ['r', 'b', 'c']
    ta = [0,60,90]
    for target, color in zip(ta, colors):
        ax.scatter(principalDf.loc[target, "P1"], principalDf.loc[target, "P2"], c = color, s = 50)

    #ax.legend(ta)
    ax.grid()
    plt.legend(ta, title = "stress angle")
    plt.show()


##############################################################MAIN#################################################################################
##############################################################MAIN#################################################################################
##############################################################MAIN#################################################################################
##############################################################MAIN#################################################################################
##############################################################MAIN#################################################################################
classify_with_roughness3d()


