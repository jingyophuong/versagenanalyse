#from Phuong Pham - 01.2022 
#from torch import scatter

from pyrsistent import v
#from . import roughness3D
#from . import roughness2D
import roughness2D
import  roughness3D
import os
import pandas as pd
import numpy as np
import pandas as pd


from sklearn.decomposition import PCA
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsRegressor
#from sklearn.svm import SVC
#from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor


#trying the classifiers
def trying_with_some_regression(data_train, targets_train, data_test, targets_test):
    names = [
        "Nearest Neighbors",
       
       
        "Decision Tree",
        "Random Forest",
      
    ]
    regressors = [

        #KNeighborsRegressor(3),
        #SVC(kernel="linear", C=0.025),
        #SVC(gamma=2, C=1),
        #GaussianProcessClassifier(1.0 * RBF(1.0)),
        #DecisionTreeRegressor(max_depth=5),
        RandomForestRegressor(max_depth=5, n_estimators=10, max_features=1),
        #MLPClassifier(alpha=1, max_iter=1000),
        #AdaBoostClassifier(),
        #GaussianNB(),
        #QuadraticDiscriminantAnalysis(),
    ]
    #pca = PCA(n_components=2)

    #X_reduced = pca.fit_transform(data)

    regressions_score = []
    for name, rf in zip(names, regressors):
        print(name)
        sum_score = 0
        for i in range(1):
            #X_train, X_test, y_train, y_test = train_test_split(data.values, targets)
            rf.fit(data_train.values, targets_train)
            #print(rf.predict_proba(X_test))
            #score = rf.score(data_test.values, targets_test)
            #print(score)
            #sum_score += score 
            for test, test_label in zip(data_test.values, targets_test):
                t = rf.predict([test])
                print(abs(t[0] - test_label) < 5)
        regressions_score.append(sum_score/1)

    regressions_score = np.asarray(regressions_score)
    colors = np.where(regressions_score> 0.5, 'green','red')
   
    print(regressions_score)
    plt.bar(x = names, height = regressions_score, width = 0.4, color = colors)
   
    for index,data in enumerate(regressions_score#
    
    
    ):
        plt.text(x=index-0.3 , y =data+0.05 , s=f"{data}" , fontdict=dict(fontsize=10))
    plt.title("Results of some classifiers")
    plt.ylim(0,1)
    plt.tight_layout()
    plt.show()
    

def feature_importance(data, targets, test_x, test_y):
    feature_names = data.columns
    X = data.values
    
    y = targets
    n = np.zeros(len(feature_names))
   
    model = RandomForestRegressor(random_state=0)
    #model =  DecisionTreeClassifier(max_depth=5)
    #X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.3)
    #print(X_test, y_test)
    model.fit(X, y)
        # display the relative importance of each attribute
    impotances =  model.feature_importances_

   
    forest_importances = pd.Series(impotances, index=feature_names)
    #std = np.std([tree.feature_importances_ for tree in model.estimators_], axis=0)
    mean_impotance = 1/len(feature_names) 
   
    fig, ax = plt.subplots()
    moreImpotance = (forest_importances > mean_impotance).astype(int).values
    colors = np.where(moreImpotance, 'green','red')
    forest_importances.plot.bar(ax = ax, color = colors)
    ax.set_title("Feature importances using MDI")
    ax.set_ylabel("Mean decrease in impurity")
    fig.tight_layout()
    plt.show()
 
    #print(mean_impotance)
    for feature in feature_names: 
        
        if(forest_importances[feature] <= mean_impotance):
            data.drop(feature, inplace=True, axis=1)
            test_x.drop(feature, inplace=True, axis=1)
    
    return [data, test_x]

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
 


def classify_with_roughness3d(save_file ='roughness3d_y.txt', test_angle = 30):
    # #1.Step: loading the general information of all samples

    samples = pd.read_csv("Klebeverbindungen_Daten/Proben.csv" ,sep=';', error_bad_lines=False)
    samples = samples.set_index("ID")
    targets = []
    feature_names = ['Sa', 'Sq', 'Sz', 'Ssk', 'Ssu']
    data = pd.DataFrame()
    test = pd.DataFrame()
    if os.path.isfile('Klebeverbindungen_Daten/' + save_file):
        data = pd.read_csv('Klebeverbindungen_Daten/' + save_file, sep = ",")
        
        test = test.append(data[data.targets == test_angle], ignore_index=True)
        data = data.drop(data[data.targets == test_angle].index)

        data_targets = data.targets.values
        test_targets = test.targets.values
        data = data.drop(['targets'], axis = 1)
        test = test.drop(['targets'], axis = 1)
        #print(data)
    
    #data = features_selection(data,targets)
    #print(data,test, data_targets, test_targets)
    data, test = feature_importance(data, data_targets, test, test_targets)
    
    #perform_PCA(data, targets)

    trying_with_some_regression(data, data_targets, test, test_targets)
   
def classify_surface_with_roughness3d(data_dir, save = True, cal = False):
    # #1.Step: loading the general information of all samples

    samples = pd.read_csv("Klebeverbindungen_Daten/Proben.csv" ,sep=';', error_bad_lines=False)
    samples = samples.set_index("ID")
    targets = []
    feature_names = ['Sa', 'Sq', 'Sz', 'Ssk', 'Ssu']
    data = pd.DataFrame()
    if os.path.isfile('Klebeverbindungen_Daten/roughness3d.csv') and not cal:
        data = pd.read_csv('Klebeverbindungen_Daten/roughness3d.csv', sep = ",")
        targets = data.targets
        data = data.drop('targets', axis = 1)
        #print(data)
    else:
  
        for index, row in samples.iterrows():
            path1 = data_dir +"Probe" + index + "_1-relH.txt"
            path2 = data_dir+"Probe" + index + "_2-relH.txt"
            
            
            if(os.path.isfile(path1) and os.path.isfile(path2)):
                for path in [path1, path2]:
                    roughness_surface = roughness3D.cal_Roughness_params(0,path)
                    #roughness_probe =roughness_probe.append(roughness3D.cal_Roughness_params(0,path2), ignore_index = True)
                    # surface_topo = pd.DataFrame()
                    # for col_name in roughness_probe.columns: 
                    #     m = roughness_probe[col_name].mean()
                    #     std = roughness_probe[col_name].std()
                    #     #cov = std/m
                    #     surface_topo['m_'+col_name]= [m]
                    #     surface_topo['std_'+col_name] = [std]
                    data = data.append(roughness_surface,ignore_index= True)
                    targets.append(row['stress angle'])

    if save:  
        data['targets'] = targets  
        data.to_csv('Klebeverbindungen_Daten/eachsurfaceClassify3D.csv', index=False)
        data = data.drop(['targets'], axis = 1)
    print(data)
    data = feature_importance(data, targets)
    
    perform_PCA(data, targets)

    trying_with_some_classifiers(data, targets)
 
##############################################################CLASSIFICATION-WITH-ROUGHNESS2D#################################################################################
##############################################################CLASSIFICATION-WITH-ROUGHNESS2D#################################################################################
##############################################################CLASSIFICATION-WITH-ROUGHNESS2D#################################################################################
##############################################################CLASSIFICATION-WITH-ROUGHNESS2D#################################################################################
##############################################################CLASSIFICATION-WITH-ROUGHNESS2D#################################################################################
##############################################################CLASSIFICATION-WITH-ROUGHNESS2D#################################################################################

def classify_surface_with_roughness2d(dir , cal = False, save = False, save_file = "surface_roughness2d_y"):
    # #1.Step: loading the general information of all samples

    samples = pd.read_csv("Klebeverbindungen_Daten/Proben.csv" ,sep=';', error_bad_lines=False)
    samples = samples.set_index("ID")
    targets = []
    #feature_names = ['Sa', 'Sq', 'Sz', 'Ssk', 'Ssu']
    data = pd.DataFrame()
    if not cal:
        if os.path.isfile('Klebeverbindungen_Daten/' + save_file + '.txt'):
            sdata = pd.read_csv('Klebeverbindungen_Daten/' + save_file + '.txt', sep = ",")
            #print('sdata = ', sdata)
            targets = sdata['targets']
            data = sdata.drop(['targets'], axis=1)
    if cal: 
         for index, row in samples.iterrows():
            path1 = dir + "Probe" + index + "_1-rel.txt"
            path2 = dir + "Probe" + index + "_2-rel.txt"
            if(os.path.isfile(path1) and os.path.isfile(path2)):
                for path in [path1, path2]:
                    t = roughness2D.compute_topography_of_surface_h(path)
                    surface_topo = pd.DataFrame()
                    for col_name in t.columns: 
                        m = t[col_name].mean()
                        std = t[col_name].std()
                        cov = std/m
                        surface_topo['m_'+col_name]= [m]
                        surface_topo['std_'+col_name] = [std]
                    #print(surface_topo)
                    data = data.append(surface_topo, ignore_index=True)
                    targets.append(row['stress angle'])
    if save:
        data['targets'] = targets
        data.to_csv('Klebeverbindungen_Daten/' + save_file + '.txt', index=False)
        data = data.drop(['targets'], axis = 1)
    
    print(data)
    data = feature_importance(data, targets)
   
    perform_PCA(data, targets)
    trying_with_some_classifiers(data, targets)

def classify_with_roughness2d(dir , cal = False, save = False, y = True, save_file = "roughness2d_y"):
    # #1.Step: loading the general information of all samples

    samples = pd.read_csv("Klebeverbindungen_Daten/Proben.csv" ,sep=';', error_bad_lines=False)
    samples = samples.set_index("ID")
    targets = []
    #feature_names = ['Sa', 'Sq', 'Sz', 'Ssk', 'Ssu']
    data = pd.DataFrame()
    if not cal:
        if os.path.isfile('Klebeverbindungen_Daten/' + save_file + '.txt'):
            sdata = pd.read_csv('Klebeverbindungen_Daten/' + save_file + '.txt', sep = ",")
            #print('sdata = ', sdata)
            targets = sdata['targets']
            data = sdata.drop(['targets'], axis=1)
    if cal: 
         for index, row in samples.iterrows():
            # path1 = "Klebeverbindungen_Daten/3d/RelativDataH/Probe" + index + "_1H.txt"
            # path2 = "Klebeverbindungen_Daten/3d/RelativDataH/Probe" + index + "_2H.txt"
            # if(os.path.isfile(path1) and os.path.isfile(path2)):
            
            #     roughness = roughness2D.compute_topography_of_a_probe_h(path1, path2)
            # # #   
            #
            #   
            if y:   
                path1 = dir + "Probe" + index + "_1-rel.txt"
                path2 = dir + "Probe" + index + "_2-rel.txt"
                if(os.path.isfile(path1) and os.path.isfile(path2)):
              
                    roughness = roughness2D.compute_topography_of_a_probe_h(path1, path2)
                #print(roughness)
                #print(index, 'has given results:', roughness)
        
                    data = data.append(roughness, ignore_index=True)
                    targets.append(row['stress angle'])

            else: 
                path1 = dir + "Probe" + index + "_1-relH.txt"
                path2 = dir + "Probe" + index + "_2-relH.txt"
                if(os.path.isfile(path1) and os.path.isfile(path2)):
              
                    roughness = roughness2D.compute_topography_of_a_probe_h(path1, path2)
                #print(roughness)
                #print(index, 'has given results:', roughness)
        
                    data = data.append(roughness, ignore_index=True)
                    targets.append(row['stress angle'])

    if save:
        data['targets'] = targets
        data.to_csv('Klebeverbindungen_Daten/' + save_file + '.txt', index=False)
        data = data.drop(['targets'], axis = 1)
    print(data)
    data = feature_importance(data, targets)
    #data = features_selection(data,targets)
    perform_PCA(data, targets)
    trying_with_some_classifiers(data, targets)
 
def classify_with_roughness2d_hv(data_h_path, data_v_path):
    data = pd.DataFrame()
    targets = []
    if os.path.isfile(data_h_path):
        data_x = pd.read_csv(data_h_path, sep = ",", header=None, skiprows = [0])
        data_x = data_x.iloc[: , :-1]
        
    if os.path.isfile(data_v_path):
        data_y = pd.read_csv(data_v_path, sep = ",")
        targets = data_y['targets']
        data_y = data_y.iloc[: , :-1]
        data = pd.concat([data_x, data_y], axis = 1)
        
        print(data, targets)
    if(data.empty or len(targets) == 0):
        return
    data = feature_importance(data, targets)
    #data = features_selection(data,targets)
    perform_PCA(data, targets)
    trying_with_some_classifiers(data,targets)

##############################################################CLASSIFICATION-WITH-ROUGHNESS2D AND 3D#################################################################################
##############################################################CLASSIFICATION-WITH-ROUGHNESS2D AND 3D#################################################################################
##############################################################CLASSIFICATION-WITH-ROUGHNESS2D AND 3D#################################################################################
##############################################################CLASSIFICATION-WITH-ROUGHNESS2D AND 3D#################################################################################
##############################################################CLASSIFICATION-WITH-ROUGHNESS2D AND 3D#################################################################################
##############################################################CLASSIFICATION-WITH-ROUGHNESS2D AND 3D#################################################################################
def classify_with_combi_2dand3d():
    #2d
    if os.path.isfile('Klebeverbindungen_Daten/roughness2d_h.txt'):
        sdata = pd.read_csv('Klebeverbindungen_Daten/roughness2d_h.txt', sep = ",")
          
        targets = sdata['targets']
        data2d = sdata.drop(['targets'], axis=1)
    if os.path.isfile('Klebeverbindungen_Daten/roughness3d.csv'):
        sdata = pd.read_csv('Klebeverbindungen_Daten/roughness3d.csv', sep = ",")
        data3d = sdata.drop(['targets'], axis=1)
    if data2d.empty or data3d.empty:
        return
    data = pd.concat([data2d, data3d], axis = 1)
    data = feature_importance(data, targets)
   
    perform_PCA(data, targets)
    trying_with_some_classifiers(data,targets)
    
def classify_with_combi_textur_and_topo(images_path, topo_path, save = True, cal = False, save_file = 'tt_data.txt'):
    samples = pd.read_csv("Klebeverbindungen_Daten/Proben.csv" ,sep=';', error_bad_lines=False)
    targets =   []
    samples = samples.set_index('ID')
    data = []
    if not cal:
        if os.path.isfile('Klebeverbindungen_Daten/' + save_file + '.txt'):
            sdata = pd.read_csv('Klebeverbindungen_Daten/' + save_file + '.txt', sep = ",")
            #print('sdata = ', sdata)
            targets = sdata['targets']
            data = sdata.drop(['targets'], axis=1)
    else: 
        for index, row in samples.iterrows():
            full = True
            #texture
            path1 = images_path + 'Probe' + index + "_1.png"
            path2 =images_path + 'Probe'+ index + "_2.png"
            
            if(os.path.isfile(path1) and os.path.isfile(path2)):
                
                f1 =texturefeaturesExtract.extract_haralick_features(0,path1, round_object='R' in index, underground=255)
                f2 = texturefeaturesExtract.extract_haralick_features(0, path2, round_object= 'R' in index, underground= 255)
                f = np.row_stack((f1, f2))
                f = np.mean(f, axis = 0).T
            else: 
                full = False
            #topo
            path1 = topo_path + "Probe" + index + "_1-rel.txt"
            path2 = topo_path + "Probe" + index + "_2-rel.txt"
            if(os.path.isfile(path1) and os.path.isfile(path2)):
                roughness = roughness2D.compute_topography_of_a_probe_h(path1, path2)
            else: 
                full = False
            if full:
                f = pd.concat([f, roughness], axis = 1)
                data = data.append(f)
                targets.append(row['stress angle'])
    if save:
        data['targets'] = targets
        data.to_csv('Klebeverbindungen_Daten/' + save_file + '.txt', index=False)
        data = data.drop(['targets'], axis = 1)
    
    data = feature_importance(data, targets)
   
    perform_PCA(data, targets)
    trying_with_some_classifiers(data, targets)

    
##############################################################MAIN#################################################################################
##############################################################MAIN#################################################################################
##############################################################MAIN#################################################################################
##############################################################MAIN#################################################################################
##############################################################MAIN#################################################################################
if __name__ == "__main__":
    classify_with_roughness3d( save_file = 'roughness3d_y.txt')
    #classify_with_roughness2d('Klebeverbindungen_Daten/AP5-3D Punktwolken/BM/NoiseReduced/' , cal = False, save = False, save_file = "roughness2d_v")
    #classify_with_roughness2d('Klebeverbindungen_Daten/AP5-3D Punktwolken/BM/RelativDataH/' , cal = False, save = False,y=False, save_file = "roughness2d_h")

    #classify_with_roughness2d_hv('Klebeverbindungen_Daten/roughness2d_h.txt', 'Klebeverbindungen_Daten/roughness2d_v.txt')
    #classify_with_combi_2dand3d()

    imagepath = r'Klebeverbindungen_Daten/2D-MakroImages/new_bm2/'
    topopath = r'Klebeverbindungen_Daten/AP5-3D Punktwolken/BM/NoiseReduced/'
    #classify_with_combi_textur_and_topo(imagepath, topopath, save = True, cal = True)

    #classify_surface_with_roughness3d(data_dir= r"Klebeverbindungen_Daten/AP5-3D Punktwolken/BM/NoiseReduced/", cal = True, save = True)

    #classify_surface_with_roughness2d('Klebeverbindungen_Daten/AP5-3D Punktwolken/BM/NoiseReduced/' , cal = True, save = True)

    #classify_with_roughness3d(data_dir= r"Klebeverbindungen_Daten/AP5-3D Punktwolken/SK/NoiseReduced/", cal = True, save = True, save_file = 'sk_roughness3d_y.txt')
    #classify_with_roughness2d('Klebeverbindungen_Daten/AP5-3D Punktwolken/SK/NoiseReduced/' , cal = True, save = True, save_file = "sk_roughness2d_y")
    #classify_with_roughness2d_xy('Klebeverbindungen_Daten/roughness2d_x.txt', 'Klebeverbindungen_Daten/roughness2d_y.txt')
    #classify_with_roughness2d('Klebeverbindungen_Daten/AP5-3D Punktwolken/SK/HorizontalProfiles/' , cal = True, save = True,y=False, save_file = "sk_roughness2d_x")

    #classify_with_combi_2dand3d()