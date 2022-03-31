#from Phuong Pham - 01.2022 
#from torch import scatter
from pyexpat import features
from this import d
from trace import Trace
import roughness3D
import  roughness2D
import os
import pandas as pd
import numpy as np
import pandas as pd

from sklearn.decomposition import PCA
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

import matplotlib.pyplot as plt 

#Feature Importance
# load the iris datasets
#dataset = datasets.load_iris()
# fit an Extra Trees model to the data
def feature_importance(data, targets):
    #feature_names=["Angular Second Moment", "Contrast", "Correlation", "Sum of Squares: Variance", "Inverse Difference Moment", "Sum Average", 
    #                        "Sum Variance", "Sum Entropy", "Entropy", "Difference Variance", "Difference Entropy", "Info. Measure of Correlation 1", "Info. Measure of Correlation 2"]
    feature_names = data.columns
    X = data.values
    y = targets
    n = np.zeros(len(feature_names))
    for i  in range(100):
        X_train, X_test, y_train, y_test = train_test_split(X, y)
        model = RandomForestClassifier(random_state=0)
        model.fit(X_train, y_train)
        # display the relative importance of each attribute
        n += model.feature_importances_

    impotances = n/100

    forest_importances = pd.Series(impotances, index=feature_names)
    #print(forest_importances)
    std = np.std([tree.feature_importances_ for tree in model.estimators_], axis=0)
    mean_impotance = 1/len(feature_names)
    
    fig, ax = plt.subplots()
    moreImpotance = (forest_importances > mean_impotance).astype(int).values
    colors = np.where(moreImpotance, 'green','red')
    #print(moreImp)
    forest_importances.plot.bar(yerr = std, ax = ax, color = colors)
    #plt.axhline(y=mean_impotance, color='r', linestyle='-')
    ax.set_title("Feature importances using MDI")
    ax.set_ylabel("Mean decrease in impurity")
    fig.tight_layout()
    plt.show()
    #calculate mean of impotances of all features and drop features with less impotant than mean_impotances
     
   
    #print(mean_impotance)
    for feature in feature_names: 
        
        if(forest_importances[feature] <= mean_impotance):
            data.drop(feature, inplace=True, axis=1)
    
    return data

def perform_PCA(data, targets):
        
    pca = PCA(n_components=2)
    principalComponent = pca.fit_transform(data.values)
    principalDf = pd.DataFrame(data = principalComponent, index = targets, columns=["P1", "P2"])

    fig  = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(1,1,1)
    ax.set_xlabel('P1', fontsize = 15)
    ax.set_ylabel('P2', fontsize = 15)
    ax.set_title('Correlation between texture features and stress angle', fontsize = 20)

    colors = ['r', 'g', 'b', 'c']
    ta = [0,60,90]
    for target, color in zip(ta, colors):
        ax.scatter(principalDf.loc[target, "P1"], principalDf.loc[target, "P2"], c = color, s = 50)

    #ax.legend(ta)
    ax.grid()
    plt.legend(ta, title = "stress angle")
    plt.show()




#trying the classifiers
def trying_with_some_classifiers(data, targets):
    names = [
        "Nearest Neighbors",
        "Linear SVM",
        "RBF SVM",
        "Gaussian Process",
        "Decision Tree",
        "Random Forest",
        "Neural Net",
        "AdaBoost",
        "Naive Bayes",
        "QDA",
    ]
    classifiers = [
        KNeighborsClassifier(3),
        SVC(kernel="linear", C=0.025),
        SVC(gamma=2, C=1),
        GaussianProcessClassifier(1.0 * RBF(1.0)),
        DecisionTreeClassifier(max_depth=5),
        RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
        MLPClassifier(alpha=1, max_iter=1000),
        AdaBoostClassifier(),
        GaussianNB(),
        QuadraticDiscriminantAnalysis(),
    ]
    pca = PCA(n_components=2)

    X_reduced = pca.fit_transform(data)

    for name, rf in zip(names, classifiers):
        sum_score = 0
        for i in range(100):
            X_train, X_test, y_train, y_test = train_test_split(X_reduced, targets)
            rf.fit(X_train, y_train)
            #print(rf.predict_proba(X_test))
            score = rf.score(X_test, y_test)
            #print(score)
            sum_score += score 

        print('mean score of 100 test runs = ' + str(sum_score/100)+ 'with ' + name)


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
                
                #r = roughness2D.
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

def classify_with_roughness2d(cal = False, save = False, save_file = "roughness2d_y"):
    # #1.Step: loading the general information of all samples

    samples = pd.read_csv("Klebeverbindungen_Daten/Proben.csv" ,sep=';', error_bad_lines=False)
    samples = samples.set_index("ID")
    targets = []
    #feature_names = ['Sa', 'Sq', 'Sz', 'Ssk', 'Ssu']
    data = pd.DataFrame()
    if not cal:
        if os.path.isfile('Klebeverbindungen_Daten/roughness2d.csv'):
            sdata = pd.read_csv('Klebeverbindungen_Daten/roughness2d.csv', sep = ",")
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
            path1 = "Klebeverbindungen_Daten/3d/RelativData/Probe" + index + "_1.txt"
            path2 = "Klebeverbindungen_Daten/3d/RelativData/Probe" + index + "_2.txt"
            if(os.path.isfile(path1) and os.path.isfile(path2)):
                roughness = roughness2D.compute_topography_of_a_probe(path1, path2)
        

                #print(index, 'has given results:', roughness)
    
                data = data.append(roughness, ignore_index=True)
                targets.append(row['stress angle'])
   
   
    data = feature_importance(data, targets)
   
    perform_PCA(data, targets)
 
    if save:
        save_data = data
        save_data['targets'] = targets
        save_data.to_csv('Klebeverbindungen_Daten/' + save_file + '.txt', index=False)

def classify_with_roughness2d_xy(data_x_path, data_y_path):
    data = pd.DataFrame()
    targets = []
    if os.path.isfile(data_x_path):
        data_x = pd.read_csv(data_x_path, sep = ",", header=None, skiprows = [0])
        data_x = data_x.iloc[: , :-1]
        
    if os.path.isfile(data_y_path):
        data_y = pd.read_csv(data_y_path, sep = ",")
        targets = data_y['targets']
        data_y = data_y.iloc[: , :-1]
        data = pd.concat([data_x, data_y], axis = 1)
        
        print(data, targets)
    if(data.empty or len(targets) == 0):
        return
    data = feature_importance(data, targets)
   
    perform_PCA(data, targets)
##############################################################MAIN#################################################################################
##############################################################MAIN#################################################################################
##############################################################MAIN#################################################################################
##############################################################MAIN#################################################################################
##############################################################MAIN#################################################################################
if __name__ == "__main__":
    #classify_with_roughness3d()
    #classify_with_roughness2d(True, True, save_file = "roughness2d_y")
    classify_with_roughness2d_xy('Klebeverbindungen_Daten/roughness2d.csv', 'Klebeverbindungen_Daten/roughness2d_y.txt')
    