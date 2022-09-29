#Author: Phuong Pham 
#Datum: from 11.2021 - 


from os import sep
import os
import re
import  cv2
from cv2 import sort
import numpy as np
from numpy.core.defchararray import index
import pandas as pd
#from PIL import Image, ImageOps
import matplotlib.pyplot as plt
from pandas.core.construction import array
import configparser
import roughness2D

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
from sklearn.feature_selection import SelectFromModel
# Import the necessary libraries first
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
import matplotlib.pyplot as plt 
#from sklearn.feature_selection import SequentialFeatureSelector
#from mlxtend.feature_selection import SequentialFeatureSelector 
import regression


def get_roughness_data(img_path):
    #img_path = r'Klebeverbindungen_Daten\AP5-3D Punktwolken\BM\double_images_0_30_60_90\0\seed0000.png'
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

    h,w = img.shape
    new_img = []

    for i in range(h):
        r = img[i]
        k = np.array(r, dtype=int)-5
        k = k.clip(min =0)
        nzero_count = np.count_nonzero(k)
        if(nzero_count < len(k)*4/5):
            continue
        new_img.append(r)
    new_img = np.transpose(new_img)
    new_img2 = []
    h,w = new_img.shape

    for i in range(h):
        r = new_img[i]
        k = np.array(r, dtype=int)-5
        k = k.clip(min =0)
        nzero_count = np.count_nonzero(k)
        if(nzero_count < len(k)*4/5):
            continue
        new_img2.append(r)
    new_img2 = np.transpose(new_img2)

    #plt.imshow(new_img2)
    #plt.show()

    h,w = new_img2.shape
    
    surface_topo = pd.DataFrame()
   
    topo = pd.DataFrame()
    x =  np.arange(w)
    y = np.zeros(w)
    profil = pd.DataFrame(columns=['x', 'y', 'z', 'zr', 'zw'])
    for i in range(h):
        
        z = new_img2[i]
       
        profil['x'] = x
        profil['y'] = y+i
        profil['z'] = z -np.mean(z)
       
        res = roughness2D.gaussfilter(profil.z, dx = 1)
        profil['zr'] = res[0]
        profil['zw'] = res[1]
       
        r = roughness2D.cal_roughness_params_for_a_profil(profil = profil, dx = 1, y = i)
        w = roughness2D.cal_waveness_for_a_profil(profil = profil, dx = 1)
        rw = pd.concat([r,w], axis = 1)
       
        surface_topo = surface_topo.append(rw)
        profil.iloc[0:0]
    surface_topo = surface_topo.drop(['y'], axis = 1)
    for col_name in surface_topo.columns: 
        m = surface_topo[col_name].mean()
        std = surface_topo[col_name].std()
        cov = std/m
        topo['m_'+col_name]= [m]
        topo['std_'+col_name] = [std]
    return topo


def get_data(directory):
#directory = 'Klebeverbindungen_Daten/AP5-3D Punktwolken/BM/double_images_0_30_60_90/0'
    angles  = [0,30,60]
    train_data = pd.DataFrame()
    targets = []
    for angle in angles: 
        dir = directory + "\\"+  str(angle)
        for filename in os.listdir(dir):
            f = os.path.join(dir , filename)

            if os.path.isfile(f) and '.png' in f:
                print(f)
                train_data = train_data.append(get_roughness_data(f))
                targets.append(angle)
    train_data['targets'] = targets
    return train_data
                #name = filename.split('.')[0]
def feature_importance(x_train, y_train,  feature_names):
  
   
    model = RandomForestClassifier(random_state=0)
    #model =  DecisionTreeClassifier(max_depth=5)
    #X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.3)
    #print(X_test, y_test)
    model.fit(x_train, y_train)
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
 
    # #print(mean_impotance)
    # for feature in feature_names: 
        
    #     if(forest_importances[feature] <= mean_impotance):
    #         data.drop(feature, inplace=True, axis=1)
    
    #return data

    return forest_importances
def features_selection(data,targets):
    feature_names = data.columns
    X = data.values
    y = targets
    feature_selector = SequentialFeatureSelector(RandomForestClassifier(n_jobs=-1),
           k_features='parsimonious',
           forward=True,
           verbose=2,
           scoring='accuracy',
           cv=4)
    features = feature_selector.fit(X, y)
    filtered_features= feature_names[list(features.k_feature_idx_)]
    return data[filtered_features]

def perform_PCA(x_train, y_train):
        
    pca = PCA(n_components=2)
    principalComponent = pca.fit_transform(x_train)
    principalDf = pd.DataFrame(data = principalComponent, index = y_train, columns=["P1", "P2"])

    fig  = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(1,1,1)
    ax.set_xlabel('P1', fontsize = 15)
    ax.set_ylabel('P2', fontsize = 15)
    ax.set_title('Correlation between topo features and stress angle', fontsize = 20)

    colors = ['r', 'g', 'b']
    ta = [0,30,60]
    for target, color in zip(ta, colors):
        ax.scatter(principalDf.loc[target, "P1"], principalDf.loc[target, "P2"], c = color, s = 50)

    #ax.legend(ta)
    ax.grid()
    plt.legend(ta, title = "stress angle")
    plt.show()




#trying the classifiers
def trying_with_some_classifiers(x_train, y_train, x_test, y_test,):
    names = [
        "Nearest Neighbors",
        "Linear SVM",
        #"RBF SVM",
        #"Gaussian Process",
        "Decision Tree",
        "Random Forest",
        #"Neural Net",
        #"AdaBoost",
        #"Naive Bayes",
        #"QDA",
    ]
    classifiers = [

        KNeighborsClassifier(3),
        SVC(kernel="linear", C=0.025),
        #SVC(gamma=2, C=1),
        #GaussianProcessClassifier(1.0 * RBF(1.0)),
        DecisionTreeClassifier(max_depth=5),
        RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
        #MLPClassifier(alpha=1, max_iter=1000),
        #AdaBoostClassifier(),
        #GaussianNB(),
        #QuadraticDiscriminantAnalysis(),
    ]
    #pca = PCA(n_components=2)

    #X_reduced = pca.fit_transform(data)

    classifiers_score = []
    for name, rf in zip(names, classifiers):
        sum_score = 0
        for i in range(100):
            #X_train, X_test, y_train, y_test = train_test_split(data.values, targets)
            rf.fit(x_train, y_train)
            #print(rf.predict_proba(X_test))
            score = rf.score(x_test, y_test)
            #print(score)
            sum_score += score 
            
        classifiers_score.append(sum_score/100)

    classifiers_score = np.asarray(classifiers_score)
    colors = np.where(classifiers_score> 0.5, 'green','red')
   
    print(classifiers_score)
    plt.bar(x = names, height = classifiers_score, width = 0.4, color = colors)
   
    for index,data in enumerate(classifiers_score):
        plt.text(x=index-0.3 , y =data+0.05 , s=f"{data}" , fontdict=dict(fontsize=10))
    plt.title("Results of some classifiers")
    plt.ylim(0,1)
    plt.tight_layout()
    plt.show()

def get_train_test_data(train_directory, test_directory):
    #train_data = get_data(train_directory)
    if os.path.isfile(train_directory + '\\train.csv'):
       train_data = pd.read_csv(train_directory + '\\train.csv', sep = ",")
    #train_data.to_csv(train_directory + '\\train.csv', index=False)
    #test_data = get_data(test_directory)
    if os.path.isfile(test_directory + '\\test.csv'):
       test_data = pd.read_csv(test_directory + '\\test.csv', sep = ",")
    
    test_data.to_csv(test_directory + '\\test.csv', index=False)

    y_train = train_data['targets']
    y_test = test_data['targets']

    # feature_names =  train_data.columns[:-1]
    # print(y_train)
    # print(train_data.values)
    # forest_importances = feature_importance(train_data.iloc[:,:-1].values, y_train , feature_names)
    # mean_impotance = 1/len(feature_names) 
    # for feature in feature_names: 
        
    #     if(forest_importances[feature] <= mean_impotance):
    #         train_data.drop(feature, inplace=True, axis=1)
    #         test_data.drop(feature, inplace=True, axis=1)
    

    
    #targets = np.concatenate([train_targets, test_targets], axis=0)
    train_data = train_data.drop(['targets'], axis = 1)
    test_data = test_data.drop(['targets'], axis = 1)

    x_train = train_data
    x_test = test_data
    return x_train, y_train, x_test, y_test


train_directory = r'Klebeverbindungen_Daten\AP5-3D Punktwolken\BM\double_images_0_30_60_90'
test_directory = r'Klebeverbindungen_Daten\AP5-3D Punktwolken\BM\GrayscaleImagesFromPC - Kopie'

x_train, y_train, x_test, y_test = get_train_test_data(train_directory=train_directory, test_directory=test_directory)
#perform_PCA(x_train, y_train)
#trying_with_some_classifiers(x_train, y_train, x_test, y_test)
#print(x_train, y_train, x_test, y_test)
x_train, x_test = regression.feature_importance(x_train, y_train, x_test, y_test)
regression.trying_with_some_regression(x_train, y_train, x_test, y_test)
