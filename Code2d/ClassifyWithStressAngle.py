#from Phuong T. Pham - Begin: 29.10.2021
from cv2 import fastAtan2
import texturefeaturesExtract
import os
import pandas as pd
import numpy as np
import cv2
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
from sklearn.feature_selection import RFECV

columns_name=["Angular Second Moment", "Contrast", "Correlation", "Sum of Squares: Variance", "Inverse Difference Moment", "Sum Average", 
                           "Sum Variance", "Sum Entropy", "Entropy", "Difference Variance", "Difference Entropy", "Info. Measure of Correlation 1", "Info. Measure of Correlation 2"]
columns_name_std=["Angular Second Moment (Std)", "Contrast (Std)", "Correlation (Std)", "Sum of Squares: Variance (Std)", "Inverse Difference Moment (Std)", "Sum Average (Std)", 
                           "Sum Variance (Std)", "Sum Entropy (Std)", "Entropy (Std)", "Difference Variance (Std)", "Difference Entropy (Std)", "Info. Measure of Correlation 1 (Std)", "Info. Measure of Correlation 2 (Std)"]

def load_features(images_path = "",  extract = False, save = False, separat = False, save_name = ""):
    if(extract):  
        samples = pd.read_csv("Klebeverbindungen_Daten/Proben.csv" ,sep=';', error_bad_lines=False)
        targets =   []
        samples = samples.set_index('ID')
        data = []
        for index, row in samples.iterrows():
            path1 = images_path + 'Probe' + index + "_1.png"
            path2 =images_path + 'Probe'+ index + "_2.png"
           
            if(os.path.isfile(path1) and os.path.isfile(path2)):# and os.path.isfile(path_3d1) and os.path.isfile(path_3d2)):
               
                f1 =texturefeaturesExtract.extract_haralick_features(0,path1, round_object='R' in index, underground=255)

                f2 = texturefeaturesExtract.extract_haralick_features(0, path2, round_object= 'R' in index, underground= 255)
                if separat: 
                    data.append(f1)
                    targets.append(row['stress angle'])
                    data.append(f2)
                    targets.append(row['stress angle'])
                else: 
                    f = [f1,f2]
                    m  = np.mean(f, axis = 0)
                    std= np.std(f, axis = 0)
                    f = np.concatenate((m,std))
                    data.append(f)
                    targets.append(row['stress angle'])
        mean_HF_of_all_samples = pd.DataFrame(data=data)  
        if save:
            
            mean_HF_of_all_samples['targets'] = targets 
            mean_HF_of_all_samples.to_csv(images_path + save_name, sep = ";")
            mean_HF_of_all_samples = mean_HF_of_all_samples.drop(columns = ['targets'])
        return [mean_HF_of_all_samples, targets]
    else:
        data = pd.read_csv(images_path + save_name, sep = ";", error_bad_lines=False)
        targets = np.array(data['targets'])
        mean_HF_of_all_samples = data.drop(columns = ['targets'])
        return [mean_HF_of_all_samples, targets]



def trying_with_some_classifiers(data , targets):
    names = [
        "Nearest Neighbors",
        "Linear SVM",
        "RBF SVM",
        #"Gaussian Process",
        "Decision Tree",
        "Random Forest",
        
    ]
    classifiers = [

        KNeighborsClassifier(3),
        SVC(kernel="linear", C=0.025),
        SVC(gamma=2, C=1),
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
    n = 1000
    classifiers_score = []
    for name, rf in zip(names, classifiers):
        print(name)
        sum_score = 0
        for i in range(n):
            X_train, X_test, y_train, y_test = train_test_split(data.values, targets)
            rf.fit(X_train, y_train)
            #print(rf.predict_proba(X_test))
            score = rf.score(X_test, y_test)
            #print(score)
            sum_score += score 
            
        classifiers_score.append(sum_score/n)

    classifiers_score = np.asarray(np.round(classifiers_score, 4))
    colors = np.where(classifiers_score> 0.5, 'green','red')
   
    print(score)
    plt.bar(x = names, height = classifiers_score, width = 0.4, color = colors)
   
    for index,data in enumerate(classifiers_score):
        plt.text(x=index-0.3 , y =data+0.05 , s=f"{data}" , fontdict=dict(fontsize=10))
    plt.title("Results of some classifiers")
    plt.ylim(0,1)
    plt.tight_layout()
    plt.show()
    
def feature_selection(data, targets):
    feature_names = data.columns
    X = data.values
    
    y = targets
    n = np.zeros(len(feature_names))
    model = SVC(kernel="linear", C=0.025)
    #model = RandomForestClassifier(random_state=0)
    #model =  DecisionTreeClassifier(max_depth=5)
    #selector = RFECV(model, step = 1, cv=5, min_features_to_select=1)
    rfecv = RFECV(
        estimator=model,
        step=1,
        cv=5,
        scoring="accuracy",
        min_features_to_select=1,)
    rfecv.fit(X, y)
    print(rfecv.ranking_)
    

def feature_importance(data, targets):
    feature_names = data.columns
    X = data.values
    
    y = targets
    n = np.zeros(len(feature_names))
   
    #model =  KNeighborsClassifier(3)
    model = RandomForestClassifier(random_state=0)
    #model =  DecisionTreeClassifier(max_depth=5)
    #X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.3)
    #print(X_test, y_test)
    model.fit(X, y)
        # display the relative importance of each attribute
    impotances =  model.feature_importances_

   
    forest_importances = pd.Series(impotances, index = columns_name + columns_name_std)
    #std = np.std([tree.feature_importances_ for tree in model.estimators_], axis=0)
    mean_impotance = 1/len(feature_names) 
   
    fig, ax = plt.subplots()
    moreImpotance = (forest_importances > mean_impotance).astype(int).values
    colors = np.where(moreImpotance, 'green','red')
    forest_importances.plot.bar(ax = ax, color = colors)
    #ax.set_title("Random Forest Feature Importances")
    ax.set_ylabel("Feature Importances")
    fig.tight_layout()
    plt.show()
 
    #print(mean_impotance)
    for feature in feature_names: 
        
        if(forest_importances[feature] <= mean_impotance):
            data.drop(feature, inplace=True, axis=1)
            #test_x.drop(feature, inplace=True, axis=1)
    
    return data
def perform_PCA(data, targets):
        
    pca = PCA(n_components=2)
    principalComponent = pca.fit_transform(data.values)
    principalDf = pd.DataFrame(data = principalComponent, index = targets, columns=["P1", "P2"])

    fig  = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(1,1,1)
    ax.set_xlabel('P1', fontsize = 15)
    ax.set_ylabel('P2', fontsize = 15)
    ax.set_title('Correlation between texture features and stress angle', fontsize = 10)

    colors = ['r', 'g', 'b', 'c']
    ta = [0,30,60,90]
    for target, color in zip(ta, colors):
        ax.scatter(principalDf.loc[target, "P1"], principalDf.loc[target, "P2"], c = color, s = 50)

    #ax.legend(ta)
    ax.grid()
    plt.legend(ta, title = "stress angle")
    plt.show()




def try_classify_withLBP(images_path):
        # loop over the training images
    samples = pd.read_csv("Klebeverbindungen_Daten/Proben.csv" , sep=';', error_bad_lines=False)
    targets =   []
    samples = samples.set_index('ID')
    data = []
    for index, row in samples.iterrows():
        path1 = images_path + 'Probe' + index + "_1.png"
        path2 = images_path + 'Probe'+ index + "_2.png"
        
        if(os.path.isfile(path1) and os.path.isfile(path2)):
            # load the image, convert it to grayscale, and describe it
            image1= cv2.imread(path1)
            gray1= cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
            hist1 = texturefeaturesExtract.extract_LBP(gray1)
            image2= cv2.imread(path2)
            gray2= cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
            hist2 = texturefeaturesExtract.extract_LBP(gray2)
            # extract the label from the image path, then update the
            # label and data lists
            targets.append(index)
            data.append(hist1 + hist2)

    # train a Linear SVM on the data
    trying_with_some_classifiers(data, targets)
if __name__ == "__main__":
    #data, features =  load_features(images_path="Klebeverbindungen_Daten/2D-MakroImages/", save_name = "all_features.csv")
     
    #data, features =  load_features(images_path="Klebeverbindungen_Daten/2D-MakroImages/SikaPower533/",  extract= True)
        
    data, targets =  load_features(images_path="Klebeverbindungen_Daten/2D-MakroImages/BM/", save = True,extract = True, separat = False, save_name='Sk_2d.txt')
    
    data = feature_importance(data, targets)
    #feature_selection(data, targets)
    #print(data,targets)
    #try_classify_withLBP(images_path="Klebeverbindungen_Daten/2D-MakroImages/Betamate 1496V/")
    
    #
    perform_PCA(data, targets)
    trying_with_some_classifiers(data, targets)