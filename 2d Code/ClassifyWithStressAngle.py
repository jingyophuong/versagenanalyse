#from Phuong T. Pham - Begin: 29.10.2021
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



def load_features(images_path = "",  extract = False, save = False, save_name = ""):
    if(extract):  
        samples = pd.read_csv("Klebeverbindungen_Daten/Proben.csv" ,sep=';', error_bad_lines=False)
        targets =   []
        samples = samples.set_index('ID')
        data = []
        for index, row in samples.iterrows():
            path1 = images_path + 'Probe' + index + "_1.png"
            path2 =images_path + 'Probe'+ index + "_2.png"
          
            if(os.path.isfile(path1) and os.path.isfile(path2)):
                #
                #Haralick Features
                #
                f1 =texturefeaturesExtract.extract_haralick_features(0,path1, round_object='R' in index, underground=255)

                f2 = texturefeaturesExtract.extract_haralick_features(0, path2, round_object= 'R' in index, underground= 255)
                #s = np.maximum(f1, f2)
                d = np.abs(f2-f1) 
                # print(f)
                f = np.row_stack((f1, f2))
                #data.append(np.abs(np.diff(f, axis = 0)).T)
                #data.append(d / (np.mean(f, axis = 0).T))
                data.append(np.mean(f, axis = 0).T)
                #print(data)
                #data.append(f)

                #
                #GABOR FILTERS
                #
                # res = []
                # sobel_filter = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
                # gabor_kernels = texturefeaturesExtract.build_filters()
                # res1 = texturefeaturesExtract.getGaborFeatures(path1,sobel_filter, gabor_kernels)
                # res2 = texturefeaturesExtract.getGaborFeatures(path2,sobel_filter, gabor_kernels)
                # res.append(res1)
                # res.append(res2)
                # data.append(res)

                targets.append(row['stress angle'])
        mean_HF_of_all_samples = pd.DataFrame(data=data, columns=["Angular Second Moment", "Contrast", "Correlation", "Sum of Squares: Variance", "Inverse Difference Moment", "Sum Average", 
                           "Sum Variance", "Sum Entropy", "Entropy", "Difference Variance", "Difference Entropy", "Info. Measure of Correlation 1", "Info. Measure of Correlation 2"])  
        if save:
            save_data = mean_HF_of_all_samples
            save_data['targets'] = targets 
            mean_HF_of_all_samples.to_csv(images_path + save_name, sep = ",")
        return [mean_HF_of_all_samples, targets]
    else:
        data = pd.read_csv(images_path + save_name, sep = ",", error_bad_lines=False)
        targets = np.array(data['targets'])
        mean_HF_of_all_samples = data.drop(columns = ['targets'])
        return [mean_HF_of_all_samples, targets]





#Feature Importance
# load the iris datasets
#dataset = datasets.load_iris()
# fit an Extra Trees model to the data
def feature_importance(data, targets):
    feature_names=["Angular Second Moment", "Contrast", "Correlation", "Sum of Squares: Variance", "Inverse Difference Moment", "Sum Average", 
                            "Sum Variance", "Sum Entropy", "Entropy", "Difference Variance", "Difference Entropy", "Info. Measure of Correlation 1", "Info. Measure of Correlation 2"]
  
    X = data.values
    y = targets
    n = np.zeros(13)
    for i  in range(100):
        X_train, X_test, y_train, y_test = train_test_split(X, y)
        model = RandomForestClassifier(random_state=0)
        model.fit(X_train, y_train)
        # display the relative importance of each attribute
        n += model.feature_importances_

    impotances = n/100

    forest_importances = pd.Series(impotances, index=feature_names)
    print(forest_importances)
    std = np.std([tree.feature_importances_ for tree in model.estimators_], axis=0)
    mean_impotance = 1/13
    
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
    ax.set_title('Correlation between texture features and stress angle', fontsize = 10)

    colors = ['r', 'g', 'b', 'c']
    ta = [0,30,60,90]
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

from sklearn.svm import LinearSVC
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
        
    data, targets =  load_features(images_path="Klebeverbindungen_Daten/2D-MakroImages/new_sika2/", extract = True)
    data = feature_importance(data, targets)
    #try_classify_withLBP(images_path="Klebeverbindungen_Daten/2D-MakroImages/Betamate 1496V/")
    
    #
    perform_PCA(data, targets)
    #trying_with_some_classifiers(data, targets)