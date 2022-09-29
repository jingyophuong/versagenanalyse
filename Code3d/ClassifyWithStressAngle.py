#from Phuong Pham - 01.2022 
#from torch import scatter

import roughness2D
import  roughness3D
import os
import pandas as pd
import numpy as np
import pandas as pd
import curvature
#from featurewiz import featurewiz
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler 


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


# def feature_selection(data, target):
#     return featurewiz(data, target, corr_limit=0.7, verbose=2, sep=",",
#                         header=0,test_data="", feature_engg="", category_encoders="")
def perform_PCA(data, targets):
        
    x = StandardScaler().fit_transform(data)
    pca = PCA(n_components=2)
    principalComponent = pca.fit_transform(x)
    principalDf = pd.DataFrame(data = principalComponent, index = targets, columns=["P1", "P2"])

    fig  = plt.figure(figsize=(5,5))
    ax = fig.add_subplot(1,1,1)
    ax.set_xlabel('P1', fontsize = 14)
    ax.set_ylabel('P2', fontsize = 14)
    #ax.set_title('Correlation between horizontal linear roughness and stress angle', fontsize = 10)

    colors = ['r', 'g', 'b', 'c']
    ta = np.unique(targets)
    for target, color in zip(ta, colors):
        ax.scatter(principalDf.loc[target, "P1"], principalDf.loc[target, "P2"], c = color, s = 50)

    #ax.legend(ta)
    ax.grid()
    plt.legend(ta, title = "stress angle")
    plt.show()



#trying the classifiers
def trying_with_some_classifiers_with_test(data_file, test_file):
    if os.path.isfile('Klebeverbindungen_Daten/' + data_file + '.txt') and os.path.isfile('Klebeverbindungen_Daten/' + test_file + '.txt'):
        data = pd.read_csv('Klebeverbindungen_Daten/' + data_file + '.txt', sep = ";")
        #print(data)
        targets = data.targets
        data = data.drop('targets', axis = 1)

        test_data = pd.read_csv('Klebeverbindungen_Daten/' + test_file + '.txt', sep = ";")
        test_targets = test_data.targets
        test_data = test_data.drop('targets', axis = 1)
  
    data = feature_importance(data, targets)
    for col in test_data.columns:
        if(col not in data.columns): 
            test_data = test_data.drop(col, axis = 1)
    print(data)
    print(test_data)
    data =  StandardScaler().fit_transform(data)
    test_data = StandardScaler().fit_transform(test_data)
  
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
        #SVC(kernel="linear", C=0.025),
        SVC(kernel="linear", C=5),
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
    n = 1000
    classifiers_score = []
    for name, rf in zip(names, classifiers):
        print(name)
        sum_score = 0
        for i in range(n):
            X_train, X_test, y_train, y_test = data, test_data, targets, test_targets
            rf.fit(X_train, y_train)
            #print(rf.predict_proba(X_test))
            score = rf.score(X_test, y_test)
            #print(score)
            sum_score += score 
        print(sum_score/n) 
        classifiers_score.append(sum_score/n)

    classifiers_score = np.asarray(np.round(classifiers_score, 4))
    colors = np.where(classifiers_score> 0.5, 'green','red')
   
    print(score)
    plt.figure(figsize=(5,5))
    plt.bar(x = names, height = classifiers_score, width = 0.4, color = colors)
   
    for index,data in enumerate(classifiers_score):
        plt.text(x=index-0.3 , y =data+0.05 , s=f"{data}" , fontdict=dict(fontsize=14))
    plt.title("Results of some classifiers")
    plt.ylim(0,1)
    plt.tight_layout()
    plt.show()
    

#trying the classifiers
def trying_with_some_classifiers(data , targets):
    
    data =  StandardScaler().fit_transform(data)

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
        #SVC(kernel="linear", C=0.025),
        SVC(kernel="linear", C=5),
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
    n = 1000
    classifiers_acc = []
    classifiers_cohen = []
    classifiers_matt = []
    for name, rf in zip(names, classifiers):
        print(name)
        acc_sum = 0.0
        cohen_sum = 0.0
        mat_sum = 0.0

        abs_sum = 0.0
        for i in range(n):
            X_train, X_test, y_train, y_test = train_test_split(data, targets, test_size = 0.3)
            rf.fit(X_train, y_train)
            #print( y_test)
            y_predict = rf.predict(X_test)

            #score = rf.score(X_test, y_test)
            #print(score)
            #sum_score += score
            # acurracy score
            acc_sum += accuracy_score(y_true=y_test, y_pred=y_predict)
            cohen_sum += cohen_kappa_score(y_test, y_predict)
            mat_sum += matthews_corrcoef(y_true=y_test, y_pred=y_predict)
        print('accurracy = ', acc_sum/n)
        print('cohen kappa score = ', cohen_sum/n)
        print('matthew cooffizient =' , mat_sum/n)
        print('abs', 1- abs_sum /n) 
        classifiers_acc.append(acc_sum/n *100)
        classifiers_cohen.append(cohen_sum/n *100)
        classifiers_matt.append(mat_sum/n *100)

    classifiers_acc = np.asarray(np.round(classifiers_acc , 2))
    classifiers_cohen = np.asarray(np.round(classifiers_cohen, 2))
    classifiers_matt = np.asarray(np.round(classifiers_matt, 2))
    
    #colors = np.where(classifiers_acc> 0.5, 'green','red')
   
    x = np.arange(len(classifiers))  # the label locations
    width = 0.25# the width of the bars

    fig, ax = plt.subplots(figsize=(5,5))  
    rects1 = ax.bar(x - width, classifiers_acc, width, label='Accuracy')
    rects2 = ax.bar(x , classifiers_cohen, width, label='Cohens Kappa Score')
    #rects3 = ax.bar(x + width, classifiers_matt, width, label='Matthews correlation coefficient')
    for bar in ax.patches:
        # The text annotation for each bar should be its height.
        bar_value = bar.get_height()
        # Format the text with commas to separate thousands. You can do
        # any type of formatting here though.
        text = f'{bar_value:,}'
        # This will give the middle of each bar on the x-axis.
        text_x = bar.get_x() + bar.get_width() / 2
        # get_y() is where the bar starts so we add the height to it.
        text_y = bar.get_y() + bar_value
        # If we want the text to be the same color as the bar, we can
        # get the color like so:
        bar_color = bar.get_facecolor()
        # If you want a consistent color, you can just set it as a constant, e.g. #222222
        ax.text(text_x, text_y, text, ha='center', va='bottom', color=bar_color,
                size=12)
    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Scores')
    ax.set_title('Classifiction metrics')
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.legend()
    
    ##ax.bar_label(rects1, padding=3)
    #ax.bar_label(rects2, padding=3)
    #ax.bar_label(rects3, padding=3)
    
    fig.tight_layout()
    plt.ylim([0,100])
    plt.show()
    

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

   
    forest_importances = pd.Series(impotances, index = feature_names)
    print(forest_importances)
    #std = np.std([tree.feature_importances_ for tree in model.estimators_], axis=0)
    mean_impotance = 1/len(feature_names) 
   
    # fig, ax = plt.subplots()
    # moreImpotance = (forest_importances > mean_impotance).astype(int).values
    # colors = np.where(moreImpotance, 'green','red')
    # forest_importances.plot.bar(ax = ax, color = colors)
    # ax.set_title("Random Forest Feature Importances")
    # ax.set_ylabel("Tree's Feature Importances")
    # fig.tight_layout()
    # plt.show()
 
    #print(mean_impotance)
    for feature in feature_names: 
        
        if(forest_importances[feature] < mean_impotance):
            data.drop(feature, inplace=True, axis=1)
            #test_x.drop(feature, inplace=True, axis=1)
    
    return data

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
 


   
def classify_with_roughness3d(data_dir, save = True, cal = False, save_file = ''):
    # #1.Step: loading the general information of all samples

    samples = pd.read_csv("Klebeverbindungen_Daten/Proben.csv" ,sep=';', error_bad_lines=False)
    samples = samples.set_index("ID")
    targets = []
    feature_names = ['Sa', 'Sq', 'Sz', 'Ssk', 'Ssu']
    data = pd.DataFrame()
    if os.path.isfile('Klebeverbindungen_Daten/' + save_file + '.txt') and not cal:
        data = pd.read_csv('Klebeverbindungen_Daten/' + save_file + '.txt', sep = ",")
        targets = data.targets
        data = data.drop('targets', axis = 1)
        #print(data)
    else:
  
        for index, row in samples.iterrows():
            path1 = data_dir +"/Probe" + index + "_1-rel.txt"
            path2 = data_dir+"/Probe" + index + "_2-rel.txt"
            
            
            if(os.path.isfile(path1) and os.path.isfile(path2)):
                print(index)
                
                roughness_surface = roughness3D.cal_Roughness_params(0,path1)
                roughness_surface =roughness_surface.append(roughness3D.cal_Roughness_params(0,path2), ignore_index = True)
                surface_topo = pd.DataFrame()
                for col_name in roughness_surface.columns: 
                    m = roughness_surface[col_name].mean()
                    std = roughness_surface[col_name].std()
                    #cov = std/m
                    diff = np.abs(roughness_surface[col_name][1] - roughness_surface[col_name][0] )
                    surface_topo['m_'+col_name]= [m]
                    surface_topo['std_'+col_name] = [std]
                    #surface_topo['diff_'+col_name] = [diff]

                data = data.append(surface_topo,ignore_index= True)
                targets.append(row['stress angle'])

    if save:  
        data['targets'] = targets  
        data.to_csv('Klebeverbindungen_Daten/' + save_file+ '.txt', index=False)
        data = data.drop(['targets'], axis = 1)
    print(data)
    data = feature_importance(data, targets)
    
    #data['targets'] = targets  
    #features, train = feature_selection(data, 'targets')
    #data = train.drop(['targets'],axis=1)
    

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

def classify_with_roughness2d(dir , cal = False, save = False, scan_direction_vectical = True, save_file = "roughness2d_y"):
    # #1.Step: loading the general information of all samples

    samples = pd.read_csv("Klebeverbindungen_Daten/Proben.csv" ,sep=';', error_bad_lines=False)
    samples = samples.set_index("ID")
    targets = []
    #feature_names = ['Sa', 'Sq', 'Sz', 'Ssk', 'Ssu']
    data = pd.DataFrame()
    #data = []
    if not cal:
        if os.path.isfile('Klebeverbindungen_Daten/' + save_file + '.txt'):
            sdata = pd.read_csv('Klebeverbindungen_Daten/' + save_file + '.txt', sep = ";")
            #print('sdata = ', sdata)
            targets = sdata['targets']
            data = sdata.drop(['targets'], axis=1)
    if cal: 
        count = int(len(os.listdir(dir)) / 2)
        i = 1
        for index, row in samples.iterrows():
        
            if scan_direction_vectical:   
                path1 = dir + "Probe" + index + "_1.txt"
                path2 = dir + "Probe" + index + "_2.txt"
                if(os.path.isfile(path1) and os.path.isfile(path2)):
                    print(index, ' ', i, '/', count)
                    roughness = roughness2D.compute_topography_of_a_probe_h(path1, path2)
                    
                    data = data.append(roughness, ignore_index=True)
                    #data.append(roughness_hist)
                    targets.append(row['stress angle'])
                    i+=1

            else: 
                path1 = dir + "Probe" + index + "_1.txt"
                path2 = dir + "Probe" + index + "_2.txt"
                if(os.path.isfile(path1) and os.path.isfile(path2)):
                    print(index, ' ', i, '/', count)
                    i+=1
                    roughness = roughness2D.compute_topography_of_a_probe_v(path1, path2)
                
                    data = data.append(roughness, ignore_index=True)
                    targets.append(row['stress angle'])

    if save:
        data['targets'] = targets
        data.to_csv('Klebeverbindungen_Daten/' + save_file + '.txt', index=False, sep = ';')
        data = data.drop(['targets'], axis = 1)
    print(data)
    data = feature_importance(data, targets)

    
    #data['targets'] = targets  
    #features, train = feature_selection(data, 'targets')
    #data = train.drop(['targets'],axis=1)
    
    #data = features_selection(data,targets)
    perform_PCA(data, targets)
    trying_with_some_classifiers(data, targets)
 

def classifywith_roughness2d(data_h_path, data_v_path):
    data = pd.DataFrame()
    targets = []
    if os.path.isfile(data_h_path) and os.path.isfile(data_v_path):
        data_x = pd.read_csv(data_h_path, sep = ";", header=None, skiprows = [0])
        data_x = data_x.iloc[: , :-1]
    
        data_y = pd.read_csv(data_v_path, sep = ";")
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

def classify_with_roughness2d_hv(dir , cal = False, save = False, save_file = "roughness2d_y"):
    # data = pd.DataFrame()
    # targets = []
    # if os.path.isfile(data_h_path):
    #     data_x = pd.read_csv(data_h_path, sep = ",", header=None, skiprows = [0])
    #     data_x = data_x.iloc[: , :-1]
        
    # if os.path.isfile(data_v_path):
    #     data_y = pd.read_csv(data_v_path, sep = ",")
    #     targets = data_y['targets']
    #     data_y = data_y.iloc[: , :-1]
    #     data = pd.concat([data_x, data_y], axis = 1)
        
    #     print(data, targets)
    # if(data.empty or len(targets) == 0):
    #     return
    # data = feature_importance(data, targets)
    # #data = features_selection(data,targets)
    # perform_PCA(data, targets)
    # trying_with_some_classifiers(data,targets)
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
            print(index)
              
            path1 = dir + "Probe" + index + "_1.txt"
            path2 = dir + "Probe" + index + "_2.txt"
            if(os.path.isfile(path1) and os.path.isfile(path2)):
            
                roughness = roughness2D.compute_topography_of_a_probe_hv(path1, path2)
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




def classifywith_roughness2dandAllcurvature(data_h_path, data_v_path, data_all_cur):
    data = pd.DataFrame()
    targets = []

    data_x = pd.DataFrame()
    if os.path.isfile(data_h_path):
        data_x = pd.read_csv(data_h_path, sep = ";")
        targets = data_x['targets']
        data_x = data_x.iloc[: , :-1]

    data_y =  pd.DataFrame()
    if os.path.isfile(data_v_path):
        data_y = pd.read_csv(data_v_path, sep = ";",header=None, skiprows=1)
        #targets = data_y['targets']
        data_y = data_y.iloc[: , :-1]
     
    data_c =  pd.DataFrame()
    if os.path.isfile(data_all_cur):
      
        data_c = pd.read_csv(data_all_cur, sep = ";")
        data_c= data_c.iloc[: , :-1]
    
    data = pd.concat([data_x, data_y, data_c], axis = 1)
    
    print(data, targets)

    if(data.empty or len(targets) == 0):
        return
    data = feature_importance(data, targets)
    #data = features_selection(data,targets)
    perform_PCA(data, targets)
    trying_with_some_classifiers(data,targets)



def classifywith_roughness2dandAllcurvatureAnd2d(data_h_path, data_v_path, data_all_cur, data_2d):
    data = pd.DataFrame()
    targets = []
    if os.path.isfile(data_h_path) and os.path.isfile(data_v_path)  and os.path.isfile(data_2d):
        data_x = pd.read_csv(data_h_path, sep = ";", header=None, skiprows = [0])
        data_x = data_x.iloc[: , :-1]
    
        data_y = pd.read_csv(data_v_path, sep = ";")
        targets = data_y['targets']
        data_y = data_y.iloc[: , :-1]
        data = pd.concat([data_x, data_y], axis = 1)

        #data = feature_importance(data, targets)

        # data_c = pd.read_csv(data_all_cur, sep = ";")
        # data_c= data_c.iloc[: , :-1]
        # data = pd.concat([data, data_c], axis = 1)

        data_2d = pd.read_csv(data_2d, sep = ";")
        data_2d= data_2d.iloc[: , :-1]
        data = pd.concat([data, data_2d], axis = 1)
    
    
        print(data.shape, targets)

        if(data.empty or len(targets) == 0):
            return
        data = feature_importance(data, targets)
        #data = features_selection(data,targets)
        perform_PCA(data, targets)
        trying_with_some_classifiers(data,targets)



def classify_surface_with_curvatures(data_dir, save = True, cal = False, curvature_version = 1, save_file= ''):
    # #1.Step: loading the general information of all samples

    samples = pd.read_csv("Klebeverbindungen_Daten/Proben.csv" ,sep=';', error_bad_lines=False)
    samples = samples.set_index("ID")
    targets = []
    #feature_names = ['Sa', 'Sq', 'Sz', 'Ssk', 'Ssu']
    #data = pd.DataFrame()
    
    data = []
    if os.path.isfile('Klebeverbindungen_Daten/curvatures.csv') and not cal:
        data = pd.read_csv('Klebeverbindungen_Daten/curvatures.csv', sep = ",")
        targets = data.targets
        data = data.drop('targets', axis = 1)
        #print(data)
    else:
        count = int(len(os.listdir(data_dir)) / 2)
        i = 1
        for index, row in samples.iterrows():
            path1 = data_dir +"/Probe" + index + "_1-rel-depthmap.png"
            path2 = data_dir+"/Probe" + index + "_2-rel-depthmap.png"
            path1csv= data_dir +"../FinalPC1/Probe" + index + "_1.txt"
            path2csv = data_dir+"../FinalPC1/Probe" + index + "_2.txt"
            
            if(os.path.isfile(path1) and os.path.isfile(path2)):
                print(index, i, '/', count)
                i+=1
                surface_curvatures1 = curvature.cal_curvatures_each_point_of_a_surface(path1, path1csv)
                surface_curvatures2 = curvature.cal_curvatures_each_point_of_a_surface(path2, path2csv)

                if curvature_version != 5: #maximal curvature
                    data.append((surface_curvatures1[curvature_version-1] + surface_curvatures2[curvature_version-1])/(surface_curvatures1[4]+ surface_curvatures2[4]))
                    #data.append((surface_curvatures1[curvature_version-1]+surface_curvatures2[curvature_version-1])/2)
                    #data.append(abs(surface_curvatures1[curvature_version-1]-surface_curvatures2[curvature_version-1]))
                
                if curvature_version == 5: 
                    
                    #for all of curvatures
                    max = (surface_curvatures1[0] + surface_curvatures2[0])/(surface_curvatures1[4]+ surface_curvatures2[4]) 
                    min = (surface_curvatures1[1] + surface_curvatures2[1])/(surface_curvatures1[4]+ surface_curvatures2[4]) 
                    gauss = (surface_curvatures1[2] + surface_curvatures2[2])/(surface_curvatures1[4]+ surface_curvatures2[4]) 
                    mean = (surface_curvatures1[3] + surface_curvatures2[3])/(surface_curvatures1[4]+ surface_curvatures2[4]) 
                    data.append(np.hstack((max, min, gauss, mean)))


                targets.append(row['stress angle'])
    n = len(data[0])

    
    #print(data)
    data = pd.DataFrame(data = data)
    data.columns=["F"+str(i) for i in range(0, n)]

    data = feature_importance(data, targets)
    if save:  
        data['targets'] = targets  
        data.to_csv('Klebeverbindungen_Daten/' + save_file + '.txt', sep = ';', index=False)
        data = data.drop(['targets'], axis = 1)
    
 
    perform_PCA(data, targets)

    trying_with_some_classifiers(data, targets)
 

##############################################################CLASSIFICATION-WITH-ROUGHNESS2D AND 3D#################################################################################
##############################################################CLASSIFICATION-WITH-ROUGHNESS2D AND 3D#################################################################################
##############################################################CLASSIFICATION-WITH-ROUGHNESS2D AND 3D#################################################################################
##############################################################CLASSIFICATION-WITH-ROUGHNESS2D AND 3D#################################################################################
##############################################################CLASSIFICATION-WITH-ROUGHNESS2D AND 3D#################################################################################
##############################################################CLASSIFICATION-WITH-ROUGHNESS2D AND 3D#################################################################################
def classify_with_combi_2dand3d(data_file_3d, data_file_2d):
    #2d
    if os.path.isfile('Klebeverbindungen_Daten/' + data_file_2d):
        sdata = pd.read_csv('Klebeverbindungen_Daten/' + data_file_2d, sep = ",")
          
        targets = sdata['targets']
        data2d = sdata.drop(['targets'], axis=1)
    if os.path.isfile('Klebeverbindungen_Daten/' + data_file_3d):
        sdata = pd.read_csv('Klebeverbindungen_Daten/' + data_file_3d, sep = ",")
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
    # classify_surface_with_curvatures( data_dir= r"Klebeverbindungen_Daten/AP5-3D Punktwolken/BM/Final/", cal = True, curvature_version= 5, save_file='bm_all_curvature')
    # classify_surface_with_curvatures( data_dir= r"Klebeverbindungen_Daten/AP5-3D Punktwolken/BM/Final/", cal = True, curvature_version= 2)
    # classify_surface_with_curvatures( data_dir= r"Klebeverbindungen_Daten/AP5-3D Punktwolken/BM/Final/", cal = True, curvature_version= 3)
    # classify_surface_with_curvatures( data_dir= r"Klebeverbindungen_Daten/AP5-3D Punktwolken/BM/Final/", cal = True, curvature_version= 4)

    #classify_with_roughness3d( data_dir= r"Klebeverbindungen_Daten/AP5-3D Punktwolken/BM/RelativData", cal = True, save_file='BMroughness3d')
    #classify_with_roughness2d('Klebeverbindungen_Daten/AP5-3D Punktwolken/BM/FinalPC/' , cal = False, save = True, save_file = "BMroughness2d_h_030922")
    #classify_with_roughness2d('Klebeverbindungen_Daten/AP5-3D Punktwolken/BM/FinalPC/' , cal = False, save = True, scan_direction_vectical=False, save_file = "BMroughness2d_v_030922")
    #plt.show()
    #classifywith_roughness2d('Klebeverbindungen_Daten/BMroughness2d_h_030922.txt', 'Klebeverbindungen_Daten/BMroughness2d_v_030922.txt')
    #classify_with_roughness2d_hv('Klebeverbindungen_Daten/AP5-3D Punktwolken/BM/FinalPC/' , cal = True, save = True, save_file = "BMroughness_hv_v5")
    #classify_with_combi_2dand3d('BMrough3d.csv', 'BMroughness2d_new_v5.txt')
    
    #classifywith_roughness2dandAllcurvature('Klebeverbindungen_Daten/BMroughness2d_h_040822.txt', 'Klebeverbindungen_Daten/BMroughness2d_v_040822.txt', 'Klebeverbindungen_Daten/bm_all_curvature.txt')
    classifywith_roughness2dandAllcurvature('Klebeverbindungen_Daten/BMroughness2d_h_030922.txt', 'Klebeverbindungen_Daten/BMroughness2d_v_030922.txt', 'Klebeverbindungen_Daten/bm_all_curvature.txt')
    
    #classifywith_roughness2dandAllcurvatureAnd2d('Klebeverbindungen_Daten/BMroughness2d_h_040822.txt', 'Klebeverbindungen_Daten/BMroughness2d_v_040822.txt', 'Klebeverbindungen_Daten/bm_all_curvature.txt', 'Klebeverbindungen_Daten/bm_2d.txt')
    #classifywith_roughness2dandAllcurvatureAnd2d('Klebeverbindungen_Daten/BMroughness2d_h_030922.txt', 'Klebeverbindungen_Daten/BMroughness2d_h_030922.txt', 'Klebeverbindungen_Daten/bm_all_curvature.txt', 'Klebeverbindungen_Daten/bm_2d.txt')


    #classify_with_roughness2d('Klebeverbindungen_Daten/AP5-3D Punktwolken/BM-TEST/FinalPC/' , cal = False, save = True, scan_direction_vectical=False, save_file = "BMroughness2d_v-test")
    #trying_with_some_classifiers_with_test('BMroughness2d_h_290722', 'BMroughness2d_h-test')



    #classify_with_roughness2d('Klebeverbindungen_Daten/AP5-3D Punktwolken/BM/FinalPC/RT2/' , cal = False, save = True, save_file = "BMroughness2d_h_RT2")
    #classify_with_roughness2d('Klebeverbindungen_Daten/AP5-3D Punktwolken/BM/FinalPC/RT2/' , cal = False, save = True, scan_direction_vectical=False, save_file = "BMroughness2d_v_RT2")
     
    # classify_surface_with_curvatures( data_dir= r"Klebeverbindungen_Daten/AP5-3D Punktwolken/BM/Final/RT2/", cal = True, curvature_version= 1)
    #classify_surface_with_curvatures( data_dir= r"Klebeverbindungen_Daten/AP5-3D Punktwolken/BM/Final1/", cal = True, curvature_version= 2)
    # classify_surface_with_curvatures( data_dir= r"Klebeverbindungen_Daten/AP5-3D Punktwolken/BM/Final/RT2/", cal = True, curvature_version= 3)
    # classify_surface_with_curvatures( data_dir= r"Klebeverbindungen_Daten/AP5-3D Punktwolken/BM/Final/RT2/", cal = True, curvature_version= 4)

    

    imagepath = r'Klebeverbindungen_Daten/2D-MakroImages/new_bm2/'
    topopath = r'Klebeverbindungen_Daten/AP5-3D Punktwolken/BM/NoiseReduced/'
    #classify_surface_with_curvatures(data_dir= r"Klebeverbindungen_Daten/AP5-3D Punktwolken/SK/Final/", cal = True, curvature_version= 5, save_file= 'sk_all_cur')
    # classify_surface_with_curvatures(data_dir= r"Klebeverbindungen_Daten/AP5-3D Punktwolken/SK/Final/", cal = True, curvature_version= 2)
    # classify_surface_with_curvatures(data_dir= r"Klebeverbindungen_Daten/AP5-3D Punktwolken/SK/Final/", cal = True, curvature_version= 3)
    # classify_surface_with_curvatures(data_dir= r"Klebeverbindungen_Daten/AP5-3D Punktwolken/SK/Final/", cal = True, curvature_version= 4)

    #classify_surface_with_curvatures( data_dir= r"Klebeverbindungen_Daten/AP5-3D Punktwolken/SK/RelativData/", cal = True, save_file = 'SKroughness3d')


    #classify_with_roughness3d( data_dir= r"Klebeverbindungen_Daten/AP5-3D Punktwolken/SK/RelativData", cal = True,save_file='SKroughness3d')
    #classify_with_roughness2d('Klebeverbindungen_Daten/AP5-3D Punktwolken/SK/FinalPC/' , cal = True, save = True, save_file = "SKroughness2d_v5_030922")
    #classify_with_roughness2d('Klebeverbindungen_Daten/AP5-3D Punktwolken//SK/FinalPC/' , cal = True, save = True, scan_direction_vectical=False, save_file = "Skroughness2d_v_030922")

    ##classifywith_roughness2d('Klebeverbindungen_Daten/SKroughness2d_v5_030922.txt', 'Klebeverbindungen_Daten/Skroughness2d_v_030922.txt')
    #classifywith_roughness2dandAllcurvature('Klebeverbindungen_Daten/SKroughness2d_v5_030922.txt', 'Klebeverbindungen_Daten/Skroughness2d_v_030922.txt','Klebeverbindungen_Daten/sk_all_cur.txt')
    #classifywith_roughness2dandAllcurvatureAnd2d('Klebeverbindungen_Daten/SKroughness2d_v5_040822.txt', 'Klebeverbindungen_Daten/Skroughness2d_v_040822.txt','Klebeverbindungen_Daten/sk_all_cur.txt', 'Klebeverbindungen_Daten/Sk_2d.txt')

    #classify_with_roughness2d('Klebeverbindungen_Daten/AP5-3D Punktwolken/SK/FinalPC/RT2/' , cal = False, save = True, scan_direction_vectical=True, save_file = "SKroughness2d_h-RT2")
    #classify_with_roughness2d('Klebeverbindungen_Daten/AP5-3D Punktwolken/SK/FinalPC/RT2/' , cal = True, save = True, scan_direction_vectical=False, save_file = "SKroughness2d_v-RT2")