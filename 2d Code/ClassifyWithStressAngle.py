#from Phuong T. Pham - Begin: 29.10.2021
import texturefeaturesExtract
import os
import pandas as pd
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt 

# #1.Step: loading the general information of all samples

samples = pd.read_csv("Klebeverbindungen_Daten/Proben.csv" ,sep=';', error_bad_lines=False)
samples = samples.set_index("ID")
targets =   samples['stress angle'].to_numpy() 

feature_names = ["Angular Second Moment", "Contrast", "Correlation", "Sum of Squares: Variance", "Inverse Difference Moment", "Sum Average", 
                             "Sum Variance", "Sum Entropy", "Entropy", "Difference Variance", "Difference Entropy", "Info. Measure of Correlation 1", "Info. Measure of Correlation 2"]
# #print(proben)

# #2.Step: calculating the mean - HF for all samples, if the calculated values are already available, they should be called from the .csv file. 
# if os.path.isfile('Klebeverbindungen_Daten/meanHF.csv'):
#     mean_HF_of_all_samples = pd.read_csv('Klebeverbindungen_Daten/meanHF.csv', sep = ",")


# if(not(os.path.isfile('Klebeverbindungen_Daten/meanHF.csv')) or mean_HF_of_all_samples.empty or len(samples.index) != len(mean_HF_of_all_samples.index)):
#     data = []
#     for index, row in samples.iterrows():
#         path1 = "Klebeverbindungen_Daten/2D-MakroImages/Probe" + index + "_1.png"
#         path2 = "Klebeverbindungen_Daten/2D-MakroImages/Probe" + index + "_2.png"
#         HF_features = texturefeaturesExtract.extract_HF_mean_of_a_probe(path1, path2)
#         #HF_features = np.append(HF_features, row['stress angle'])
#         data.extend(HF_features)
#     mean_HF_of_all_samples = pd.DataFrame(data=data, columns=feature_names)  
    
#     mean_HF_of_all_samples.to_csv('Klebeverbindungen_Daten/meanHF.csv', index=False)



# #3.Step: visualize all calculated features with pca

# #mean_HF_of_all_samples.drop(feature_names[1], inplace=True, axis=1)
# #mean_HF_of_all_samples.drop(feature_names[3], inplace=True, axis=1)
# #mean_HF_of_all_samples.drop(feature_names[6], inplace=True, axis=1)

# print(mean_HF_of_all_samples.head())

# pca = PCA(n_components=2)
# principalComponent = pca.fit_transform(mean_HF_of_all_samples.values)
# principalDf = pd.DataFrame(data = principalComponent, index = targets, columns=["P1", "P2"])

# print(principalDf)

# fig  = plt.figure(figsize=(10,10))
# ax = fig.add_subplot(1,1,1)
# ax.set_xlabel('P1', fontsize = 15)
# ax.set_ylabel('P2', fontsize = 15)
# ax.set_title('Correlation between texture features and stress angle', fontsize = 20)

# colors = ['r', 'g', 'b', 'c']
# ta = [0,30,60,90]
# for target, color in zip(ta, colors):
#     ax.scatter(principalDf.loc[target, "P1"], principalDf.loc[target, "P2"], c = color, s = 50)

# #ax.legend(ta)
# ax.grid()
# plt.legend(ta, title = "stress angle")
# plt.show()



from sklearn import datasets
from sklearn import metrics
from sklearn.ensemble import ExtraTreesClassifier
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.cluster import KMeans


samples = pd.read_csv("Klebeverbindungen_Daten/Proben.csv" ,sep=';', error_bad_lines=False)
y =   samples['stress angle'].to_numpy()  / 30
samples = samples.set_index('ID')
#print(samples)
#2.Step: calculating the mean - HF for all samples, if the calculated values are already available, they should be called from the .csv file. 
if os.path.isfile('Klebeverbindungen_Daten/meanHF.csv'):
    mean_HF_of_all_samples = pd.read_csv('Klebeverbindungen_Daten/meanHF.csv', sep = ",")


if(not(os.path.isfile('Klebeverbindungen_Daten/meanHF.csv')) or mean_HF_of_all_samples.empty or len(samples.index) != len(mean_HF_of_all_samples.index)):
    data = []
    for index, row in samples.iterrows():
        path1 = "Klebeverbindungen_Daten/2D-MakroImages/Probe" + index + "_1.png"
        path2 = "Klebeverbindungen_Daten/2D-MakroImages/Probe" + index + "_2.png"
        HF_features = texturefeaturesExtract.extract_HF_mean_of_a_probe(path1, path2)
        #HF_features = np.append(HF_features, row['stress angle'])
        data.append(HF_features)
    mean_HF_of_all_samples = pd.DataFrame(data=data, columns=["Angular Second Moment", "Contrast", "Correlation", "Sum of Squares: Variance", "Inverse Difference Moment", "Sum Average", 
                            "Sum Variance", "Sum Entropy", "Entropy", "Difference Variance", "Difference Entropy", "Info. Measure of Correlation 1", "Info. Measure of Correlation 2"])  
    
    mean_HF_of_all_samples.to_csv('Klebeverbindungen_Daten/meanHF.csv', index=False)

X = mean_HF_of_all_samples.values

#print(X)
#print(X_reduced)


# # Recursive Feature Elimination
# from sklearn import datasets
# from sklearn.feature_selection import RFE
# from sklearn.linear_model import LogisticRegression
# from sklearn.tree import DecisionTreeClassifier
# estimator = RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1)
# selector = RFE(estimator, n_features_to_select=5, step=1)
# selector = selector.fit(X, y)
# print(selector.support_)
# print(selector.ranking_)



#Feature Importance
# load the iris datasets
#dataset = datasets.load_iris()
# fit an Extra Trees model to the data
n = np.zeros(13)
for i  in range(100):
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    model = RandomForestClassifier(random_state=0)
    model.fit(X_train, y_train)
    # display the relative importance of each attribute
    n += model.feature_importances_

impotances = n/100


import pandas as pd

forest_importances = pd.Series(impotances, index=feature_names)
print(forest_importances)
std = np.std([tree.feature_importances_ for tree in model.estimators_], axis=0)

fig, ax = plt.subplots()
forest_importances.plot.bar(yerr=std, ax=ax)
ax.set_title("Feature importances using MDI")
ax.set_ylabel("Mean decrease in impurity")
fig.tight_layout()
plt.show()


mean_impotance = np.mean(impotances)
print(mean_impotance)
for feature in feature_names: 
    
    if(forest_importances[feature] <= mean_impotance):
        mean_HF_of_all_samples.drop(feature, inplace=True, axis=1)

#print(mean_HF_of_all_samples.head())

pca = PCA(n_components=2)
principalComponent = pca.fit_transform(mean_HF_of_all_samples.values)
principalDf = pd.DataFrame(data = principalComponent, index = targets, columns=["P1", "P2"])

#print(principalDf)

fig  = plt.figure(figsize=(10,10))
ax = fig.add_subplot(1,1,1)
ax.set_xlabel('P1', fontsize = 15)
ax.set_ylabel('P2', fontsize = 15)
ax.set_title('Correlation between texture features and stress angle', fontsize = 20)

colors = ['r', 'g', 'b', 'c']
ta = [0,30,60,90]
for target, color in zip(ta, colors):
    ax.scatter(principalDf.loc[target, "P1"], principalDf.loc[target, "P2"], c = color, s = 50)

#ax.legend(ta)
ax.grid()
plt.legend(ta, title = "stress angle")
plt.show()


# from sklearn.inspection import permutation_importance
# import time

# X_train, X_test, y_train, y_test = train_test_split(X, y)
# model = RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1)
# model.fit(X_train,y_train)
# start_time = time.time()
# result = permutation_importance(
#     model, X_test, y_test, n_repeats=10, random_state=42, n_jobs=2)
# elapsed_time = time.time() - start_time
# print(f"Elapsed time to compute the importances: {elapsed_time:.3f} seconds")

# forest_importances = pd.Series(result.importances_mean, index=np.arange(13))

# fig, ax = plt.subplots()
# forest_importances.plot.bar(yerr=result.importances_std, ax=ax)
# ax.set_title("Feature importances using permutation on full model")
# ax.set_ylabel("Mean accuracy decrease")
# fig.tight_layout()
# plt.show()

#trying the classifiers
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

X_reduced = pca.fit_transform(mean_HF_of_all_samples.values)

for name, rf in zip(names, classifiers):
    sum_score = 0
    for i in range(100):
        X_train, X_test, y_train, y_test = train_test_split(X_reduced, y)
        rf.fit(X_train, y_train)
        #print(rf.predict_proba(X_test))
        score = rf.score(X_test, y_test)
        #print(score)
        sum_score += score 

    print('mean score of 100 test runs = ' + str(sum_score/100)+ 'with ' + name)

