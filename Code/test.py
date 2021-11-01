#from Phuong T. Pham - Begin: 29.10.2021
import texturefeaturesExtract

import pandas as pd
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt 

#1.Step: loading the general information of all samples

samples = pd.read_csv("Klebeverbindungen_Daten/Proben.csv" ,sep=';', error_bad_lines=False)
samples = samples.set_index("ID")
#print(proben)

#2.Step: calculating the mean - HF for all samples

data = []
for index, row in samples.iterrows():
    path1 = "Klebeverbindungen_Daten/2D-MakroImages/Probe" + index + "_1.png"
    path2 = "Klebeverbindungen_Daten/2D-MakroImages/Probe" + index + "_2.png"
    HF_features = texturefeaturesExtract.extract_HF_mean_of_a_probe(path1, path2)
    #HF_features = np.append(HF_features, row['stress angle'])
    data.append(HF_features)
mean_HF_of_all_samples = pd.DataFrame(data=data, columns=["Angular Second Moment", "Contrast", "Correlation", "Sum of Squares: Variance", "Inverse Difference Moment", "Sum Average", 
                        "Sum Variance", "Sum Entropy", "Entropy", "Difference Variance", "Difference Entropy", "Info. Measure of Correlation 1", "Info. Measure of Correlation 2"])  
targets =   samples['stress angle'].to_numpy() 
mean_HF_of_all_samples['target'] = targets


mean_HF_of_all_samples.to_csv('Klebeverbindungen_Daten/meanHF.csv', index=False)



#3.Step: visualize all calculated features with pca

#angular second moment and Difference Variance delete
mean_HF_of_all_samples.drop("Angular Second Moment", inplace=True, axis=1)

mean_HF_of_all_samples.drop("Difference Entropy", inplace=True, axis=1)
mean_HF_of_all_samples.drop("Contrast", inplace=True, axis=1)

mean_HF_of_all_samples.drop("Correlation", inplace=True, axis=1)

mean_HF_of_all_samples.drop("Sum of Squares: Variance", inplace=True, axis=1)

mean_HF_of_all_samples.drop("Sum Variance", inplace=True, axis=1)
mean_HF_of_all_samples.drop("Sum Entropy", inplace=True, axis=1)
mean_HF_of_all_samples.drop("Info. Measure of Correlation 2", inplace=True, axis=1)


print(mean_HF_of_all_samples.head())

pca = PCA(n_components=2)
principalComponent = pca.fit_transform(mean_HF_of_all_samples.values)
principalDf = pd.DataFrame(data = principalComponent, index = targets, columns=["P1", "P2"])

print(principalDf)

fig  = plt.figure(figsize=(10,10))
ax = fig.add_subplot(1,1,1)
ax.set_xlabel('P1', fontsize = 15)
ax.set_ylabel('P2', fontsize = 15)
ax.set_title('2 component PCA', fontsize = 20)

colors = ['r', 'g', 'b', 'c']
ta = [0,30,60,90]
for target, color in zip(ta, colors):
    ax.scatter(principalDf.loc[target, "P1"], principalDf.loc[target, "P2"], c = color, s = 50)

ax.legend(ta)
ax.grid()
plt.show()

