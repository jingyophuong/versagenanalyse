#from Phuong T. Pham - Begin: 29.10.2021
import texturefeaturesExtract
import os
import pandas as pd
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt 

#1.Step: loading the general information of all samples

samples = pd.read_csv("Klebeverbindungen_Daten/Proben.csv" ,sep=';', error_bad_lines=False)
samples = samples.set_index("ID")
targets =   samples['stress angle'].to_numpy() 
print(samples)

#2.Step: calculating the mean - HF for all samples, if the calculated values are already available, they should be called from the .csv file. 


if(not(os.path.isfile('Klebeverbindungen_Daten/HF.csv'))):
    data = []
    grid = []
    for index, row in samples.iterrows():
        path1 = "Klebeverbindungen_Daten/2D-MakroImages/Probe" + index + "_1.png"
        path2 = "Klebeverbindungen_Daten/2D-MakroImages/Probe" + index + "_2.png"
        HF_features = texturefeaturesExtract.extract_HF_of_a_probe(path1, path2, axis=1)
        data.extend(HF_features)

        labels = texturefeaturesExtract.get_labels_of_grid()
        for i in range(40):
            d = [index + '_'+labels[i], row['stress angle'], row['stress rate'], row['temperature']]
            grid.append(d)
        #HF_features = np.append(HF_features, row['stress angle'])
        #data.append(HF_features)
    
    grid_data = pd.DataFrame(data=grid, columns=['ID', 'stress angle', 'stress rate', 'temperature'])  
    
    grid_HF_of_all_samples = pd.DataFrame(data=data, columns=["Angular Second Moment", "Contrast", "Correlation", "Sum of Squares: Variance", "Inverse Difference Moment", "Sum Average", 
                            "Sum Variance", "Sum Entropy", "Entropy", "Difference Variance", "Difference Entropy", "Info. Measure of Correlation 1", "Info. Measure of Correlation 2"])  
    
    grid_HF_of_all_samples.to_csv('Klebeverbindungen_Daten/HF.csv', index=False)
    grid_data.to_csv('Klebeverbindungen_Daten/grid_data.csv', index=False)

    print(grid_HF_of_all_samples.head())
else:
    grid_HF_of_all_samples = pd.read_csv('Klebeverbindungen_Daten/HF.csv', sep=',', error_bad_lines=False)


grid_samples = pd.read_csv("Klebeverbindungen_Daten/grid_data.csv" ,sep=',', error_bad_lines=False)

print(grid_samples.head())
targets =   grid_samples['stress angle'].to_numpy() 
#3.Step: visualize all calculated features with pca

#angular second moment and Difference Variance delete
grid_HF_of_all_samples.drop("Angular Second Moment", inplace=True, axis=1)

grid_HF_of_all_samples.drop("Difference Entropy", inplace=True, axis=1)
grid_HF_of_all_samples.drop("Contrast", inplace=True, axis=1)

grid_HF_of_all_samples.drop("Correlation", inplace=True, axis=1)

grid_HF_of_all_samples.drop("Sum of Squares: Variance", inplace=True, axis=1)

grid_HF_of_all_samples.drop("Sum Variance", inplace=True, axis=1)
grid_HF_of_all_samples.drop("Sum Entropy", inplace=True, axis=1)
grid_HF_of_all_samples.drop("Info. Measure of Correlation 2", inplace=True, axis=1)

pca = PCA(n_components=2)
principalComponent = pca.fit_transform(grid_HF_of_all_samples.values)
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

