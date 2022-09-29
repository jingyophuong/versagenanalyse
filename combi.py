from Code2d import texturefeaturesExtract
from Code3d.ClassifyWithStressAngle import trying_with_some_classifiers, feature_importance, perform_PCA
from Code3d.roughness2D import  compute_topography_of_a_probe_h
import os
import pandas as pd
import numpy as np
def classify_with_combi_textur_and_topo(images_path, topo_path, save = True, cal = False, save_file = 'tt_data.txt'):
    samples = pd.read_csv("Klebeverbindungen_Daten/Proben.csv" ,sep=';', error_bad_lines=False)
    targets =   []
    samples = samples.set_index('ID')
    data = pd.DataFrame()
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
                f = np.mean(f, axis = 0)

                ft = pd.DataFrame(data=[f], columns=["Angular Second Moment", "Contrast", "Correlation", "Sum of Squares: Variance", "Inverse Difference Moment", "Sum Average", 
                           "Sum Variance", "Sum Entropy", "Entropy", "Difference Variance", "Difference Entropy", "Info. Measure of Correlation 1", "Info. Measure of Correlation 2"]) 
              
            else: 
                full = False
            #topo
            path1 = topo_path + "Probe" + index + "_1-rel.txt"
            path2 = topo_path + "Probe" + index + "_2-rel.txt"
            if(os.path.isfile(path1) and os.path.isfile(path2)):
                roughness = compute_topography_of_a_probe_h(path1, path2)
            else: 
                full = False
            if full:
                f = pd.concat([ft, roughness], axis = 1)
                data = data.append(f)
                targets.append(row['stress angle'])
    if save:
        data['targets'] = targets
        data.to_csv('Klebeverbindungen_Daten/' + save_file + '.txt', index=False)
        data = data.drop(['targets'], axis = 1)
    
    data = feature_importance(data, targets)
   
    perform_PCA(data, targets)
    trying_with_some_classifiers(data, targets)


if __name__=='__main__':
    
    imagepath = r'Klebeverbindungen_Daten/2D-MakroImages/Betamate 1496V/'
    topopath = r'Klebeverbindungen_Daten/AP5-3D Punktwolken/BM/NoiseReduced/'
    classify_with_combi_textur_and_topo(imagepath, topopath, save = True, cal = True, save_file='bm-tt-data_alt.txt')