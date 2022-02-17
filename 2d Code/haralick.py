from asyncio.windows_events import NULL
from turtle import distance
import jinja2
from matplotlib.pyplot import axis
from skimage.feature import greycomatrix
import numpy as np
import mahotas as mt

class Haralick: 
    
    haralick_labels = ["Angular Second Moment",
                   "Contrast",
                   "Correlation",
                   "Sum of Squares: Variance",
                   "Inverse Difference Moment",
                   "Sum Average",
                   "Sum Variance",
                   "Sum Entropy",
                   "Entropy",
                   "Difference Variance",
                   "Difference Entropy",
                   "Information Measure of Correlation 1",
                   "Information Measure of Correlation 2",
                   "Maximal Correlation Coefficient"]
    def check_nD(array, ndim, arg_name='image'):
        array = np.asanyarray(array)
        msg_incorrect_dim = "The parameter `%s` must be a %s-dimensional array"
        msg_empty_array = "The parameter `%s` cannot be an empty array"
        if isinstance(ndim, int):
            ndim = [ndim]
        if array.size == 0:
            raise ValueError(msg_empty_array % (arg_name))
        if array.ndim not in ndim:
            raise ValueError(
                msg_incorrect_dim % (arg_name, '-or-'.join([str(n) for n in ndim]))
            )
     # default constructor
    def __init__(self, image, levels = None):
        self.image = image
        self.distances = [1]
        self.P = NULL
        self.levels = levels
        self.angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]
    def glcm(self):
        #return greycomatrix(self.image, self.distances, self.angles, levels=4)
        #self.check_nD(self.image, 2)
        #self.check_nD(self.distances, 1, 'distance')
        
        image = np.ascontiguousarray(self.image)
        image_max = image.max()
        if self.levels is None:
            self.levels = image_max +1

        #if image_max >= self.levels:
            #raise ValueError("The maximum grayscale value in the image should be "
            #             "smaller than the number of levels.")

        self.P = np.zeros((self.levels, self.levels, len(self.distances), len(self.angles)), 
                        dtype=np.uint32, order='C')
       
        self.distances = np.ascontiguousarray(self.distances)
        self.angles = np.ascontiguousarray(self.angles)
        #count coo histogramm
        rows, cols = image.shape
        
        for a_idx in range(self.angles.shape[0]): 
            angle = self.angles[a_idx]
            for d_ix in range(self.distances.shape[0]):
                distance = self.distances[d_ix]

                offset_row = round(np.sin(angle) * distance)
                offset_col = round(np.cos(angle)*distance)

                start_row = int(max(0, -offset_row))
                end_row = min(rows, int(rows - offset_row))
                
                start_col = int(max(0, -offset_col))
                end_col = min (cols,int(cols - offset_col))
                for r in range(start_row, end_row):
                    for c in range(start_col, end_col):
                        i = image[r,c]
                        row = int(r + offset_row)
                        col = int(c + offset_col)
                        j = image[row,col]
                        if(0 <= i < self.levels and 0 <= j <self.levels):
                            self.P[i,j, d_ix, a_idx] +=1
        
        Pt = np.transpose(self.P, (1, 0, 2, 3))
        self.P = self.P + Pt

        self.P = [self.P[: , :, 0, 0], self.P[: , :, 0, 1], self.P[: , :, 0, 2], self.P[: , :, 0, 3]]
        return self.P
   
    def props(self):
        if self.P == NULL: 
            self.glcm()
        f = mt.features.texture.haralick_features(self.P)
        return f

image = np.array([[0, 0, 1, 1],
                  [0, 0, 1, 1],
                  [0, 2, 2, 2],
                  [2, 2, 3, 3]], dtype=np.uint8)

image1 = np.array([[-1, 0, 0, 1, 1],
                  [-1, 0, 0, 1, 1],
                  [-1, 0, 2, 2, 2],
                  [-1, 2, 2, 3, 3]], dtype=np.int32)


h = Haralick(image)

print('mit meiner manipunierten Ansatz für image  \n', h.props())

h1 = Haralick(image1)
print('mit meiner manipunierten Ansatz für image1  \n', h1.props())
