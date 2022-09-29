#from Phuong Pham - 18.02.2022 based on texture-feature sourcecode of skikit-image modul
#this file is used for compute the haralick features of a surface, included the round surface, by those the underground pixels shouldn't be counted on glcm
from asyncio.windows_events import NULL
import numpy as np
import mahotas as mt
import cv2
from _texture import (_glcm_loop)

class Haralick: 
    def object_repair(self, img, underground = 0):
        c =  img == underground
        return np.where(c, -1, img)

    def round_object_repair(self, img, underground = 0):
        width, height = img.shape
        r = int(min(width, height)/2)
        m_x = int(width / 2)
        m_y = int(height /2)
        img = np.array(img)

        I , J = np.ogrid[0: width, 0: height]

        dist = np.sqrt((I-m_x)** 2 + (J - m_y)**2)
        c =  np.logical_and(dist > r, img == underground)
        return np.where(c, -1, img)

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
                   
    def check_nD(array, ndim,  arg_name='image'):
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
    def __init__(self, image,  underground = 0, levels = None): #is_round_object = False,
        if(len(image.shape) > 2): 
            raise("Just grayscale image would be excepted!")
        #if(is_round_object): 
        #    self.image = self.round_object_repair(img = image, underground = underground)
            #print(self.image)
        #else:
            #self.image = image
        self.image = self.object_repair(img = image, underground = underground)
     
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
       
        self.distances = np.ascontiguousarray(self.distances, dtype=np.float64)
        self.angles = np.ascontiguousarray(self.angles, dtype=np.float64)
        #count coo histogramm
        rows, cols = image.shape

        _glcm_loop(image, self.distances, self.angles, self.levels, self.P)

        Pt = np.transpose(self.P, (1, 0, 2, 3))
        self.P = self.P + Pt

        self.P = [self.P[: , :, 0, 0], self.P[: , :, 0, 1], self.P[: , :, 0, 2], self.P[: , :, 0, 3]]
        return self.P
   
    def props(self):
        if self.P == NULL: 
            self.glcm()
        
        f = mt.features.texture.haralick_features(self.P)
        return f


def test():
    image = np.array([[0, 0, 1, 1,255],
                    [0, 0, 1, 1, 255],
                    [0, 2, 2, 2, 255],
                    [2, 2, 3, 3, 255]], dtype=np.uint8)

    image1 = np.array([[-1, 0, 0, 1, 1, -1],
                    [-1, 0, 0, 1, 1, -1],
                    [-1, 0, 2, 2, 2, -1],
                    [-1, 2, 2, 3, 3, -1]], dtype=np.int32)


    h = Haralick(image, True, 255)
    #k = h.round_object_repair(image, 255)
    print(h.image)

    print('mit meiner manipunierten Ansatz für image  \n', h.props().mean())

    h1 = Haralick(image1, True,255)
    print('mit meiner manipunierten Ansatz für image1  \n', h1.props().mean())
    print(h1.image)

    print()

if __name__ == '__main__':
    # image1 = cv2.imread(r"Klebeverbindungen_Daten\2D-MakroImages\Betamate 1496V\ProbeR1_1.png")
    # gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    
    # h = Haralick(gray1, True, 255)
    # print(h.props())
    test()