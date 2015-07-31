__author__ = 'tpl'

from scipy.misc import imresize,lena
import numpy as np
import pickle
import matplotlib.pyplot as plt

def fit(true_screen, recon_screen):
    """

    :param true_screen: stacked phase screen
    :param recon_screen: the WFS's small reconstructed WF
    :return:
    """

    # TODO:THIS FUNCTION IS AN ULGY HACK; LOOK THROUGH IT LATER

    b = imresize(recon_screen,true_screen.shape,interp='cubic')
    a = true_screen

    #Getting shapes and prealocating the auxillairy variables
    k = np.shape(a)

    #Calculating mean values
    AM=np.mean(a)
    BM=np.mean(b)

    # Vectorized versions of c,d,e
    c_vect = (a-AM)*(b-BM)
    d_vect = (a-AM)**2
    e_vect = (b-BM)**2

    # Finally get r using those vectorized versions
    r_out = np.sum(c_vect)/float(np.sqrt(np.sum(d_vect)*np.sum(e_vect)))

    return r_out

def _SquaredDifferenceFunction(img, imgRef, xShift,yShift):
    assert img.shape[0]<imgRef.shape[0]

    # find the starting corner of imgRef
    x1 = imgRef.shape[0]/2.0 - img.shape[0]/2.0 + int(xShift)
    y1 = imgRef.shape[1]/2.0 - img.shape[1]/2.0 + int(yShift)

    diff = (img-imgRef[y1:y1+img.shape[1],x1:x1+img.shape[0]])**2

    return np.sum(np.sum(diff))

def SDF(img, imgRef, shiftLimit):

    c_matrix = np.zeros((2*shiftLimit+1,2*shiftLimit+1))

    bias = c_matrix.shape[0]/2

    for j in range(c_matrix.shape[0]):
        for i in range(c_matrix.shape[0]):
            c_matrix[j,i] = _SquaredDifferenceFunction(img,imgRef,i-bias,j-bias)

    return c_matrix

def c_to_s_matrix (c_matrix):
    min = np.unravel_index(c_matrix.argmin(),c_matrix.shape)
    print min

    if min[0] == 0 or min[0] == c_matrix.shape[0] \
        or min[1] == 0 or min[1] == c_matrix.shape[1]:
        raise RuntimeError("Minimum is found on an edge")

    s_matrix = c_matrix[min[0]-1:min[0]+2,min[1]-1:min[1]+2]

    return s_matrix

def TwoDLeastSquare(s_matrix):
    a2 = (np.average(s_matrix[:,2]) - np.average(s_matrix[:,0]))/2.0
    a3 = (np.average(s_matrix[:,2]) - 2*np.average(s_matrix[:,1]) + np.average(s_matrix[:,0])) / 2.0
    a4 = (np.average(s_matrix[2,:]) - np.average(s_matrix[0,:]))/2.0
    a5 = (np.average(s_matrix[2,:]) - 2*np.average(s_matrix[1,:]) + np.average(s_matrix[0,:])) / 2.0
    a6 = (s_matrix[2,2] - s_matrix[2,0] - s_matrix[0,2] + s_matrix[0,0]) / 4.0

    print (a2,a3,a4,a5,a6)

    # 1D Minimum (I have no idea what this means)
    # x_min = -a2/(2*a3)
    # y_min = -a4/(2*a5)

    # 2D Minimum
    x_min = (2*a2*a5 - a4*a6)/(a6**2 - 4*a3*a5)
    y_min = (2*a3*a4-a2*a6)/(a6**2 - 4*a3*a5)

    return (x_min,y_min)

def TwoDQuadratic(s_matrix):
    a2 = (s_matrix[1,2]-s_matrix[1,0])/2.0
    a3 = (s_matrix[1,2] - 2*s_matrix[1,1] + s_matrix[1,0])/2.0
    a4 = (s_matrix[2,1] - s_matrix[0,1])/2.0
    a5 = (s_matrix[2,1] - 2*s_matrix[1,1] + s_matrix[0,1])/2.0
    a6 = (s_matrix[2,2] - s_matrix[2,0] - s_matrix[0,2] + s_matrix[0,0]) / 4.0

    # 1D Minimum (I have no idea what this means)
    # x_min = -a2/(2*a3)
    # y_min = -a4/(2*a5)

    # 2D Minimum
    x_min = (2*a2*a5 - a4*a6)/(a6**2 - 4*a3*a5)
    y_min = (2*a3*a4-a2*a6)/(a6**2 - 4*a3*a5)

    return (x_min,y_min)


if __name__ == "__main__":
    f = open("TestDimg.dimg",'rb')
    dimg = pickle.load(f)
    c = SDF(dimg,lena(),3)
    s = c_to_s_matrix(c)
    print TwoDLeastSquare(s)
    plt.imshow(c)
    plt.colorbar()
    plt.show()



