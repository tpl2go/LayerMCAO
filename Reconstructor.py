__author__ = 'tpl'

import scipy.io as sio
import subprocess
import numpy as np
import os

class ReconMethods(object):
    @staticmethod
    def LeastSquare(oXSlope,oYSlope):
        #TODO: translate matlab code to native python code
        if oXSlope.shape != oYSlope.shape:
            raise ValueError("Context: Trying to reconstruct surface from x,y gradient"
                             "Problem: x,y gradient matrices not of the same shape"
                             "Solution: Review the code that generated the gradient matrices")
        # Slope Arguments
        sio.savemat('slopes.mat',{'slopes':(oXSlope,oYSlope)})

        # Support Vector Arguments
        x = np.arange(oXSlope.shape[0])
        y = np.arange(oYSlope.shape[1])
        sio.savemat('support_vectors.mat',{'support_vectors':(x,y)})

        # Calling subprocess
        path = os.getcwd()
        print path
        subprocess.call(["matlab","-nodesktop","-nojvm","-nosplash","noFigurewindows","-r",\
                         "run('" + path + "/LeastSquareRecon.m');quit","1>/dev/null"])

        # Retrieving Results
        surface = sio.loadmat('surface.mat')
        surface = surface['surface']
        return surface
