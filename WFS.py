import numpy as np
from scipy.misc import lena
from Reconstructor import ReconMethods
import matplotlib.pyplot as plt
from Atmosphere import *
from Telescope import Telescope
import pickle
import sys
from PIL import Image
import cProfile


class WideFieldSHWFS(object):
    """
    Usage:
        Methods prefixed with "_" are internal functions for the inner workings of the WFS
        Methods prefixed with "display" are for user interactions
        Methods prefixed with "run" are used by external classes

    """

    def __init__(self, height, num_lenslet, pixels_lenslet, atmosphere, telescope):
        """
        :param height: Conjugated h/eight of the WFS [meters]
        :param num_lenslet: Number of lenslet spanning one side of pupil [int]
        :param pixels_lenslet: Number of detector pixels per lenslet [int]
        :param delta: Size of detector pixel [meters]
        :param atmosphere: Atmosphere object
        """

        # WFS attributes
        ### by default, WFS spans size of metapupil
        self.conjugated_height = height  # [meters]
        self.num_lenslet = num_lenslet  # [int]
        self.pixels_lenslet = pixels_lenslet  # [int]
        self.conjugated_size = telescope.pupil_diameter + telescope.field_of_view * height  # [meters]

        # Lenslet attributes (Conjugated)
        self.conjugated_delta = self.conjugated_size / float(num_lenslet) / float(pixels_lenslet)  # [meters]
        self.conjugated_lenslet_size = self.conjugated_size / float(num_lenslet)  # [meters]

        # Lenslet attributes (Physical)
        ### For physical sanity check only. Not needed for simulation
        self.lenslet_size = self.conjugated_lenslet_size / float(telescope.Mfactor)  # [meters]

        # Detector attributes
        ### Angular resolution valid for small angles only
        self.angular_res = telescope.field_of_view / pixels_lenslet  # [rad/pixel]

        # Relevant objects
        self.atmos = atmosphere
        self.tel = telescope

        # Experimental algorithms
        ### Strategy pattern for custom ImageSimulation / ImageInterpretor algorithm
        self.ImgSimulator = ImageSimulator(atmosphere, telescope, self)
        self.ImgInterpreter = ImageInterpreter(self.ImgSimulator)

    def _reconstruct_WF(self, slopes):
        """
        Reconstruct the WF surface from the slopes sensed by WFS.

        Usage: Many reconstruction algorithm exists. They have been packaged in a class.
         Change the choice of algorithm under the wrapper of this function
        :param slopes: (x-slope ndarray, y-slope ndarray) # [radian/meter ndarray]
        :return: # [radian ndarray]
        """
        # slopes = self._sense_slopes()
        surface = ReconMethods.LeastSquare(slopes[0], slopes[1])
        return surface

    def _get_metascreen(self, scrn):
        """
        Returns the portion of screen that has contributed information to the
        WFS

        Refer to diagram to understand this implementation
        :param scrn: the Screen object whose metascreen we are finding
        :return: portion of screen # [radian ndarray]
        """

        # basis parameters
        FoV = self.tel.field_of_view
        theta = FoV / 2.0
        radius = self.tel.pupil_diameter / 2.0

        # find meta radius
        if scrn.height > self.conjugated_height:
            meta_radius = radius + (scrn.height - self.conjugated_height) * np.tan(theta)
        elif scrn.height > self.conjugated_height / 2.0:
            meta_radius = radius + (self.conjugated_height - scrn.height) * np.tan(theta)
        else:
            meta_radius = radius + scrn.height * np.tan(theta)

        # convert to array indices
        x_mid = scrn.phase_screen.shape[0] / 2.0  # [index]
        y_mid = scrn.phase_screen.shape[1] / 2.0  # [index]

        # convert lenslets size to phase screen indices; constant for all directions
        sizeX = int(meta_radius / scrn.delta)  # [index]
        sizeY = int(meta_radius / scrn.delta)  # [index]

        # frame to capture
        x1 = int(x_mid) - sizeX
        x2 = int(x_mid) + sizeX
        y1 = int(y_mid) - sizeY
        y2 = int(y_mid) + sizeY

        return scrn.phase_screen[y1:y2, x1:x2]

    def _get_meta_pupil(self, scrn):
        size_of_WFS = self.num_lenslet * self.pixels_lenslet * self.conjugated_delta
        num_scrn_pixels = size_of_WFS / scrn.delta
        shapeX, shapeY = scrn.phase_screen.shape
        x1 = int(shapeX / 2.0 - num_scrn_pixels / 2.0)

        return scrn.phase_screen[x1:x1 + num_scrn_pixels, x1:x1 + num_scrn_pixels]

    def set_size(self, mode=None, **kwargs):
        """
        By default, the WFS is set to the size of the meta pupil.
        Use keyword argument to change the size of the WFS

        Available keywords:
        cpixelsize, pixelsize,clensletsize,lensletsize,cwfssize
        """
        pass

    def runWFS(self):
        """
        This is the main method for other classes to interact with this WFS class.
        Unlike display methods, this method always returns a value
        :return:
        """

        # This uses a hybrid template pattern, strategy pattern and delegation pattern :P
        all_dimg = self.ImgSimulator.all_dimg()
        all_slopes = self.ImgInterpreter.all_dimg_to_slopes(all_dimg)

        return all_slopes


# [Software Design] I am resisting the implementation of an abstract ImageInterpreter / ImageSimulator class because
# I do not want the contracts to be set in stone. This code is still highly developmental and fluid. And the fact
# that it is written in python means that it is still a highly dynamic and rapidly evolving piece of code.

class ImageSimulator(object):
    """
    ImageSimulator generates the subimages behind the Shack-Hartmann lenslets given information
    about the atmosphere, telescope and Shack-Hartmann WFS

    Algorithm:
     This version of ImageSimulator first generates a distortion map by stacking phase screens for each direction.
     Image is then distorted by shifting individual pixels. No smearing of pixels is implemented

    Architecture:
     This class is meant to be embeded within a SH-WFS class as a strategy.

    Contract:
     Implements all_dimg()

    """

    def __init__(self, atmosphere, telescope, SH_WFS):

        # Physical simulation objects
        self.atmos = atmosphere
        self.tel = telescope
        self.wfs = SH_WFS

        # Silence warning for numpy floating point operations
        np.seterr(invalid='ignore', divide='ignore')

        # Get the test image
        self.test_img = _TestImgGenerator.test_img2(self.wfs.pixels_lenslet, self.tel.field_of_view)

        # Eager initialization of theta map / conjugated position map
        self.theta_map = self._get_theta_map()
        self.c_pos_map = self._get_c_pos_map()

    def _get_lensletscreen(self, scrn, angle, c_lenslet_pos):
        """
        Portion of a screen seen by a lenslet in one particular direction
        :param scrn: Screen object
        :param angle: (x,y) angle of view from pupil # [radian tuple]
        :param c_lenslet_pos: (x,y) position of conjugated lenslet # [meter tuple]
        """
        theta_x = angle[0]  # [radian]
        theta_y = angle[1]  # [radian]

        c_lenslet_pos_x = c_lenslet_pos[0]  # [meters]
        c_lenslet_pos_y = c_lenslet_pos[1]  # [meters]

        # finding center of meta pupil
        pos_x = c_lenslet_pos_x + np.tan(theta_x) * (self.wfs.conjugated_height - scrn.height)  # [meters]
        pos_y = c_lenslet_pos_y + np.tan(theta_y) * (self.wfs.conjugated_height - scrn.height)  # [meters]

        # convert to array indices
        x_mid = scrn.phase_screen.shape[0] / 2.0 + pos_x / float(scrn.delta)  # [index]
        y_mid = scrn.phase_screen.shape[1] / 2.0 + pos_y / float(scrn.delta)  # [index]

        # convert lenslets size to phase screen indices; constant for all directions
        # TODO: This is repeated calculation. anyway to remove it?
        sizeX = int(self.wfs.pixels_lenslet / 2.0 * self.wfs.conjugated_delta / scrn.delta)  # [index]
        sizeY = int(self.wfs.pixels_lenslet / 2.0 * self.wfs.conjugated_delta / scrn.delta)  # [index]

        Vint = np.frompyfunc(int,1,1)
        # frame to capture
        x1 = Vint(x_mid) - sizeX
        x2 = Vint(x_mid) + sizeX
        y1 = Vint(y_mid) - sizeY
        y2 = Vint(y_mid) + sizeY

       # TODO: implement adaptive phase screen resize

        # grab a snapshot the size of a lenslet
        # TODO: find a nicer way to make slicing numpy aware
        if type(x1) == np.ndarray:
            output = np.empty((self.wfs.pixels_lenslet,self.wfs.pixels_lenslet),np.ndarray)
            for j in range(self.wfs.pixels_lenslet):
                for i in range(self.wfs.pixels_lenslet):
                    output[j,i] = scrn.phase_screen[y1[j,i]:y2[j,i],x1[j,i]:x2[j,i]]
        else:
            output = scrn.phase_screen[y1:y2,x1:x2]

        return output

    def _stack_lensletscreen(self, angle, c_lenslet_pos):
        """
        Stacks the portions of all the screens seen by a lenslet in a particular direction
        :param angle: (x,y) angle of view from pupil # [radian tuple]
        :param c_lenslet_pos: (x,y) position of conjugated lenslet # [meter tuple]
        :return: Net phase screen seen by the lenslet # [radian ndarray]
        """
        # # Sanity check: that detector is implemented correctly
        # assert angle[0] < self.tel.field_of_view / 2.0
        # assert angle[1] < self.tel.field_of_view / 2.0

        # Initialize output variable
        outphase = None

        # Assuming that atmos.scrns is reverse sorted in height
        for scrn in self.atmos.scrns:
            change = self._get_lensletscreen(scrn, angle, c_lenslet_pos)
            if outphase == None:
                outphase = change
            else:
                outphase = outphase + change

        return outphase

    def _screen_to_shifts(self, stacked_phase, angle=None):
        """
        Assumption: the tip-tilt distortion is more significant than other modes of optical abberation.
         This requires fried radius to be about the conjugated lenslet size
        Usage: The various possible algorithms used to tiltify are stored in a class. User can change
         the choice of algorithm under the wrapper of this function
        :param stacked_phase: Net phase screen seen by the lenslet # [radian ndarray]
        :return: (x_shift,y_shift) in conjugate image plane # [meter]
        """
        # Tilt introduced by phase screen
        # numpy awareness
        ## TODO: think of a better method
        # if stacked_phase.dtype == np.ndarray:
        #     VTiltify = np.vectorize(_TiltifyMethods.tiltify1)
        #     tilts = VTiltify(stacked_phase, self.tel.wavelength, self.wfs.conjugated_lenslet_size)
        # else:
        #     tilts = _TiltifyMethods.tiltify1(stacked_phase, self.tel.wavelength,
        #                                                     self.wfs.conjugated_lenslet_size)
        tilts = _TiltifyMethods.tiltify1(stacked_phase, self.tel.wavelength,
                                                            self.wfs.conjugated_lenslet_size)

        if angle == None:
            # Use small angle resolution
            xShift, yShift = tilts[0] / self.wfs.angular_res, tilts[1] / self.wfs.angular_res
        else:
            xShift = float(self.wfs.pixels_lenslet) / self.wfs.tel.field_of_view * \
                     np.tan(tilts[0]) * (1 - np.tan(angle[0]) ** 2) / (1 + np.tan(angle[0]) * np.tan(tilts[0]))
            yShift = float(self.wfs.pixels_lenslet) / self.wfs.tel.field_of_view * \
                     np.tan(tilts[1]) * (1 - np.tan(angle[1]) ** 2) / (1 + np.tan(angle[1]) * np.tan(tilts[1]))

        return (xShift, yShift)

    def _index_to_c_pos(self, i, j):
        """
        Generates the (x,y) conjugated position of lenslet given its indices
        :param i: x index # [int]
        :param j: y index # [int]
        :return: (x,y) conjugated position # [meter tuple]
        """
        bias = (self.wfs.num_lenslet - 1) / 2.0  # [index]
        # conjugated position of lenslet
        pos_x = (i - bias) * self.wfs.conjugated_lenslet_size  # [meters]
        pos_y = (j - bias) * self.wfs.conjugated_lenslet_size  # [meters]

        # numpy aware
        if type(pos_x) == np.ndarray:
            lenslet_pos = np.array((pos_x,pos_y))
        else:
            lenslet_pos = (pos_x, pos_y)

        return lenslet_pos

    def _vignettify(self, c_lenslet_pos, angle):
        # function is deprecated
        ### TODO: Try translating this to C code and see if there is performance improvement
        ### TODO: Any numerical / algebraic tricks to make this run faster?

        # find z
        theta_x = angle[0]
        theta_y = angle[1]

        x = c_lenslet_pos[0] - np.tan(theta_x) * self.wfs.conjugated_height  # figure out the +/- later
        y = c_lenslet_pos[1] - np.tan(theta_y) * self.wfs.conjugated_height  # figure out the +/- later

        # vignetting algorithm
        z = np.sqrt(x ** 2 + y ** 2)
        R = self.tel.pupil_diameter / 2.0
        r = self.wfs.conjugated_lenslet_size
        if z < R - r:  # Case 1
            p = 1
        elif z > (R + r):  # Case 2
            p = 0
        elif z > R:  # Case 3
            s = (R + r + z) / 2.0  # semiperimeter
            area = np.sqrt((s) * (s - r) * (s - z) * (s - R))  # Heron formula
            theta_R = np.arccos((R ** 2 + z ** 2 - r ** 2) / (2 * R * z))
            theta_r = np.arccos((r ** 2 + z ** 2 - R ** 2) / (2 * r * z))
            hat = 2 * ((0.5 * theta_R * R ** 2) + (0.5 * theta_r * r ** 2) - area)
            p = hat / (np.pi * r ** 2)

        else:  # Case 4
            theta_R = 2 * np.arccos((R ** 2 + z ** 2 - r ** 2) / (2 * R * z))
            theta_r = 2 * np.arcsin(np.sin(theta_R / 2.0) * R / r)
            tri = 0.5 * R ** 2 * np.sin(theta_R)
            cap = 0.5 * r ** 2 * theta_r - 0.5 * r ** 2 * np.sin(theta_r)
            cres = tri + cap - 0.5 * R ** 2 * theta_R
            p = 1 - (cres / (np.pi * r ** 2))

        return p

    def _get_vignette_mask(self, c_lenslet_pos):
        """
        Generates a vignette mask for a given lenslet image
         Mask values between 0 and 1
        :param c_lenslet_pos: (x,y) position of conjugated lenslet # [meter tuple]
        :return:vignette mask # [ndarray]
        """
        R = self.tel.pupil_diameter / 2.0
        r = self.wfs.conjugated_lenslet_size

        TX = self.theta_map[0, :, :]
        TY = self.theta_map[1, :, :]

        xx = c_lenslet_pos[0] - np.tan(TX) * self.wfs.conjugated_height
        yy = c_lenslet_pos[1] - np.tan(TY) * self.wfs.conjugated_height

        zz = np.sqrt(xx ** 2 + yy ** 2)

        mask = np.empty(zz.shape)

        ### Ugly but fast method.
        # TODO: Read how to extend numpy universal functions

        # Case 1
        mask[zz < (R - r)] = 1.0

        # Case 2
        mask[zz > (R + r)] = 0.0

        # Case 3
        ss = (R + r + zz) / 2.0  # semiperimeter
        area = np.sqrt((ss) * (ss - r) * (ss - zz) * (ss - R))  # Heron formula
        theta_R = np.arccos((R ** 2 + zz ** 2 - r ** 2) / (2 * R * zz))
        theta_r = np.arccos((r ** 2 + zz ** 2 - R ** 2) / (2 * r * zz))
        hat = 2 * ((0.5 * theta_R * R ** 2) + (0.5 * theta_r * r ** 2) - area)
        p = hat / (np.pi * r ** 2)
        mask[(R < zz) & (zz < (R + r))] = p[(R < zz) & (zz < (R + r))]

        # Case 4
        theta_R = 2 * np.arccos((R ** 2 + zz ** 2 - r ** 2) / (2 * R * zz))
        theta_r = 2 * np.arcsin(np.sin(theta_R / 2.0) * R / r)
        tri = 0.5 * R ** 2 * np.sin(theta_R)
        cap = 0.5 * r ** 2 * theta_r - 0.5 * r ** 2 * np.sin(theta_r)
        cres = tri + cap - 0.5 * R ** 2 * theta_R
        p = 1 - (cres / (np.pi * r ** 2))
        mask[((R - r) < zz) & (zz < R)] = p[((R - r) < zz) & (zz < R)]

        return mask

    def _get_theta_map(self, SmallApprox = True):
        """
        Generates a map of conjugated angles associated with each pixel in lenslet image
        :return: theta map # [radian ndarray]
        """
        bias = (self.wfs.pixels_lenslet - 1) / 2.0  # [index]

        X, Y = np.meshgrid(np.arange(self.wfs.pixels_lenslet),np.arange(self.wfs.pixels_lenslet),indexing = 'xy')
        X, Y = X - bias, Y - bias

        tan_x, tan_y = X * self.wfs.angular_res, Y * self.wfs.angular_res

        if SmallApprox == False:
            # Small angle approximation
            theta_x, theta_y = np.arctan(tan_x), np.arctan(tan_y)
        else:
            theta_x, theta_y = tan_x, tan_y

        return np.array((theta_x,theta_y))

    def _get_c_pos_map(self):
        X, Y = np.meshgrid(np.arange(self.wfs.num_lenslet),np.arange(self.wfs.num_lenslet),indexing = 'xy')
        bias = (self.wfs.num_lenslet - 1) / 2.0  # [index]
        c_pos_map = self._index_to_c_pos(X,Y)

        return c_pos_map

    def dmap(self, c_lenslet_pos):
        """
        Generates distortion map --- the (x,y) shift for every pixel in lenslet image

        Context: In layer oriented MCAO, if a WFS is misconjugated from a screen, different pixels, each representing
         different directions, see different parts
         of the screen and hence is shifted by different amounts
        :param c_lenslet_pos: (x,y) position of conjugated lenslet # [meter tuple]
        :return: distortion map --- (x-shift matrix, y-shift matrix) # [meter ndarray]
        """
        # Grab and stack sub screens for each pixel
        screens = self._stack_lensletscreen(self.theta_map,c_lenslet_pos)
        # convert sub screens to shifts
        shifts = self._screen_to_shifts(screens)

        return shifts

    def all_dmap(self):
        """
        Generates the (x,y) shift for every pixel in lenslet image for all lenslets
        :return: x-shifts y-shifts # [meters ndarray]
        """
        # Initialize output array
        output = np.empty((self.wfs.num_lenslet, self.wfs.num_lenslet), np.ndarray)

        # Iterate over lenslet index
        for j in range(self.wfs.num_lenslet):
            for i in range(self.wfs.num_lenslet):
                sys.stdout.write('\r' + "Now computing dmap index " + str((i, j)))
                c_lenslet_pos = self._index_to_c_pos(i, j)
                distortion = self.dmap(c_lenslet_pos)
                output[j, i] = distortion

        sys.stdout.write(" Done!")

        return output

    def dimg(self, c_lenslet_pos):
        """
        Generates the distorted image behind the given SH lenslet
        :param c_lenslet_pos: conjugated position of SH lenslet # [meter tuple]
        :return: distorted image # [ndarray]
        """
        # Call dmap
        distortion = self.dmap(c_lenslet_pos)

        x_distortion = distortion[0]
        y_distortion = distortion[1]

        # Crop test image
        assert x_distortion.shape[0] < self.test_img.shape[0]  # TODO: implement adaptive resize?
        x1 = self.test_img.shape[0] / 2 - x_distortion.shape[0] / 2
        oimg = np.zeros((self.wfs.pixels_lenslet, self.wfs.pixels_lenslet))

        # Distortion Process

        shift = lambda (x, y): (x + int(x_distortion[x, y]), y + int(y_distortion[x, y]))

        invalid_count = 0

        for j in range(oimg.shape[0]):
            for i in range(oimg.shape[1]):
                try:
                    pos = shift((i, j))
                    oimg[j, i] = self.test_img[x1 + pos[1], x1 + pos[0]]
                except IndexError:
                    invalid_count += 1

        # Vignetting Process
        mask = self._get_vignette_mask(c_lenslet_pos)
        oimg = oimg * mask

        print invalid_count

        return oimg

    def save_dimg(self, c_lenslet_pos):
        """
        A wrapper of ImageSimulator.dimg function
         Saves the dimg output
        :param c_lenslet_pos:
        :return:
        """
        dimg = self.dimg(c_lenslet_pos)
        filename = "TestDimg" + str(c_lenslet_pos) + "_" + self.hash_function() + ".dimg"
        f = open(filename, 'wb')
        pickle.dump(dimg, f, pickle.HIGHEST_PROTOCOL)
        print "A dimg has been saved: " + filename
        return dimg

    def all_dimg(self):
        """
        Generates the distorted image for every lenslet in the WFS
        :return: image ndarray
        """
        # Initialize output array
        output = np.empty((self.wfs.num_lenslet, self.wfs.num_lenslet), np.ndarray)

        # Convert
        for j in range(self.wfs.num_lenslet):
            for i in range(self.wfs.num_lenslet):
                sys.stdout.write('\r' + "Now computing dimg index " + str((i, j)))
                c_pos = self._index_to_c_pos(i, j)
                dimg = self.dimg(c_pos)
                output[j, i] = dimg

        sys.stdout.write(" Done!")

        return output

    def save_all_dimg(self):
        """
        A wrapper of ImageSimulator.all_dimg
         Saves the all_dimg output
        :return:
        """
        all_dimg = self.all_dimg()
        filename = "TestAllDimg" + self.hash_function() + ".dimg"
        f = open(filename, 'wb')
        pickle.dump(all_dimg, f, pickle.HIGHEST_PROTOCOL)

        print "An all_dimg has been saved: " + filename
        return all_dimg

    def get_test_img(self):
        """
        :return: portion of standard test image used to represent dmap
        """

        assert self.wfs.pixels_lenslet <= self.test_img.shape[0]

        # Crop standard test image
        x1 = self.test_img.shape[0] / 2 - self.wfs.pixels_lenslet / 2.0
        x2 = self.test_img.shape[0] / 2 + self.wfs.pixels_lenslet / 2.0
        cropped_img = self.test_img[x1:x2, x1:x2]

        return cropped_img

    def hash_function(self):
        # Pretty stupid hash but it works for my file naming purposes
        hash = str(self.wfs.conjugated_height) + str(self.wfs.num_lenslet) + str(self.wfs.pixels_lenslet)
        for scrn in self.atmos.scrns:
            hash = hash + str(scrn.height) + str(scrn.ID)
        return hash

    def all_vignette_mask(self):
        output = np.empty((self.wfs.num_lenslet, self.wfs.num_lenslet), np.ndarray)
        for j in range(self.wfs.num_lenslet):
            for i in range(self.wfs.num_lenslet):
                c_lenslet_pos = self._index_to_c_pos(i, j)
                output[j, i] = self._get_vignette_mask(c_lenslet_pos)
        return output


class ImageInterpreter(object):
    """
    ImageInterpreter extract the slope iformation above each lenslet given their subimages

    Algorithm:
     This version of ImageInterpreter implements corelation and interpolation algorithms found in
     "Evaluation of image-shift measurement algorithms for solar Shack-Hartmann wavefront sensors, 	Lofdahl, M. G(2010)"

    Architecture:
     This class is meant to be embeded within a SH-WFS class as a strategy object.

    Contract:
     Implements all_dimg_to_slopes()
    """

    def __init__(self, ImgSimulator):
        self.ImgSimulator = ImgSimulator
        # this class is a downstream user of the dimg products
        # if there is any problems the user can contact the manufacturer of the dimg products

    def _dmap_to_slope(self, distortion_map):
        """
        Assumption: There is a common shift component to all pixels of a distortion map
         This component comes from the phase screen that the lenslet is conjugated to.
         Misconjugated screens should have an average shift contribution close to zero
        :param distortion_map: (x,y) shift used to distort image. Matrix shape should be (2,N,N) # [meter ndarray]
        :return: slopes # [radian per mater]
        """
        # TODO: manage the units

        assert distortion_map.shape[0] == 2
        return (distortion_map[0].mean(), distortion_map[1].mean())

    def all_dmap_to_slopes(self, d_map_array):
        """
        Generate the net WF slopes sensed by WFS. Each lenslet acts as one gradient (slope)
        sensor
        :param d_map_array: 2D list of distortion maps # [meters ndarray list list]
        :return: (x-slopes, y-slopes) # [radian/meter ndarray]
        """
        (ySize, xSize) = d_map_array.shape
        slopes = np.zeros((2, ySize, xSize))

        for (j, line) in enumerate(d_map_array):
            for (i, dmap) in enumerate(line):
                (x, y) = self._dmap_to_slope(dmap)
                slopes[0, j, i] = x
                slopes[1, j, i] = y

        return slopes

    def all_dimg_to_slopes(self, all_dimg):
        return self.all_measure(all_dimg)

    def compare_refImg(self, dimg, recon_dimg):
        plt.figure(1)

        ax1 = plt.subplot(1, 3, 1)
        ax1.imshow(dimg)
        ax1.set_title("dimg")

        ax2 = plt.subplot(1, 3, 2)
        ax2.imshow(recon_dimg)
        ax2.set_title("recon_dimg")

        ax3 = plt.subplot(1, 3, 3)
        true_img = self.ImgSimulator.get_test_img()
        ax3.imshow(true_img)
        ax3.set_title("true_img")

        plt.show()

    def _measure_dimg(self, dimg, refImg):
        c_matrix = _CorrelationAlgorithms.SDF(dimg, refImg, 5)
        s_matrix = _InterpolationAlgorithms.c_to_s_matrix(c_matrix)
        shifts = _InterpolationAlgorithms.TwoDLeastSquare(s_matrix)

        return shifts

    def all_measure(self, all_dimg):
        ref = _RefImgReconMethods.recon1(all_dimg, self.ImgSimulator.all_vignette_mask())
        iMax, jMax = all_dimg.shape
        output = np.empty((2, jMax, iMax))
        for j in range(jMax):
            for i in range(iMax):
                (xShift, yShift) = self._measure_dimg(all_dimg[j, i], ref)
                output[0, j, i] = xShift
                output[1, j, i] = yShift

        return output


class SHWFSDemonstrator(object):
    """
    A static class containing methods to test, debug and demonstrate
    the operations of the wide field extended source SH WFS
    
    Example Usage:
      tel = Telescope(2.5)
     
      at = Atmosphere()
     
      at.create_default_screen(100,0)
     
      wfs = WideFieldSHWFS(100,16,128,at,tel)
     
      SHWFS_Demonstrator.display_comparison(wfs)
     
      SHWFS_Demonstrator.compare_dmaps(wfs)
    """

    @staticmethod
    def display_dmap(wfs, lenslet_pos):
        """
        Inspect the dmap that would have been produced by a lenslet at particular conjugated position

        Science:
         1) Observe a particular dmap
        :param wfs:
        :param lenslet_pos:
        :return:
        """

        dist_map = wfs.ImgSimulator.dmap(lenslet_pos)

        plt.figure(1)

        ax1 = plt.subplot(121)
        ax1.set_title("X-Distortion")
        plt.imshow(dist_map[0])
        plt.colorbar()

        ax2 = plt.subplot(122)
        ax2.set_title("Y-Distortion")
        plt.imshow(dist_map[1])
        plt.colorbar()

        plt.show()

    @staticmethod
    def display_dimg(wfs, c_lenslet_pos):
        """
        Inspect the dimg that would have been produced by a lenslet at particular conjugated position

        Science:
         1) Observe a particular dimg
        :param wfs:
        :param c_lenslet_pos:
        :return:
        """

        cropped_lena = wfs.ImgSimulator.get_test_img()
        oimg = wfs.ImgSimulator.dimg(c_lenslet_pos)

        plt.figure(1)

        ax1 = plt.subplot(121)
        ax1.set_title("True Image")
        plt.imshow(cropped_lena, cmap=plt.cm.gray)

        ax2 = plt.subplot(122)
        ax2.set_title("Distorted Image")
        plt.imshow(oimg, cmap=plt.cm.gray)

        plt.show()

    @staticmethod
    def display_all_dmap(wfs, axis=0):
        """
        Inspect the dmaps used by wfs.ImgSimulator to generate the dimgs

        Science:
         1) Observe variation in distortion maps with lenslet positions
        :param wfs:
        :param axis:
        :return:
        """
        # sanity check
        if axis != 0 and axis != 1:
            raise ValueError("\nContext: Displaying all distortion maps\n" +
                             "Problem: Choice of axes (x,y) invalid\n" +
                             "Solution: Input 'axis' argument should be 0 (x-axis) or 1 (y-axis)")

        all_dmaps = wfs.ImgSimulator.all_dmap()


        # Display process
        fig = plt.figure(1)
        for i in range(wfs.num_lenslet):
            for j in range(wfs.num_lenslet):
                plt.subplot(wfs.num_lenslet, wfs.num_lenslet, i * wfs.num_lenslet + j + 1)
                plt.axis('off')
                im = plt.imshow(all_dmaps[j, i][axis], vmin=-0.1, vmax=0.1)
        fig.subplots_adjust(right=0.8)
        cbar_ax = fig.add_axes([0.85, 0.15, 0.03, 0.7])
        fig.colorbar(im, cax=cbar_ax)
        plt.show()

    @staticmethod
    def display_all_dimg(wfs):
        """
        Inspect subimages produced on SH WHF's detector plane

        Science:
         1) Observe vignetting effect
         2) Observe variation in distortions
        :param wfs:
        :return:
        """
        all_dimg = wfs.ImgSimulator.all_dimg()

        # Display process
        plt.figure(1)
        # Iterate over lenslet index
        for j in range(wfs.num_lenslet):
            for i in range(wfs.num_lenslet):
                plt.subplot(wfs.num_lenslet, wfs.num_lenslet, j * wfs.num_lenslet + i + 1)
                plt.axis('off')
                plt.imshow(all_dimg[j, i], cmap=plt.cm.gray, vmax=256, vmin=0)

        plt.show()

    @staticmethod
    def display_comparison_screen(wfs):  ### TODO: fix broken methods
        slopes = wfs.runWFS()
        screen = wfs._get_metascreen(wfs.atmos.scrns[0])
        sensed = wfs._reconstruct_WF(slopes)

        plt.figure(1)

        ax1 = plt.subplot(131)
        ax1.set_title("True phase scrn")
        im1 = ax1.imshow(screen)
        plt.colorbar(im1)

        ax2 = plt.subplot(132)
        ax2.set_title("Immediate")
        im2 = ax2.imshow(SHWFSDemonstrator.immediatecutslopeNrecon(screen))
        plt.colorbar(im2)

        ax3 = plt.subplot(133)
        ax3.set_title("Reconstructed")
        im3 = ax3.imshow(sensed)
        plt.colorbar(im3)

        plt.show()

    @staticmethod
    def compare_dmaps(wfs):
        all_dmaps = wfs.ImgSimulator.all_dmap()

        # Display process
        fig1 = plt.figure(1)
        for j in range(wfs.num_lenslet):
            for i in range(wfs.num_lenslet):
                plt.subplot(wfs.num_lenslet, wfs.num_lenslet, j * wfs.num_lenslet + i + 1)
                plt.axis('off')
                im = plt.imshow(all_dmaps[j, i][0], vmin=-0.1, vmax=0.1)

        fig2 = plt.figure(2)
        sc = wfs._get_metascreen(wfs.atmos.scrns[0])
        num = sc.shape[0] / 16
        im = np.zeros((wfs.num_lenslet, wfs.num_lenslet))
        for j in range(wfs.num_lenslet):
            for i in range(wfs.num_lenslet):
                x, y = wfs.ImgSimulator._screen_to_shifts(sc[j * num:(j + 1) * num, i * num:(i + 1) * num])
                im[j, i] = x
        plt.imshow(im, interpolation='none')
        plt.show()

    @staticmethod
    def display_slopes(wfs):
        screen = wfs._get_metascreen(wfs.atmos.scrns[0])
        distmap_all = wfs._all_dmap()
        slopes = wfs._sense_slopes(distmap_all)
        plt.figure(1)
        plt.subplot(131)
        plt.imshow(screen)
        plt.subplot(132)
        plt.imshow(slopes[0])
        plt.colorbar()
        plt.subplot(133)
        plt.imshow(slopes[1])
        plt.colorbar()
        # plt.axis('off')
        plt.show()

    @staticmethod
    def display_angles(wfs):
        x = []
        y = []
        for j in range(wfs.pixels_lenslet):
            for i in range(wfs.pixels_lenslet):
                angle = wfs._pixel_to_angle(i, j)
                x.append(angle[0])
                y.append(angle[1])
        plt.scatter(x, y)
        plt.show()

    @staticmethod
    def display_vignette(wfs, c_lenslet_pos):
        """
        Demonstrates the vigenetting effect for a single lenslet
        :param c_lenslet_pos:
        :return:
        """

        img = wfs.ImgSimulator._get_vignette_mask(c_lenslet_pos)

        plt.imshow(img, cmap=plt.cm.gray)
        plt.colorbar()
        plt.show()

    @staticmethod
    def display_all_vignette(wfs):
        plt.figure(1)
        for j in range(wfs.num_lenslet):
            for i in range(wfs.num_lenslet):
                c_lenslet_pos = wfs.ImgSimulator._index_to_c_pos(i, j)
                img = wfs.ImgSimulator._get_vignette_mask(c_lenslet_pos)
                plt.subplot(wfs.num_lenslet, wfs.num_lenslet, i + j * wfs.num_lenslet + 1)
                plt.axis('off')
                plt.imshow(img, cmap=plt.cm.gray, vmin=0, vmax=1)
        plt.show()

    @staticmethod
    def immediatecutslopeNrecon(phasescreen, N=16):
        num = phasescreen.shape[0] / N
        slopes = np.empty((2, N, N))
        for j in range(N):
            for i in range(N):
                oXSlope, oYSlope = _TiltifyMethods.tiltify1(phasescreen[j * num:(j + 1) * num, i * num:(i + 1) * num])
                slopes[0, j, i] = oXSlope
                slopes[1, j, i] = oYSlope

        return ReconMethods.LeastSquare(slopes[0], slopes[1])

    @staticmethod
    def immediate(phasescreen, N=16):
        num = phasescreen.shape[0] / N
        slopes = np.empty((2, N, N))
        for j in range(N):
            for i in range(N):
                oXSlope, oYSlope = _TiltifyMethods.tiltify1(phasescreen[j * num:(j + 1) * num, i * num:(i + 1) * num])
                slopes[0, j, i] = oXSlope
                slopes[1, j, i] = oYSlope
        return slopes


class _TiltifyMethods(object):
    """
    A collection of methods to fit x,y tilts in radians to a sub phase screen

    By default, use tiltify1 (fastest)
    """

    @staticmethod
    def tiltify1(screen, wavelength, conjugated_lenslet_size):
        """
        Refer to diagram
        :return: G tilt # [radian tuple]
        """
        # Method 1 - Mean end to end difference
        # Numpy aware
        ### I think this is the best speed performance I can do without using C extensions
        if screen.dtype == np.ndarray:
            S = screen[0,0].shape[0]
            slope = np.empty((2,screen.shape[0],screen.shape[1]))
            for j in range(screen.shape[0]):
                for i in range(screen.shape[1]):
                    x_tilt_acc = (screen[j,i][:, S - 1] - screen[j,i][:, 0]).sum()
                    y_tilt_acc = (screen[j,i][S - 1, :] - screen[j,i][0, :]).sum()

                    oXTilt = x_tilt_acc * wavelength / (2 * np.pi * conjugated_lenslet_size)
                    oYTilt = y_tilt_acc * wavelength / (2 * np.pi * conjugated_lenslet_size)

                    slope[0,j,i] = oXTilt
                    slope[1,j,i] = oYTilt
        else:
            S = screen.shape[0]

            x_tilt_acc = (screen[:, S - 1] - screen[:, 0]).sum()
            y_tilt_acc = (screen[S - 1, :] - screen[0, :]).sum()

            oXTilt = x_tilt_acc * wavelength / (2 * np.pi * conjugated_lenslet_size)
            oYTilt = y_tilt_acc * wavelength / (2 * np.pi * conjugated_lenslet_size)

            slope = (oXTilt, oYTilt)

        return slope

    @staticmethod
    def tiltify2(screen):
        # Method 2 - Linear regression of mean

        # Get size of screen
        # use sum()/S instead of mean() for efficiency
        S = screen.shape[0]

        # Calculate basis x-axis
        base = np.arange(S)
        base_bar = base.sum() / float(S)
        baseV = base - base_bar
        var_base = (baseV ** 2).sum()

        # Calculate X slope
        xAves = screen.sum(axis=0) / float(S)  # mean
        xAvesV = xAves - xAves.sum() / S
        cov = (xAvesV * baseV).sum()
        oXSlope = (cov / var_base)

        # Calculate Y slope
        yAves = screen.sum(axis=1) / float(S)  # mean
        yAvesV = yAves - yAves.sum() / S
        cov = (yAvesV * baseV).sum()
        oYSlope = (cov / var_base)

        return (oXSlope, oYSlope)

    @staticmethod
    def tiltify3(screen):
        # Method 3 - mean of linear regression

        # Get size of screen
        # use sum()/S instead of mean() for efficiency
        S = screen.shape[0]

        # Calculate basis x-axis
        base = np.arange(S)
        base_bar = base.sum() / float(S)
        baseV = base - base_bar
        var_base = (baseV ** 2).sum()

        # Calculate X slope
        x_bars = screen.sum(axis=1) / float(S)
        XV = screen - x_bars.reshape((S, 1))  # XV = (x_i-x_bar)
        cov = ((XV * baseV).sum(axis=1)) / var_base  # cov[x,y] = Sigma[(x_i-x_bar)(y_i-y_bar)] / Var[x]
        oXSlope = cov.sum() / S

        # Calculate Y slope
        y_bars = screen.sum(axis=0) / float(S)
        YV = screen - y_bars  # YV = (y_i-y_bar)
        cov = ((YV * baseV.reshape((S, 1))).sum(axis=0)) / var_base  # cov[x,y] = Sigma[(x_i-x_bar)(y_i-y_bar)] / Var[x]
        oYSlope = cov.sum() / S

        return (oXSlope, oYSlope)


class _TestImgGenerator(object):
    @staticmethod
    def test_img1():
        return lena()

    @staticmethod
    def test_img2(pixels_lenslet, field_of_view):
        IM = Image.open('SunGranule2.jpg', 'r').convert('L')
        # figure out size of true image
        truth_size = int(pixels_lenslet * (80.0 / 3600 * np.pi / 180) / field_of_view)
        IM = IM.resize((truth_size, truth_size), Image.BICUBIC)
        return np.array(IM)

    @staticmethod
    def test_img3():
        # TODO: a method that adds noise and degrades the test image
        # refer to Lofdahl's paper
        pass


class _RefImgReconMethods(object):
    """
    Reconstruct Reference Image for use in slope extractor
    """

    @staticmethod
    def recon1(all_dimg, all_mask):
        """
        Reconstruct Reference Image by averaging image from all lensets.
        Accounts for vignetting effect.
        :param all_dimg:
        :return:
        """
        ave = np.zeros(all_dimg[0, 0].shape)
        for j in range(all_dimg.shape[0]):
            for i in range(all_dimg.shape[1]):
                ave = ave + all_dimg[j, i]

        # correct peripheral
        mask = np.zeros(all_dimg[0, 0].shape)
        for j in range(all_mask.shape[0]):
            for i in range(all_mask.shape[1]):
                mask = mask + all_mask[j, i]

        # divide!!
        ave = ave / mask

        return ave


class _CorrelationAlgorithms(object):
    @staticmethod
    def _SquaredDifferenceFunction(img, imgRef, xShift, yShift):
        # if img.shape[0] + xShift <= imgRef.shape[0]:
        #     # find the starting corner of imgRef
        #     x1 = imgRef.shape[0] / 2.0 - img.shape[0] / 2.0 + int(xShift)
        #     y1 = imgRef.shape[1] / 2.0 - img.shape[1] / 2.0 + int(yShift)
        # 
        #     diff = (img - imgRef[y1:y1 + img.shape[1], x1:x1 + img.shape[0]]) ** 2
        #     score = np.sum(np.sum(diff)) / (img.shape[0]*img.shape[1])
        # elif :
        #     x1t = 0
        #     x2t = img.shape[0] - xShift
        #     y1t = 0
        #     y2t = img.shape[1] - yShift
        #     x1r = xShift
        #     x2r = imgRef.shape[0]
        #     y1r = yShift
        #     y2r = imgRef.shape[1]
        #     if xShift<0:
        #         # swap x
        #         
        #     if yShift<0:
        #         x2t = img.shape[0]
        #         x1r = 0
        #     diff = img[y1t:y2t,x1t:x2t] - imgRef[y1r:y2r,x1r:x2r]

        assert img.shape == imgRef.shape
        print (xShift, yShift)
        x1t = 0
        x2t = img.shape[0] - xShift
        y1t = 0
        y2t = img.shape[1] - yShift
        x1r = xShift
        x2r = imgRef.shape[0]
        y1r = yShift
        y2r = imgRef.shape[1]
        if xShift < 0:
            # # swap x
            # tmp1, tmp2 = x1t,x2t
            # x1t, x2t = x1r, x2r
            # x1r, x2r = -tmp1, tmp2

            x1t = -xShift
            x2t = img.shape[0]
            x1r = 0
            x2r = imgRef.shape[0] + xShift

        if yShift < 0:
            # # swap y
            # tmp1, tmp2 = y1t,y2t
            # y1t, y2t = y1r, y2r
            # y1r, y2r = -tmp1, tmp2

            y1t = -yShift
            y2t = img.shape[1]
            y1r = 0
            y2r = imgRef.shape[1] + yShift
        diff = img[y1t:y2t, x1t:x2t] - imgRef[y1r:y2r, x1r:x2r]

        score = np.sum(np.sum(diff)) / ((img.shape[0] - xShift) * (img.shape[1] - yShift))
        return score

    @staticmethod
    def SDF(img, imgRef, shiftLimit):

        c_matrix = np.zeros((2 * shiftLimit + 1, 2 * shiftLimit + 1))

        bias = c_matrix.shape[0] / 2

        for j in range(c_matrix.shape[0]):
            for i in range(c_matrix.shape[0]):
                c_matrix[j, i] = _CorrelationAlgorithms._SquaredDifferenceFunction(img, imgRef, i - bias, j - bias)

        return c_matrix


class _InterpolationAlgorithms(object):
    @staticmethod
    def c_to_s_matrix(c_matrix):
        min = np.unravel_index(c_matrix.argmin(), c_matrix.shape)
        print min

        if min[0] == 0 or min[0] == c_matrix.shape[0] - 1 \
                or min[1] == 0 or min[1] == c_matrix.shape[1] - 1:
            raise RuntimeError("Minimum is found on an edge")

        s_matrix = c_matrix[min[0] - 1:min[0] + 2, min[1] - 1:min[1] + 2]

        return s_matrix

    @staticmethod
    def TwoDLeastSquare(s_matrix):
        a2 = (np.average(s_matrix[:, 2]) - np.average(s_matrix[:, 0])) / 2.0
        a3 = (np.average(s_matrix[:, 2]) - 2 * np.average(s_matrix[:, 1]) + np.average(s_matrix[:, 0])) / 2.0
        a4 = (np.average(s_matrix[2, :]) - np.average(s_matrix[0, :])) / 2.0
        a5 = (np.average(s_matrix[2, :]) - 2 * np.average(s_matrix[1, :]) + np.average(s_matrix[0, :])) / 2.0
        a6 = (s_matrix[2, 2] - s_matrix[2, 0] - s_matrix[0, 2] + s_matrix[0, 0]) / 4.0

        print (a2, a3, a4, a5, a6)

        # 1D Minimum (I have no idea what this means)
        # x_min = -a2/(2*a3)
        # y_min = -a4/(2*a5)

        # 2D Minimum
        x_min = (2 * a2 * a5 - a4 * a6) / (a6 ** 2 - 4 * a3 * a5)
        y_min = (2 * a3 * a4 - a2 * a6) / (a6 ** 2 - 4 * a3 * a5)

        return (x_min, y_min)

    @staticmethod
    def TwoDQuadratic(s_matrix):
        a2 = (s_matrix[1, 2] - s_matrix[1, 0]) / 2.0
        a3 = (s_matrix[1, 2] - 2 * s_matrix[1, 1] + s_matrix[1, 0]) / 2.0
        a4 = (s_matrix[2, 1] - s_matrix[0, 1]) / 2.0
        a5 = (s_matrix[2, 1] - 2 * s_matrix[1, 1] + s_matrix[0, 1]) / 2.0
        a6 = (s_matrix[2, 2] - s_matrix[2, 0] - s_matrix[0, 2] + s_matrix[0, 0]) / 4.0

        # 1D Minimum (I have no idea what this means)
        # x_min = -a2/(2*a3)
        # y_min = -a4/(2*a5)

        # 2D Minimum
        x_min = (2 * a2 * a5 - a4 * a6) / (a6 ** 2 - 4 * a3 * a5)
        y_min = (2 * a3 * a4 - a2 * a6) / (a6 ** 2 - 4 * a3 * a5)

        return (x_min, y_min)


if __name__ == '__main__':
    tel = Telescope(2.5)
    at = Atmosphere()
    at.create_screen(0.20, 2048, 0.01, 20, 0.01, 1000)
    wfs = WideFieldSHWFS(0, 16, 128, at, tel)
    print wfs.conjugated_lenslet_size
    print wfs.angular_res
    # cProfile.run('SHWFSDemonstrator.display_all_dmap(wfs)')
    SHWFSDemonstrator.display_dmap(wfs,(0,0))
    SHWFSDemonstrator.display_dimg(wfs,(0,0))
    # plt.imshow(at.scrns[0].phase_screen)
    # plt.colorbar()
    # plt.show()
    # print wfs.conjugated_lenslet_size
    # SHWFSDemonstrator.display_all_dmap(wfs)
    # SHWFSDemonstrator.display_comparison_screen(wfs)
    # SHWFSDemonstrator.display_all_dmap(wfs)
