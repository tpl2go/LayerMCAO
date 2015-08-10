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
import time


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

        # WFS information
        self.conjugated_height = height  # [meters]
        self.num_lenslet = num_lenslet  # [int]
        self.pixels_lenslet = pixels_lenslet  # [int]

        # physical lenslet information
        # For physical sanity check: Don't really need these information for the simulation
        meta_pupil_diameter = telescope.pupil_diameter + telescope.field_of_view * height
        self.delta = meta_pupil_diameter / float(num_lenslet) / float(pixels_lenslet) / float(
            telescope.Mfactor)  # [meters]
        self.lenslet_size = self.pixels_lenslet * self.delta  # [meters]
        self.lenslet_f = self.delta * self.pixels_lenslet / 2.0 / telescope.Mfactor / np.tan(
            telescope.field_of_view / 2.0)  # [meters]

        # conjugated lenslet information
        self.conjugated_delta = meta_pupil_diameter / float(num_lenslet) / float(pixels_lenslet)  # [meters]
        self.conjugated_lenslet_size = self.pixels_lenslet * self.conjugated_delta  # [meters]

        # relevant objects
        self.atmos = atmosphere
        self.tel = telescope
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

    def _get_sensed_screen(self, scrn):
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
        if mode == None:
            if len(kwargs) > 1:
                raise ValueError("Context: Resizing WFS\n" +
                                 "Problem: Too many arguments\n" +
                                 "Solution: Provide a single argument specifying size")
            elif len(kwargs) < 1:
                raise ValueError("Context: Resizing WFS\n" +
                                 "Problem: Too little arguments\n" +
                                 "Solution: Provide a single argument specifying size")

            key, value = kwargs.items[0]

            if key == "cpixelsize":
                # physical lenslet information
                # For physical sanity check: Don't really need these information for the simulation
                self.delta = kwargs[key] / self.tel.Mfactor  # [meters]
                self.lenslet_size = self.pixels_lenslet * self.delta  # [meters]
                self.lenslet_f = self.delta * self.pixels_lenslet / 2.0 / self.tel.Mfactor / np.tan(
                    self.tel.field_of_view / 2.0)  # [meters]

                # conjugated lenslet information
                self.conjugated_delta = kwargs[key]  # [meters]
                self.conjugated_lenslet_size = self.pixels_lenslet * self.conjugated_delta  # [meters]
            elif key == "pixelsize":
                # physical lenslet information
                # For physical sanity check: Don't really need these information for the simulation
                self.delta = kwargs[key]  # [meters]
                self.lenslet_size = self.pixels_lenslet * self.delta  # [meters]
                self.lenslet_f = self.delta * self.pixels_lenslet / 2.0 / self.tel.Mfactor / np.tan(
                    self.tel.field_of_view / 2.0)  # [meters]

                # conjugated lenslet information
                self.conjugated_delta = kwargs[key] * self.tel.Mfactor  # [meters]
                self.conjugated_lenslet_size = self.pixels_lenslet * self.conjugated_delta  # [meters]
            elif key == "clensletsize":
                # physical lenslet information
                # For physical sanity check: Don't really need these information for the simulation
                self.lenslet_size = kwargs[key] / self.tel.Mfactor  # [meters]
                self.delta = float(self.lenslet_size) / self.pixels_lenslet  # [meters]
                self.lenslet_f = self.delta * self.pixels_lenslet / 2.0 / self.tel.Mfactor / np.tan(
                    self.tel.field_of_view / 2.0)  # [meters]

                # conjugated lenslet information
                self.conjugated_lenslet_size = kwargs[key]  # [meters]
                self.conjugated_delta = self.conjugated_lenslet_size / self.pixels_lenslet  # [meters]
            elif key == "lensletsize":
                # physical lenslet information
                # For physical sanity check: Don't really need these information for the simulation
                self.lenslet_size = kwargs[key]  # [meters]
                self.delta = float(self.lenslet_size) / self.pixels_lenslet  # [meters]
                self.lenslet_f = self.delta * self.pixels_lenslet / 2.0 / self.tel.Mfactor / np.tan(
                    self.tel.field_of_view / 2.0)  # [meters]

                # conjugated lenslet information
                self.conjugated_lenslet_size = kwargs[key] * self.tel.Mfactor  # [meters]
                self.conjugated_delta = self.conjugated_lenslet_size / self.pixels_lenslet  # [meters]
            elif key == "cwfssize":
                # physical lenslet information
                # For physical sanity check: Don't really need these information for the simulation
                self.lenslet_size = float(kwargs[key]) / self.num_lenslet / self.tel.Mfactor  # [meters]
                self.delta = float(self.lenslet_size) / self.pixels_lenslet  # [meters]
                self.lenslet_f = self.delta * self.pixels_lenslet / 2.0 / self.tel.Mfactor / np.tan(
                    self.tel.field_of_view / 2.0)  # [meters]

                # conjugated lenslet information
                self.conjugated_lenslet_size = float(kwargs[key]) / self.num_lenslet  # [meters]
                self.conjugated_delta = self.conjugated_lenslet_size / self.pixels_lenslet  # [meters]
            else:
                raise ValueError("Context: Resizing WFS\n" +
                                 "Problem: Key not understood\n" +
                                 "Solution: Use either cpixelsize,pixelsize,clensletsize,lensletsize,cwfssize")
        elif mode == 'pupil':
            # physical lenslet information
            # For physical sanity check: Don't really need these information for the simulation
            self.delta = self.tel.pupil_diameter / float(self.num_lenslet) / float(self.pixels_lenslet) / float(
                self.tel.Mfactor)  # [meters]
            self.lenslet_size = self.pixels_lenslet * self.delta  # [meters]
            self.lenslet_f = self.delta * self.pixels_lenslet / 2.0 / self.tel.Mfactor / np.tan(
                self.tel.field_of_view / 2.0)  # [meters]

            # conjugated lenslet information
            self.conjugated_delta = self.tel.pupil_diameter / float(self.num_lenslet) / float(
                self.pixels_lenslet)  # [meters]
            self.conjugated_lenslet_size = self.pixels_lenslet * self.conjugated_delta  # [meters]
        elif mode == 'metapupil':
            # physical lenslet information
            # For physical sanity check: Don't really need these information for the simulation
            meta_pupil_diameter = self.tel.pupil_diameter + self.tel.field_of_view * self.conjugated_height
            self.delta = meta_pupil_diameter / float(self.num_lenslet) / float(self.pixels_lenslet) / float(
                self.telescope.Mfactor)  # [meters]
            self.lenslet_size = self.pixels_lenslet * self.delta  # [meters]
            self.lenslet_f = self.delta * self.pixels_lenslet / 2.0 / self.tel.Mfactor / np.tan(
                self.tel.field_of_view / 2.0)  # [meters]

            # conjugated lenslet information
            self.conjugated_delta = meta_pupil_diameter / float(self.num_lenslet) / float(
                self.pixels_lenslet)  # [meters]
            self.conjugated_lenslet_size = self.pixels_lenslet * self.conjugated_delta  # [meters]

    def runWFS(self):
        """
        This is the main method for other classes to interact with this WFS class.
        Unlike display methods, this method always returns a value
        :return:
        """
        all_dimg = self.ImgSimulator.all_dimg()
        all_slopes = ImageInterpreter.all_dming_to_slopes(all_dimg)

        return all_slopes


class ImageSimulator(object):
    """
    Image Simulator aims to accurately generate the subimages behind the Shack-Hartmann lenslets, given information
    about the atmosphere, telescope and Shack-Hartmann WFS
    At the moment, ImageSimulator distorts an image by shifting individual pixels. No smearing of pixels is implemented

    This class is meant to be embeded within a SH-WFS class. It can generate individual/all distortion maps
     (dmap), individual/all distorted images (dimg). dimg takes vignetting effects into account.
    """

    def __init__(self, atmosphere, telescope, SH_WFS):
        self.atmos = atmosphere
        self.tel = telescope
        self.wfs = SH_WFS
        np.seterr(invalid='ignore')
        # get test image of solar granule
        # self.test_img = np.loadtxt(open('SunTesImage','rb'))
        IM = Image.open('SunGranule2.jpg', 'r').convert('L')
        # figure out size of true image
        truth_size = int(self.wfs.pixels_lenslet * (70.0 / 3600 * np.pi / 180) / self.tel.field_of_view)
        IM = IM.resize((truth_size, truth_size), Image.BICUBIC)
        self.test_img = np.array(IM)
        # self.test_img = self.test_img[:,:,0]
        # self.test_img = lena()

        # Eager initialization of theta map
        self.theta_map = self._get_theta_map()

        # Eager initialization of conjugated position map
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

        # frame to capture
        x1 = int(x_mid) - sizeX
        x2 = int(x_mid) + sizeX
        y1 = int(y_mid) - sizeY
        y2 = int(y_mid) + sizeY

        # sanity check
        # TODO remove this check by automatically doubling the phase screen size
        assert x1 > 0
        assert x2 < scrn.phase_screen.shape[0]
        assert y1 > 0
        assert y2 < scrn.phase_screen.shape[1]

        # grab a snapshot the size of a lenslet
        output = scrn.phase_screen[y1:y2, x1:x2]

        return output

    def _stack_lensletscreen(self, angle, c_lenslet_pos):
        """
        Stacks the portions of all the screens seen by a lenslet in a particular direction
        :param angle: (x,y) angle of view from pupil # [radian tuple]
        :param c_lenslet_pos: (x,y) position of conjugated lenslet # [meter tuple]
        :return: Net phase screen seen by the lenslet # [radian ndarray]
        """
        # sanity check: that lenslet_f is implemented correctly
        assert angle[0] < self.tel.field_of_view / 2.0
        assert angle[1] < self.tel.field_of_view / 2.0

        # initialize output variable
        outphase = None

        # Assuming that atmos.scrns is reverse sorted in height
        for scrn in self.atmos.scrns:
            change = self._get_lensletscreen(scrn, angle, c_lenslet_pos)
            if outphase == None:
                outphase = change
            else:
                outphase = outphase + change

        return outphase

    def _screen_to_shifts(self, stacked_phase):
        """
        Slopifies the net phase screen.

        Assumption: the tip-tilt distortion is more significant than other modes of optical abberation.
         This requires fried radius to be about just as large as conjugated lenslet size
        Usage: The various possible algorithms used to slopify are stored in a class. User can change
         the choice of algorithm under the wrapper of this function
        :param stacked_phase: Net phase screen seen by the lenslet # [radian ndarray]
        :return: (x_shift,y_shift) in conjugate image plane # [meter]
        """


        slope = _SlopifyMethods.slopify1(stacked_phase)

        # TODO: manage the units of the shifts in image plane
        # TODO: from radians/meter to meter
        (x_shift, y_shift) = slope

        return (x_shift, y_shift)

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

    def _get_theta_map(self):
        """
        Generates a map of conjugated angles associated with each pixel in lenslet image
        :return: theta map # [radian ndarray]
        """
        theta_map = np.empty((2, self.wfs.pixels_lenslet, self.wfs.pixels_lenslet))

        for j in range(self.wfs.pixels_lenslet):
            for i in range(self.wfs.pixels_lenslet):
                bias = (self.wfs.pixels_lenslet - 1) / 2.0  # [index]
                # physical position of detector pixel
                pos_x = (i - bias) * self.wfs.conjugated_delta / self.tel.Mfactor  # [meters]
                pos_y = (j - bias) * self.wfs.conjugated_delta / self.tel.Mfactor  # [meters]

                # convert physical position to physical angle (incident on physical lenslet)
                tan_x = pos_x / float(self.wfs.lenslet_f)
                tan_y = pos_y / float(self.wfs.lenslet_f)

                # convert physical angle to conjugate angle (incident on conjugate lenslet)
                theta_x = np.arctan(tan_x / self.tel.Mfactor)
                theta_y = np.arctan(tan_y / self.tel.Mfactor)

                theta_map[0, j, i] = theta_x
                theta_map[1, j, i] = theta_y

        return theta_map

    def _get_c_pos_map(self):
        output = np.empty((2,self.wfs.num_lenslet, self.wfs.num_lenslet), np.ndarray)
        for j in range(self.wfs.num_lenslet):
            for i in range(self.wfs.num_lenslet):
                c_pos = self._index_to_c_pos(i, j)
                output[0,j, i] = c_pos[0]
                output[1,j, i] = c_pos[1]

        return output


    def dmap(self, c_lenslet_pos):
        """
        Generates distortion map --- the (x,y) shift for every pixel in lenslet image

        Context: In layer oriented MCAO, if a WFS is misconjugated from a screen, different pixels, each representing
         different directions, see different parts
         of the screen and hence is shifted by different amounts
        :param c_lenslet_pos: (x,y) position of conjugated lenslet # [meter tuple]
        :return: distortion map --- (x-shift matrix, y-shift matrix) # [meter ndarray]
        """
        # initialize output variables
        oXSlope = np.zeros((self.wfs.pixels_lenslet, self.wfs.pixels_lenslet))
        oYSlope = np.zeros((self.wfs.pixels_lenslet, self.wfs.pixels_lenslet))

        # convert detector pixel index to angles
        bias = (self.wfs.pixels_lenslet - 1) / 2.0  # [index]
        for i in range(self.wfs.pixels_lenslet):
            for j in range(self.wfs.pixels_lenslet):
                # physical position of detector pixel
                pos_x = (i - bias) * self.wfs.conjugated_delta / self.tel.Mfactor  # [meters]
                pos_y = (j - bias) * self.wfs.conjugated_delta / self.tel.Mfactor  # [meters]

                # convert physical position to physical angle (incident on physical lenslet)
                tan_x = pos_x / float(self.wfs.lenslet_f)
                tan_y = pos_y / float(self.wfs.lenslet_f)

                # convert physical angle to conjugate angle (incident on conjugate lenslet)
                theta_x = np.arctan(tan_x / self.tel.Mfactor)
                theta_y = np.arctan(tan_y / self.tel.Mfactor)

                # stack metapupils and slopify
                screen = self._stack_lensletscreen((theta_x, theta_y), c_lenslet_pos)
                (x_shift, y_shift) = self._screen_to_shifts(screen)

                oXSlope[j, i] = x_shift
                oYSlope[j, i] = y_shift

        return np.array([oXSlope, oYSlope])

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
        scale = 50  # fudge factor

        shift = lambda (x, y): (x + int(scale * x_distortion[x, y]), y + int(scale * y_distortion[x, y]))

        for j in range(oimg.shape[0]):
            for i in range(oimg.shape[1]):
                try:
                    pos = shift((i, j))
                    oimg[j, i] = self.test_img[x1 + pos[1], x1 + pos[0]]
                except IndexError:
                    pass

        # Vignetting Process
        mask = self._get_vignette_mask(c_lenslet_pos)
        oimg = oimg * mask

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
                c_pos = self._index_to_c_pos(i, j)
                dimg = self.dimg(c_pos)
                output[j, i] = dimg

        return output

    def all_dimg_new(self):
        """
        Generates the distorted image for every lenslet in the WFS
        :return: image ndarray
        """
        dimgufunc = np.frompyfunc(self.dimg_new,2,1)
        # Convert
        output = dimgufunc(self.c_pos_map[0,:,:],self.c_pos_map[1,:,:])

        return output

    def dmap_new(self, x,y):
        """
        Generates distortion map --- the (x,y) shift for every pixel in lenslet image

        Context: In layer oriented MCAO, if a WFS is misconjugated from a screen, different pixels, each representing
         different directions, see different parts
         of the screen and hence is shifted by different amounts
        :param c_lenslet_pos: (x,y) position of conjugated lenslet # [meter tuple]
        :return: distortion map --- (x-shift matrix, y-shift matrix) # [meter ndarray]
        """
        # initialize output variables
        oXSlope = np.zeros((self.wfs.pixels_lenslet, self.wfs.pixels_lenslet))
        oYSlope = np.zeros((self.wfs.pixels_lenslet, self.wfs.pixels_lenslet))

        # convert detector pixel index to angles
        bias = (self.wfs.pixels_lenslet - 1) / 2.0  # [index]
        for i in range(self.wfs.pixels_lenslet):
            for j in range(self.wfs.pixels_lenslet):
                # physical position of detector pixel
                pos_x = (i - bias) * self.wfs.conjugated_delta / self.tel.Mfactor  # [meters]
                pos_y = (j - bias) * self.wfs.conjugated_delta / self.tel.Mfactor  # [meters]

                # convert physical position to physical angle (incident on physical lenslet)
                tan_x = pos_x / float(self.wfs.lenslet_f)
                tan_y = pos_y / float(self.wfs.lenslet_f)

                # convert physical angle to conjugate angle (incident on conjugate lenslet)
                theta_x = np.arctan(tan_x / self.tel.Mfactor)
                theta_y = np.arctan(tan_y / self.tel.Mfactor)

                # stack metapupils and slopify
                screen = self._stack_lensletscreen((theta_x, theta_y), (x,y))
                (x_shift, y_shift) = self._screen_to_shifts(screen)

                oXSlope[j, i] = x_shift
                oYSlope[j, i] = y_shift

        return np.array([oXSlope, oYSlope])

    def dimg_new(self, x,y):
        """
        Generates the distorted image behind the given SH lenslet
        :param c_lenslet_pos: conjugated position of SH lenslet # [meter tuple]
        :return: distorted image # [ndarray]
        """
        # Call dmap
        distortion = self.dmap_new(x,y)

        x_distortion = distortion[0]
        y_distortion = distortion[1]

        # Crop test image
        assert x_distortion.shape[0] < self.test_img.shape[0]  # TODO: implement adaptive resize?
        x1 = self.test_img.shape[0] / 2 - x_distortion.shape[0] / 2
        oimg = np.zeros((self.wfs.pixels_lenslet, self.wfs.pixels_lenslet))

        # Distortion Process
        scale = 50  # fudge factor

        shift = lambda (x, y): (x + int(scale * x_distortion[x, y]), y + int(scale * y_distortion[x, y]))

        for j in range(oimg.shape[0]):
            for i in range(oimg.shape[1]):
                try:
                    pos = shift((i, j))
                    oimg[j, i] = self.test_img[x1 + pos[1], x1 + pos[0]]
                except IndexError:
                    pass

        # Vignetting Process
        mask = self._get_vignette_mask((x,y))
        oimg = oimg * mask

        return oimg

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


class ImageInterpreter(object):
    """
    ImageInterpreter takes the subimages produced behind the SH lenslets and tries to extract the slope
    information above each lenslet.
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
        # TODO
        # find reference image
        mid = all_dimg.shape[0] / 2
        imgRef = all_dimg[mid, mid]

    def _SquaredDifferenceFunction(self, img, imgRef, xShift, yShift):
        assert img.shape[0] < imgRef.shape[0]

        # find the starting corner of imgRef
        x1 = imgRef.shape[0] / 2.0 - img.shape[0] / 2.0 + int(xShift)
        y1 = imgRef.shape[1] / 2.0 - img.shape[1] / 2.0 + int(yShift)

        diff = (img - imgRef[y1:y1 + img.shape[1], x1:x1 + img.shape[0]]) ** 2

        return np.sum(np.sum(diff))

    def SDF(self, img, imgRef, shiftLimit):

        c_matrix = np.zeros((2 * shiftLimit + 1, 2 * shiftLimit + 1))

        bias = c_matrix.shape[0] / 2

        for j in range(c_matrix.shape[0]):
            for i in range(c_matrix.shape[0]):
                c_matrix[j, i] = self._SquaredDifferenceFunction(img, imgRef, i - bias, j - bias)

        return c_matrix

    def c_to_s_matrix(self, c_matrix):
        min = np.unravel_index(c_matrix.argmin(), c_matrix.shape)
        print min

        if min[0] == 0 or min[0] == c_matrix.shape[0] \
                or min[1] == 0 or min[1] == c_matrix.shape[1]:
            raise RuntimeError("Minimum is found on an edge")

        s_matrix = c_matrix[min[0] - 1:min[0] + 2, min[1] - 1:min[1] + 2]

        return s_matrix

    def TwoDLeastSquare(self, s_matrix):
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

    def TwoDQuadratic(self, s_matrix):
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

    def spatialAverage(self, all_dimg):
        """
        Reconstruct truth image
        :param all_dimg:
        :return:
        """
        ave = np.zeros(all_dimg[0, 0].shape)
        for j in range(all_dimg.shape[0]):
            for i in range(all_dimg.shape[1]):
                ave = ave + all_dimg[j, i]

        ave = ave / (all_dimg.shape[0] * all_dimg.shape[1])
        return ave

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


class SHWFS_Demonstrator(object):
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
        all_dimg = wfs.ImgSimulator.all_dimg_new()

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
    def display_comparison(wfs):  ### TODO: fix broken methods
        all_dmap = wfs.ImgSimulator.all_dmap()
        slopes = wfs.ImgInterpreter.all_dmap_to_slopes(all_dmap)
        screen = wfs._get_metascreen(wfs.atmos.scrns[0])
        sensed = wfs._reconstruct_WF(slopes)

        plt.figure(1)

        ax1 = plt.subplot(131)
        ax1.set_title("True phase scrn")
        im1 = ax1.imshow(screen)
        plt.colorbar(im1)

        ax2 = plt.subplot(132)
        ax2.set_title("Immediate")
        im2 = ax2.imshow(SHWFS_Demonstrator.immediatecutslopeNrecon(screen))
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
                oXSlope, oYSlope = _SlopifyMethods.slopify1(phasescreen[j * num:(j + 1) * num, i * num:(i + 1) * num])
                slopes[0, j, i] = oXSlope
                slopes[1, j, i] = oYSlope

        return ReconMethods.LeastSquare(slopes[0], slopes[1])

    @staticmethod
    def immediate(phasescreen, N=16):
        num = phasescreen.shape[0] / N
        slopes = np.empty((2, N, N))
        for j in range(N):
            for i in range(N):
                oXSlope, oYSlope = _SlopifyMethods.slopify1(phasescreen[j * num:(j + 1) * num, i * num:(i + 1) * num])
                slopes[0, j, i] = oXSlope
                slopes[1, j, i] = oYSlope
        return slopes


class _SlopifyMethods(object):
    """
    This class is a collection of slopify methods we experimented with.
    By default, use slopify1
    """

    @staticmethod
    def slopify1(screen):
        # Method 1 - Mean end to end difference
        S = screen.shape[0]

        x_tilt_acc = (screen[:, S - 1] - screen[:, 0]).sum()
        y_tilt_acc = (screen[S - 1, :] - screen[0, :]).sum()
        oXSlope = x_tilt_acc / (S * S)
        oYSlope = y_tilt_acc / (S * S)

        slope = (oXSlope, oYSlope)
        return slope

    @staticmethod
    def slopify2(screen):
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
    def slopify3(screen):
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


if __name__ == '__main__':
    tel = Telescope(2.5)
    at = Atmosphere()
    at.create_default_screen(100, 2)
    wfs = WideFieldSHWFS(5000, 16, 128, at, tel)
    cProfile.run('SHWFS_Demonstrator.display_all_dimg(wfs)')
    SHWFS_Demonstrator.display_all_dmap(wfs)
