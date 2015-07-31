import numpy as np
from scipy import stats
from scipy.misc import lena
from Reconstructor import ReconMethods
import matplotlib.pyplot as plt
from Atmosphere import *
from Telescope import Telescope
import pickle


class WideFieldSHWFS(object):
    """
    Usage:
        Methods prefixed with "_" are internal functions for the inner workings of the WFS
        Methods prefixed with "display" are for user interactions
        Methods prefixed with "run" are used by external classes

    """
    def __init__(self,height, num_lenslet, pixels_lenslet, atmosphere,telescope):
        """
        :param height: Conjugated height of the WFS [meters]
        :param num_lenslet: Number of lenslet spanning one side of pupil [int]
        :param pixels_lenslet: Number of detector pixels per lenslet [int]
        :param delta: Size of detector pixel [meters]
        :param atmosphere: Atmosphere object
        """

        # WFS information
        self.conjugated_height = height # [meters]
        self.num_lenslet = num_lenslet # [int]
        self.pixels_lenslet = pixels_lenslet # [int]

        # physical lenslet information
        # For physical sanity check: Don't really need these information for the simulation
        self.delta = telescope.pupil_diameter/float(num_lenslet)/float(pixels_lenslet)/float(telescope.Mfactor) # [meters]
        self.lenslet_size = self.pixels_lenslet * self.delta # [meters]
        self.lenslet_f = self.delta*self.pixels_lenslet/2.0/telescope.Mfactor/np.tan(telescope.field_of_view/2.0) # [meters]

        # conjugated lenslet information
        self.conjugated_delta = telescope.pupil_diameter/float(num_lenslet)/float(pixels_lenslet) # [meters]
        self.conjugated_lenslet_size = self.pixels_lenslet*self.conjugated_delta # [meters]

        # relevant objects
        self.atmos = atmosphere
        self.tel = telescope
        self.ImgSimulator = ImageSimulator(atmosphere,telescope,self)

    def _reconstruct_WF(self, slopes):
        """
        Reconstruct the WF surface from the slopes sensed by WFS.

        Usage: Many reconstruction algorithm exists. They have been packaged in a class.
         Change the choice of algorithm under the wrapper of this function
        :param slopes: (x-slope ndarray, y-slope ndarray) # [radian/meter ndarray]
        :return: # [radian ndarray]
        """
        # slopes = self._sense_slopes()
        surface = ReconMethods.LeastSquare(slopes[0],slopes[1])
        return surface

    def _get_metascreen(self,scrn):
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
            meta_radius = radius + (scrn.height-self.conjugated_height)*np.tan(theta)
        elif scrn.height > self.conjugated_height/2.0 :
            meta_radius = radius + (self.conjugated_height - scrn.height)*np.tan(theta)
        else:
            meta_radius = radius + scrn.height*np.tan(theta)

        # convert to array indices
        x_mid = scrn.phase_screen.shape[0]/2.0 # [index]
        y_mid = scrn.phase_screen.shape[1]/2.0 # [index]

        # convert lenslets size to phase screen indices; constant for all directions
        sizeX = int(meta_radius/scrn.delta) # [index]
        sizeY = int(meta_radius/scrn.delta) # [index]

        # frame to capture
        x1 = int(x_mid)-sizeX
        x2 = int(x_mid)+sizeX
        y1 = int(y_mid)-sizeY
        y2 = int(y_mid)+sizeY

        return scrn.phase_screen[y1:y2,x1:x2]

    def display_dmap(self,lenslet_pos):

        dist_map = self.ImgSimulator.dmap(lenslet_pos)

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

    def display_dimg(self,c_lenslet_pos):

        cropped_lena = self.ImgSimulator.get_test_img()
        oimg = self.ImgSimulator.dimg(c_lenslet_pos)

        plt.figure(1)

        ax1 = plt.subplot(121)
        ax1.set_title("True Image")
        plt.imshow(cropped_lena,cmap=plt.cm.gray)

        ax2 = plt.subplot(122)
        ax2.set_title("Distorted Image")
        plt.imshow(oimg,cmap=plt.cm.gray)

        plt.show()

    def display_all_dmap(self, axis=0):
        # sanity check
        if axis != 0 and axis != 1:
            raise ValueError("\nContext: Displaying all distortion maps\n" +
                             "Problem: Choice of axes (x,y) invalid\n" +
                             "Solution: Input 'axis' argument should be 0 (x-axis) or 1 (y-axis)")

        all_dmaps = self.ImgSimulator.all_dmap()

        # Display process
        plt.figure(1)
        for i in range(self.num_lenslet):
            for j in range(self.num_lenslet):
                plt.subplot(self.num_lenslet,self.num_lenslet,i*self.num_lenslet+j+1)
                plt.axis('off')
                plt.imshow(all_dmaps[j][i][axis])

        plt.show()

    def display_all_dimg(self):
        all_dimg = self.ImgSimulator.all_dimg()

        # Display process
        plt.figure(1)
        # Iterate over lenslet index
        for i in range(self.num_lenslet):
            for j in range(self.num_lenslet):
                plt.subplot(self.num_lenslet,self.num_lenslet,i*self.num_lenslet+j+1)
                plt.axis('off')
                plt.imshow(all_dimg[j][i],cmap=plt.cm.gray,vmax=256,vmin=0)

        plt.show()

    def display_comparison(self):
        distmap_all = self.ImgSimulator._all_dmap()
        slopes = self._sense_slopes(distmap_all)
        sensed = self._reconstruct_WF(slopes)
        screen = self._get_metascreen(self.atmos.scrns[0])

        plt.figure(1)
        plt.subplot(121)
        plt.imshow(screen)
        plt.colorbar()
        plt.subplot(122)
        plt.imshow(sensed)
        plt.colorbar()
        # plt.axis('off')
        plt.show()

    def display_slopes(self):
        screen = self._get_metascreen(self.atmos.scrns[0])
        distmap_all = self._all_dmap()
        slopes = self._sense_slopes(distmap_all)
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

    def display_angles(self):
        x = []
        y = []
        for j in range(self.pixels_lenslet):
            for i in range(self.pixels_lenslet):
                angle = self._pixel_to_angle(i,j)
                x.append(angle[0])
                y.append(angle[1])
        plt.scatter(x,y)
        plt.show()

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

        self.test_img = lena()

    def _get_lensletscreen(self,scrn, angle,c_lenslet_pos):
        """
        Portion of a screen seen by a lenslet in one particular direction
        :param scrn: Screen object
        :param angle: (x,y) angle of view from pupil # [radian tuple]
        :param c_lenslet_pos: (x,y) position of conjugated lenslet # [float tuple]
        """
        theta_x = angle[0] # [radian]
        theta_y = angle[1] # [radian]

        c_lenslet_pos_x = c_lenslet_pos[0] # [meters]
        c_lenslet_pos_y = c_lenslet_pos[1] # [meters]

        # finding center of meta pupil
        pos_x = c_lenslet_pos_x + np.tan(theta_x)*(self.wfs.conjugated_height-scrn.height) # [meters]
        pos_y = c_lenslet_pos_y + np.tan(theta_y)*(self.wfs.conjugated_height-scrn.height) # [meters]

        # convert to array indices
        x_mid = scrn.phase_screen.shape[0]/2.0 + pos_x/float(scrn.delta) # [index]
        y_mid = scrn.phase_screen.shape[1]/2.0 + pos_y/float(scrn.delta) # [index]

        # convert lenslets size to phase screen indices; constant for all directions
        sizeX = int(self.wfs.pixels_lenslet/2.0*self.wfs.conjugated_delta/scrn.delta) # [index]
        sizeY = int(self.wfs.pixels_lenslet/2.0*self.wfs.conjugated_delta/scrn.delta) # [index]

        # frame to capture
        x1 = int(x_mid)-sizeX
        x2 = int(x_mid)+sizeX
        y1 = int(y_mid)-sizeY
        y2 = int(y_mid)+sizeY

        # sanity check
        #TODO remove this check by automatically doubling the phase screen size
        assert x1 > 0
        assert x2 < scrn.phase_screen.shape[0]
        assert y1 > 0
        assert y2 < scrn.phase_screen.shape[1]

        # grab a snapshot the size of a lenslet
        output = scrn.phase_screen[x1:x2,y1:y2]

        return np.copy(output)

    def _stack_lensletscreen(self, angle,c_lenslet_pos):
        """
        Stacks the portions of all the screens seen by a lenslet in a particular direction
        :param angle: (x,y) angle of view from pupil # [radian tuple]
        :param c_lenslet_pos: (x,y) position of conjugated lenslet # [float tuple]
        :return: Net phase screen seen by the lenslet # [radian ndarray]
        """
        # sanity check: that lenslet_f is implemented correctly
        assert angle[0] < self.tel.field_of_view/2.0
        assert angle[1] < self.tel.field_of_view/2.0

        # initialize output variable
        outphase = None

        # Assuming that atmos.scrns is reverse sorted in height
        for scrn in self.atmos.scrns:
            change = self._get_lensletscreen(scrn,angle,c_lenslet_pos)
            if outphase == None:
                outphase = change
            else:
                outphase = outphase + change

        return outphase

    def _screen_to_shifts(self,stacked_phase):
        """
        Slopifies the net phase screen.

        Assumption: the tip-tilt distortion is more significant than other modes of optical abberation.
         This requires fried radius to be about just as large as conjugated lenslet size
        Usage: The various possible algorithms used to slopify are stored in  a class. User can change
         the choice of algorithm under the wrapper of this function
        :param stacked_phase: Net phase screen seen by the lenslet # [radian ndarray]
        :return: (x_shift,y_shift) in conjugate image plane # [meter]
        """

        (x_shift,y_shift) = SlopifyMethods.slopify1(stacked_phase)
        #TODO: manage the units of the shifts in image plane
        # TODO: from radians/meter to meter
        return (x_shift,y_shift)

    def _pixel_to_angle(self,i,j):
        """
        Utility function to convert pixel index position to angle of incident ray
        :param i: horizontal index of pixel
        :param j: vertical index of pixel
        :return: angle of ray incident on conjugated lenslet
        """
        bias = (self.wfs.pixels_lenslet-1)/2.0 # [index]
        # physical position of detector pixel
        pos_x = (i-bias)*self.wfs.conjugated_delta/self.tel.Mfactor # [meters]
        pos_y = (j-bias)*self.wfs.conjugated_delta/self.tel.Mfactor # [meters]

        # convert physical position to physical angle (incident on physical lenslet)
        tan_x = pos_x/float(self.wfs.lenslet_f)
        tan_y = pos_y/float(self.wfs.lenslet_f)

        # convert physical angle to conjugate angle (incident on conjugate lenslet)
        theta_x = np.arctan(tan_x/self.tel.Mfactor)
        theta_y = np.arctan(tan_y/self.tel.Mfactor)

        return (theta_x,theta_y)

    def _index_to_c_pos(self,i,j):
        bias = (self.wfs.num_lenslet-1)/2.0 # [index]
        # conjugated position of lenslet
        pos_x = (i-bias)*self.wfs.conjugated_lenslet_size # [meters]
        pos_y = (j-bias)*self.wfs.conjugated_lenslet_size # [meters]

        lenslet_pos = (pos_x,pos_y)

        return lenslet_pos

    def _vignettify(self,c_lenslet_pos, angle):
        # find z
        theta_x = angle[0]
        theta_y = angle[1]

        # x = np.tan(theta_x) * self.conjugated_height + c_lenslet_pos[0] # figure out the +/- later
        # y = np.tan(theta_y) * self.conjugated_height + c_lenslet_pos[1] # figure out the +/- later

        x = c_lenslet_pos[0] - np.tan(theta_x) * self.wfs.conjugated_height# figure out the +/- later
        y = c_lenslet_pos[1] - np.tan(theta_y) * self.wfs.conjugated_height# figure out the +/- later

        # vignetting algorithm
        z = np.sqrt(x**2 + y**2)
        R = self.tel.pupil_diameter/2.0
        r = self.wfs.conjugated_lenslet_size

        if z < R-r:
            p = 1
        elif z > (R+r):
            p = 0
        elif z > R:
            s = (R+r+z)/2.0 # semiperimeter
            area = np.sqrt((s)*(s-r)*(s-z)*(s-R)) # Heron formula
            theta_R = np.arccos((R**2 + z**2 - r**2)/(2*R*z))
            theta_r = np.arccos((r**2 + z**2 - R**2)/(2*r*z))
            hat = 2 * ((0.5*theta_R*R**2) + (0.5*theta_r*r**2) - area)
            p = hat / (np.pi*r**2)

        else:
            theta_R = 2 * np.arccos((R**2 + z**2 - r**2) / (2*R*z))
            theta_r = 2 * np.arcsin(np.sin(theta_R/2.0)*R/r)
            tri = 0.5  * R**2 * np.sin(theta_R)
            cap = 0.5*r**2*theta_r - 0.5*r**2*np.sin(theta_r)
            cres = tri+cap - 0.5*R**2*theta_R
            p = 1 - (cres / (np.pi*r**2))

        return p

    def _get_vignette_mask(self, c_lenslet_pos):
        mask = np.ones((self.wfs.pixels_lenslet,self.wfs.pixels_lenslet))
        for j in range(self.wfs.pixels_lenslet):
            for i in range(self.wfs.pixels_lenslet):
                angle = self._pixel_to_angle(i,j)
                theta_x = angle[0]
                theta_y = angle[1]
                p = self._vignettify(c_lenslet_pos,(theta_x,theta_y))
                mask[j][i] = p
        return mask

    def dmap(self,c_lenslet_pos):
        """
        Generates distortion map --- the (x,y) shift for every pixel in lenslet image

        Context: In layer oriented MCAO, if a WFS is misconjugated from a screen, different pixels, each representing
         different directions, see different parts
         of the screen and hence is shifted by different amounts
        :param c_lenslet_pos: (x,y) position of conjugated lenslet # [float tuple]
        :return: distortion map --- (x-shift matrix, y-shift matrix) # [meter ndarray]
        """
        # initialize output variables
        oXSlope = np.zeros((self.wfs.pixels_lenslet,self.wfs.pixels_lenslet))
        oYSlope = np.zeros((self.wfs.pixels_lenslet,self.wfs.pixels_lenslet))

        # convert detector pixel index to angles
        bias = (self.wfs.pixels_lenslet-1)/2.0 # [index]
        for i in range(self.wfs.pixels_lenslet):
            for j in range(self.wfs.pixels_lenslet):

                # physical position of detector pixel
                pos_x = (i-bias)*self.wfs.conjugated_delta/self.tel.Mfactor # [meters]
                pos_y = (j-bias)*self.wfs.conjugated_delta/self.tel.Mfactor # [meters]

                # convert physical position to physical angle (incident on physical lenslet)
                tan_x = pos_x/float(self.wfs.lenslet_f)
                tan_y = pos_y/float(self.wfs.lenslet_f)

                # convert physical angle to conjugate angle (incident on conjugate lenslet)
                theta_x = np.arctan(tan_x/self.tel.Mfactor)
                theta_y = np.arctan(tan_y/self.tel.Mfactor)

                # stack metapupils and slopify
                screen = self._stack_lensletscreen((theta_x,theta_y),c_lenslet_pos)
                (x_shift,y_shift) = self._screen_to_shifts(screen)

                oXSlope[j,i] = x_shift
                oYSlope[j,i] = y_shift

        return np.array([oXSlope,oYSlope])
    
    def dimg(self,c_lenslet_pos):
        """
        Generates the distorted image behind the specified SH lenslet
        :param c_lenslet_pos:
        :return:
        """
        # Call dmap
        distortion = self.dmap(c_lenslet_pos)

        x_distortion = distortion[0]
        y_distortion = distortion[1]

        # Crop test image
        assert x_distortion.shape[0]<self.test_img.shape[0] # implement adaptive resize?
        x1 = self.test_img.shape[0]/2 - x_distortion.shape[0]/2
        oimg = np.zeros((self.wfs.pixels_lenslet,self.wfs.pixels_lenslet))
        
        # Distortion Process
        scale = 50 # fudge factor

        shift = lambda (x,y) : (x+int(scale*x_distortion[x,y]),y+int(scale*y_distortion[x,y]))

        for j in range(oimg.shape[0]):
            for i in range(oimg.shape[1]):
                try:
                    pos = shift((i,j))
                    oimg[j,i] = self.test_img[x1+pos[1], x1+pos[0]]
                except IndexError:
                    pass

        # Vignetting Process
        mask = self._get_vignette_mask(c_lenslet_pos)
        oimg = oimg * mask

        return oimg

    def save_dimg(self, c_lenslet_pos):
        """
        a wrapper that saves the dimg
        :param c_lenslet_pos:
        :return:
        """
        dimg = self.dimg(c_lenslet_pos)
        f = open("TestDimg"+".dimg",'wb')
        pickle.dump(dimg,f,pickle.HIGHEST_PROTOCOL)
        return dimg

    def save_all_dimg(self):
        all_dimg = self.all_dimg()
        f = open("TestAllDimg"+".dimg",'wb')
        pickle.dump(all_dimg,f,pickle.HIGHEST_PROTOCOL)
        return all_dimg

    def all_dmap(self):
        """
        Generates the (x,y) shift for every pixel of lenslet image for all lenslets
        :return: x-shifts y-shifts # [meters ndarray]
        """
        # Initialize output array
        output = np.empty(self.wfs.num_lenslet,np.ndarray)

        # Iterate over lenslet index
        for j in range(self.wfs.num_lenslet):
            line = np.empty(self.wfs.num_lenslet,np.ndarray)
            for i in range(self.wfs.num_lenslet):
                print (i,j)
                c_lenslet_pos = self._index_to_c_pos(i,j)
                distortion = self.dmap(c_lenslet_pos)
                line[i] = distortion
            output[j] = line

        return output

    def all_dimg(self):
        """
        Generates the distorted image for every lenslet in the WFS
        :return: image ndarray
        """
        # Initialize output array
        output = np.empty((self.wfs.num_lenslet,self.wfs.num_lenslet),np.ndarray)

        # Convert
        for j in range(self.wfs.num_lenslet):
            for i in range(self.wfs.num_lenslet):
                c_pos = self._index_to_c_pos(i,j)
                dimg = self.dimg(c_pos)
                output[j,i] = dimg

        return output

    def get_test_img(self):
        """
        Returns the portion of standard test image used to represent dmap
        :return:
        """

        assert self.wfs.pixels_lenslet <= self.test_img.shape[0]

        # Crop standard test image
        x1 = self.test_img.shape[0]/2 - self.wfs.pixels_lenslet/2.0
        x2 = self.test_img.shape[0]/2 + self.wfs.pixels_lenslet/2.0
        cropped_img = self.test_img[x1:x2,x1:x2]

        return cropped_img

    def display_vignette(self, c_lenslet_pos):
        """
        Demonstrates the vigenetting effect for a single lenslet
        :param c_lenslet_pos: 
        :return:
        """

        img = self._get_vignette_mask(c_lenslet_pos)
        
        plt.imshow(img,cmap=plt.cm.gray)
        plt.colorbar()
        plt.show()
    
    def display_all_vignette(self):
        plt.figure(1)
        for j in range(self.wfs.num_lenslet):
            for i in range(self.wfs.num_lenslet):
                c_lenslet_pos = self._index_to_c_pos(i,j)
                img = self._get_vignette_mask(c_lenslet_pos)
                plt.subplot(self.wfs.num_lenslet,self.wfs.num_lenslet,i+j*self.wfs.num_lenslet + 1)
                plt.axis('off')
                plt.imshow(img, cmap=plt.cm.gray, vmin = 0, vmax=1)
        plt.show()

    def _stupid_hash_function(self):
        hash = str(self.wfs.conjugated_height) + str(self.wfs.num_lenslet) + str(self.wfs.pixels_lenslet)
        for scrn in self.atmos.scrns:
            hash = hash + str(scrn.height) + str(scrn.ID)
        return hash

class ImageInterpreter(object):
    """
    ImageInterpreter takes the subimages produced behind the SH lenslets and tries to extract the slope
    information above each lenslet.
    """
    def _dmap_to_slope(self,distortion_map):
        """
        Assumption: There is a common shift component to all pixels of a distortion map
         This component comes from the phase screen that the lenslet is conjugated to.
         Misconjugated screens should have an average shift contribution close to zero
        :param distortion_map: (x,y) shift used to distort image. Matrix shape should be (2,N,N) # [meter ndarray]
        :return: slopes # [radian per mater]
        """
        #TODO: manage the units

        assert distortion_map.shape[0] == 2
        return (distortion_map[0].mean(),distortion_map[1].mean())

    def all_dmap_to_slopes(self,d_map_array):
        """
        Generate the net WF slopes sensed by WFS. Each lenslet acts as one gradient (slope)
        sensor
        :param d_map_array: 2D list of distortion maps # [meters ndarray list list]
        :return: (x-slopes, y-slopes) # [radian/meter ndarray]
        """
        (ySize,xSize) = d_map_array.shape
        slopes = np.zeros((2,ySize,xSize))

        for (j,line) in enumerate(d_map_array):
            for (i,dmap) in enumerate(line):
                (x,y) = ImageInterpreter._dmap_to_slope(dmap)
                slopes[0,j,i] = x
                slopes[1,j,i] = y

        return slopes

    def all_dimg_to_slopes(self,all_dimg):
        # find reference image
        mid = all_dimg.shape[0]/2
        imgRef = all_dimg[mid,mid]

    def _SquaredDifferenceFunction(self,img, imgRef, xShift,yShift):
        assert img.shape[0]<imgRef.shape[0]

        # find the starting corner of imgRef
        x1 = imgRef.shape[0]/2.0 - img.shape[0]/2.0 + int(xShift)
        y1 = imgRef.shape[1]/2.0 - img.shape[1]/2.0 + int(yShift)

        diff = (img-imgRef[y1:y1+img.shape[1],x1:x1+img.shape[0]])**2

        return np.sum(np.sum(diff))

    def SDF(self,img, imgRef, shiftLimit):

        c_matrix = np.zeros((2*shiftLimit+1,2*shiftLimit+1))

        bias = c_matrix.shape[0]/2

        for j in range(c_matrix.shape[0]):
            for i in range(c_matrix.shape[0]):
                c_matrix[j,i] = _SquaredDifferenceFunction(img,imgRef,i-bias,j-bias)

        return c_matrix

    def c_to_s_matrix (self,c_matrix):
        min = np.unravel_index(c_matrix.argmin(),c_matrix.shape)
        print min

        if min[0] == 0 or min[0] == c_matrix.shape[0] \
            or min[1] == 0 or min[1] == c_matrix.shape[1]:
            raise RuntimeError("Minimum is found on an edge")

        s_matrix = c_matrix[min[0]-1:min[0]+2,min[1]-1:min[1]+2]

        return s_matrix

    def TwoDLeastSquare(self,s_matrix):
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

    def TwoDQuadratic(self,s_matrix):
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

    def spatialAverage(self,all_dimg):
        ave = np.zeros(all_dimg[0,0].shape)
        for j in range(all_dimg.shape[0]):
            for i in range(all_dimg.shape[1]):
                ave = ave + all_dimg[j,i]

        ave = ave / (all_dimg.shape[0]*all_dimg.shape[1])
        return ave

    def compare_refImg(self,dimg, recon_dimg):
        plt.figure(1)
        plt.subplot(2,1,1)
        plt.imshow(dimg)
        plt.subplot(2,1,2)
        plt.imshow(recon_dimg)
        plt.show()



class SlopifyMethods(object):
    @staticmethod
    def slopify1(screen):

        # Method 1 - Mean end to end slope
        x_tilt_acc = 0.0
        y_tilt_acc = 0.0
        S = screen.shape[0]
        for k in range(S):
            x_tilt_acc = x_tilt_acc + (screen[k,S-1] - screen[k,0])
            y_tilt_acc = y_tilt_acc + (screen[S-1,k] - screen[0,k])
        oXSlope = float(x_tilt_acc) / (S*S)
        oYSlope= float(y_tilt_acc) / (S*S)

        slope = (oXSlope,oYSlope)
        return slope
    def slopify2(cls,screen):
        # TODO: remove hardcode
        oXSlope = np.zeros(screen.shape)
        oYSlope = np.zeros(screen.shape)

        # Method 2 - Linear regression of mean
        for j in range(16):
            sumX = np.sum(screen[32*j:32*(j+1)-1,:], axis=0)
            for i in range(16):
                slopeX, interceptX, r_valueX, p_valueX, std_errX = stats.linregress(np.arange(32), sumX[32*i:32*(i+1)])
                oXSlope[j*32:(j+1)*32,i*32:(i+1)*32] = slopeX /32.0
        for i in range(16):
            sumY = np.sum(screen[:,32*i:32*(i+1)-1], axis=1)
            for j in range(16):
                slopeY, interceptY, r_valueY, p_valueY, std_errY = stats.linregress(np.arange(32), sumY[32*j:32*(j+1)])
                oYSlope[j*32:(j+1)*32,i*32:(i+1)*32] = slopeY / 32.0
        slope = np.array([oXSlope,oYSlope])
        return slope
    def slopify3(cls,screen):
        # TODO: remove hardcode
        oXSlope = np.zeros(screen.shape)
        oYSlope = np.zeros(screen.shape)

        # Method 3 - mean of linear regression
        for i in range(16):
            for j in range(16):
                x_slope_acc = 0.0
                y_slope_acc = 0.0
                for k in range(32):
                    slopeX, interceptX, r_valueX, p_valueX, std_errX = \
                        stats.linregress(np.arange(32), screen[32*j+k,32*i:32*(i+1)])
                    x_slope_acc = x_slope_acc + slopeX
                    slopeY, interceptY, r_valueY, p_valueY, std_errY = \
                        stats.linregress(np.arange(32), screen[32*j:32*(j+1),32*i+k])
                    y_slope_acc = y_slope_acc + slopeY

                oXSlope[j*32:(j+1)*32,i*32:(i+1)*32] = x_slope_acc / 32.0
                oYSlope[j*32:(j+1)*32,i*32:(i+1)*32] = y_slope_acc / 32.0
        slope = np.array([oXSlope,oYSlope])
        return slope

if __name__ == '__main__':
    tel = Telescope(2.5)
    at = Atmosphere()
    at.create_default_screen(0,200)
    wfs = WideFieldSHWFS(100,16,32,at,tel)
    # wfs.ImgSimulator.save_all_dimg()
    # wfs.ImgSimulator.save_dimg((0,0))
    da = pickle.load(open("TestAllDimg.dimg",'rb'))
    dd = pickle.load(open("TestDimg.dimg",'rb'))
    ave = ImageInterpreter.spatialAverage(da)
    ImageInterpreter.compare_refImg(dd,ave)
    # wfs.display_all_dimg()