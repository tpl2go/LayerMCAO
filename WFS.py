import numpy as np
from Reconstructor import ReconMethods
import matplotlib.pyplot as plt
from Atmosphere import *
from Telescope import Telescope
from ImageInterpreter import ImageInterpreter
from ImageSimulator import ImageSimulator, _TiltifyMethods
import cProfile
import scipy.io as sio


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

    def _reconstruct_WF(self, shifts):
        """
        Reconstruct the WF surface from the slopes sensed by WFS.

        Usage: Many reconstruction algorithm exists. They have been packaged in a class.
         Change the choice of algorithm under the wrapper of this function
        :param slopes: (x-slope ndarray, y-slope ndarray) # [radian/meter ndarray]
        :return: # [radian ndarray]
        """
        tilts = self._shifts_to_tilts(shifts)
        slopes = self._tilts_to_gradient(tilts)
        surface = ReconMethods.LeastSquare(slopes[0], slopes[1])
        return surface

    def _shifts_to_tilts(self,shifts):
        # Linear approximation
        return shifts*self.angular_res

    def _tilts_to_gradient(self,tilts):
        return np.tan(tilts)

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
        all_shifts = self.ImgInterpreter.all_dimg_to_shifts(all_dimg)

        return all_shifts

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

        # Find the min and max
        smin = all_dmaps[0,0][axis].min()
        smax = all_dmaps[0,0][axis].max()
        for i in range(wfs.num_lenslet):
            for j in range(wfs.num_lenslet):
                if smin > all_dmaps[j,i][axis].min():
                    smin = all_dmaps[j,i][axis].min()
                if smax < all_dmaps[j,i][axis].max():
                    smax = all_dmaps[j,i][axis].max()

        # Display process
        fig = plt.figure(1)
        for i in range(wfs.num_lenslet):
            for j in range(wfs.num_lenslet):
                plt.subplot(wfs.num_lenslet, wfs.num_lenslet, i * wfs.num_lenslet + j + 1)
                plt.axis('off')
                im = plt.imshow(all_dmaps[j, i][axis], vmin=smin, vmax=smax)
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
    def actual_shifts_vs_measured_shifts(wfs):
        all_dmaps = wfs.ImgSimulator.all_dmap()
        all_shifts = wfs.runWFS()

        # TODO: All_shifts is still broken !!!
        # Display process
        fig1 = plt.figure(1)
        
        # Find the min and max
        # TODO: vectorize this operation
        asmin = all_dmaps[0,0][0].min()
        asmax = all_dmaps[0,0][0].max()
        for i in range(wfs.num_lenslet):
            for j in range(wfs.num_lenslet):
                if asmin > all_dmaps[j,i][0].min():
                    asmin = all_dmaps[j,i][0].min()
                if asmax < all_dmaps[j,i][0].max():
                    asmax = all_dmaps[j,i][0].max()

        plt.subplot(1,2,1)
        for j in range(wfs.num_lenslet):
            for i in range(wfs.num_lenslet):
                plt.subplot(wfs.num_lenslet, 2 * wfs.num_lenslet, j * 2 * wfs.num_lenslet + i + 1)
                plt.axis('off')
                plt.imshow(all_dmaps[j, i][0], vmin=asmin, vmax=asmax)

        # Finding min and max of measured shifts
        # msmin = all_shifts[0,:,:].min()
        # msmax = all_shifts[0,:,:].max()

        plt.subplot(1,2,2)
        plt.axis('off')
        Vtake = np.vectorize(np.take)
        im = Vtake(all_shifts,[0],axis=0) # take the x component of the shift
        # plt.imshow(im, vmin=-1, vmax=1, interpolation='None')
        plt.imshow(im, interpolation='None')

        plt.show()

    @staticmethod
    def dmap_vs_measured_shift(wfs,c_pos):
        dmap = wfs.ImgSimulator.dmap(c_pos)
        dimg = wfs.ImgSimulator.dimg(c_pos)
        refImg = wfs.ImgInterpreter.get_ref_img()
        shift = wfs.ImgInterpreter._measure_dimg_shifts(dimg, refImg)
        realGlobalShift = dmap[0].mean()

        print "Appied Shift = " + str(realGlobalShift)
        print "Measured Shift = " + str(shift)

    @staticmethod
    def display_recon_N_screen(wfs):  ### TODO: fix broken methods
        slopes = wfs.runWFS()
        screen = wfs._get_metascreen(wfs.atmos.scrns[0])
        sensed = wfs._reconstruct_WF(slopes)
        print sensed
        plt.figure(1)

        ax1 = plt.subplot(131)
        ax1.set_title("True phase scrn")
        im1 = ax1.imshow(screen)
        plt.colorbar(im1)

        ax2 = plt.subplot(132)
        ax2.set_title("Immediate")
        img = SHWFSDemonstrator.immediatecutslopeNrecon(screen,wfs.num_lenslet,wfs.tel.wavelength, wfs.conjugated_lenslet_size)
        im2 = ax2.imshow(img)
        plt.colorbar(im2)

        ax3 = plt.subplot(133)
        ax3.set_title("Reconstructed")
        im3 = ax3.imshow(sensed)
        plt.colorbar(im3)

        plt.show()

    @staticmethod
    def display_metapupil_N_dmaps(wfs):
        all_dmaps = wfs.ImgSimulator.all_dmap()

        # Display process
        fig1 = plt.figure(1)
        plt.subplot(1,2,1)
        plt.imshow(wfs._get_meta_pupil(wfs.atmos.scrns[0]))

        for j in range(wfs.num_lenslet):
            for i in range(wfs.num_lenslet):
                plt.subplot(wfs.num_lenslet, 2 * wfs.num_lenslet, j * 2 * wfs.num_lenslet + i + 1 + wfs.num_lenslet)
                plt.axis('off')
                plt.imshow(all_dmaps[j, i][0], vmin=-10, vmax=10)

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
    def immediatecutslopeNrecon(phasescreen, num_lenslet, wavelength, conjugated_lenslet_size):
        num = phasescreen.shape[0] / num_lenslet
        slopes = np.empty((2, num_lenslet, num_lenslet))
        for j in range(num_lenslet):
            for i in range(num_lenslet):
                oXSlope, oYSlope = _TiltifyMethods.tiltify1(phasescreen[j * num:(j + 1) * num, i * num:(i + 1) * num],wavelength,conjugated_lenslet_size)
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


if __name__ == '__main__':
    tel = Telescope(2.5)
    at = Atmosphere()
    # at.create_screen(0.20, 2048, 0.01, 20, 0.01, 0)
    # at.save_screens()
    at.load_screen(0.20, 2048, 0.01, 20, 0.01, 0, 0)
    #
    wfs = WideFieldSHWFS(0, 16, 128, at, tel)
    # print wfs.conjugated_lenslet_size
    # print wfs.angular_res
    # SHWFSDemonstrator.actual_shifts_vs_measured_shifts(wfs)
    SHWFSDemonstrator.dmap_vs_measured_shift(wfs,(0,0))
    # SHWFSDemonstrator.display_all_dmap(wfs)
    # SHWFSDemonstrator.display_recon_N_screen(wfs)
    # surface = sio.loadmat('surface.mat')
    # surface = surface['surface']
    # plt.imshow(surface,interpolation='None')
    # plt.show()