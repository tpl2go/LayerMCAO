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
        self.conjugated_lenslet_size = self.conjugated_size / float(num_lenslet)  # [meters]

        # Lenslet attributes (Physical)
        ### For physical sanity check only. Not needed for simulation
        self.lenslet_size = self.conjugated_lenslet_size / float(telescope.Mfactor)  # [meters]

        # Detector attributes
        ### Angular resolution valid for small angles only
        self.angular_res = telescope.field_of_view / pixels_lenslet  # [rad/pixel]
        self.conjugated_delta = self.conjugated_size / float(num_lenslet) / float(pixels_lenslet)  # [meters]

        # Relevant objects
        self.atmos = atmosphere
        self.tel = telescope

        # Experimental algorithms
        ### Strategy pattern for custom ImageSimulation / ImageInterpretor algorithm
        self.ImgSimulator = ImageSimulator(atmosphere, telescope, self)
        self.ImgInterpreter = ImageInterpreter(self.ImgSimulator)

    def print_attributes(self):
        print "WFS ATTRIBUTES"
        print "Conjugated height:\t" + str(self.conjugated_height)
        print "Number of lenslets:\t" + str(self.num_lenslet)
        print "Number of pixels per lenslet:\t" + str(self.pixels_lenslet)
        print "Size of conjugated WFS:\t" + str(self.conjugated_size)

        print ""

        print "LENSLET ATTRIBUTES"
        print "Size of conjugate lenslet:\t" + str(self.conjugated_lenslet_size)
        print "Size of physical lenslet:\t" + str(self.lenslet_size)

        print ""

        print "DETECTOR ATTRIBUTES"
        print "Angular resolution of pixel:\t" + str(self.angular_res)
        print "Size of conjugate pixel:\t" + str(self.conjugated_delta)

    def _reconstruct_WF(self, all_shifts):
        # TODO: should this function accept shifts or tilts?
        # ans: if it accept tilts, this will be a static function
        """
        Reconstruct the WF surface from the shifts sensed by WFS.

        Usage: Many reconstruction algorithm exists. They have been packaged in a class.
         Change the choice of algorithm under the wrapper of this function
        :param all_shifts: (xShift, yShift) ndarray # [pixel]
        :return: # [radian ndarray]
        """
        tilts = self._shifts_to_tilts(all_shifts)
        slopes = self._tilts_to_gradient(tilts)

        # Extract xSlope and ySlope from nested ndarray
        Vtake = np.vectorize(np.take)
        xSlopes = Vtake(slopes, [0], axis=0)  # [radians/meter]
        ySlopes = Vtake(slopes, [1], axis=0)  # [radians/meter]
        surface = ReconMethods.LeastSquare(xSlopes, ySlopes)

        return surface

    def _shifts_to_tilts(self, shifts):
        # Linear approximation
        return shifts * self.angular_res

    def _tilts_to_gradient(self, tilts):
        """
        Converts tilts [radians] to gradient [radians/meter]
        Function is numpy-aware
        :param tilts: [radians]
        :return: gradient [radians/meter]
        """
        # numpy awareness
        if tilts.dtype == np.ndarray:
            Vtan = np.vectorize(np.tan, otypes=[np.ndarray])
        else:
            Vtan = np.tan

        return Vtan(tilts)

    def _get_metascreen(self, scrn):
        """
        Returns the portion of phase screen that WFS senses.

        Differs from metapupil when WFS not the same size as metapupil

        Refer to diagram to understand this implementation
        :param scrn: the Screen object whose metascreen we are finding
        :return: portion of screen # [radian ndarray]
        """

        # basis parameters
        FoV = self.tel.field_of_view
        theta = FoV / 2.0
        radius = self.conjugated_size / 2.0

        # find meta radius
        # meta radius: half length of metascreen

        if scrn.height > self.conjugated_height:
            meta_radius = radius + (scrn.height - self.conjugated_height) * np.tan(theta)
        else:
            threshold_height = (
                                   (radius - self.tel.pupil_diameter / 2.0) / np.tan(
                                       theta) + self.tel.pupil_diameter) / 2.0

            if scrn.height > threshold_height:
                meta_radius = radius + abs(self.conjugated_height - scrn.height) * np.tan(theta)
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
        """
        Returns the portion of phase screen in field of view of telescope.

        :param scrn: Screen object
        :return: sub phase screen [ndarray] [radian]
        """
        size_of_WFS = self.num_lenslet * self.pixels_lenslet * self.conjugated_delta
        num_scrn_pixels = size_of_WFS / scrn.delta
        shapeX, shapeY = scrn.phase_screen.shape
        x1 = int(shapeX / 2.0 - num_scrn_pixels / 2.0)

        return scrn.phase_screen[x1:x1 + num_scrn_pixels, x1:x1 + num_scrn_pixels]

    def set_size(self, c_size):
        """
        Sets size of WFS.
        WFS is centered in metapupil

        Context:
            By default the WFS size is set to that of the metapupil
        :param c_size: conjugated size of WFS
        """
        self.conjugated_size = c_size  # [meters]
        self.conjugated_lenslet_size = self.conjugated_size / float(self.num_lenslet)  # [meters]
        self.lenslet_size = self.conjugated_lenslet_size / float(self.tel.Mfactor)  # [meters]
        self.conjugated_delta = self.conjugated_size / float(self.num_lenslet) / float(self.pixels_lenslet)  # [meters]

    def runWFS(self):
        """
        Observes the atmosphere and sense the image shifts in all lenslets

        Design:
            The simulation of lenslet images and the deduction of image shift values from those images
            are delegated to two classes --- ImageSimulator and Image Interpreter. The only information
            passed between them should only be the lenslet images.

            The use of template and strategy patterns
            is intended to allow easy modifications to algorithms without disruption AO simulation setup.

            This function is intended to be the main function from which other AO components read WFS outputs
        """

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

    """ Display dmap/dimg Methods """

    @staticmethod
    def display_dmap(dmap):
        """
        Inspect the dmap used by wfs.ImgSimulator to generate a particular dimg

        Usage:
         c_lenslet_pos = (0.2,0.5)
         dmap = wfs.ImgSimulator.dmap(c_lenslet_pos)
         SHWFSDemonstrator.display_dmap(dmap)
        """

        plt.figure(1)

        ax1 = plt.subplot(121)
        ax1.set_title("X-Distortion")
        plt.imshow(dmap[0])
        plt.colorbar()

        ax2 = plt.subplot(122)
        ax2.set_title("Y-Distortion")
        plt.imshow(dmap[1])
        plt.colorbar()

        plt.show()

    @staticmethod
    def display_all_dmap(all_dmaps, axis=0):
        """
        Inspect the dmaps used by wfs.ImgSimulator to generate the dimgs

        Usage:
         all_dmaps = wfs.ImgSimulator.all_dmap()
         SHWFSDemonstrator.display_all_dmap(all_dmaps)
        """
        # sanity check
        if axis != 0 and axis != 1:
            raise ValueError("\nContext: Displaying all distortion maps\n" +
                             "Problem: Choice of axes (x,y) invalid\n" +
                             "Solution: Input 'axis' argument should be 0 (x-axis) or 1 (y-axis)")

        # Find the min and max
        smin = all_dmaps[0, 0][axis].min()
        smax = all_dmaps[0, 0][axis].max()
        for i in range(wfs.num_lenslet):
            for j in range(wfs.num_lenslet):
                if smin > all_dmaps[j, i][axis].min():
                    smin = all_dmaps[j, i][axis].min()
                if smax < all_dmaps[j, i][axis].max():
                    smax = all_dmaps[j, i][axis].max()

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
    def display_dimg(dimg):
        """
        Inspect the dimg that would have been produced by a lenslet at particular conjugated position

        Usage:
         c_lenslet_pos = (0.2,0.5)
         dimg = wfs.ImgSimulator.dimg(c_lenslet_pos)
         SHWFSDemonstrator.display_dimg(dimg)
        """

        plt.imshow(dimg, cmap=plt.cm.gray)
        plt.suptitle("Distorted Image")
        plt.show()

    @staticmethod
    def display_all_dimg(all_dimg):
        """
        Inspect subimages produced on SH WHF's detector plane

        Usage:
         all_dimg = wfs.ImgSimulator.all_dimg()
         SHWFSDemonstrator.display_all_dimg

        Science:
         1) Observe vignetting effect
         2) Observe variation in distortions
        """

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
    def display_all_shifts(wfs):
        """
        Visualize the shifts
        """
        all_shifts = wfs.runWFS()

        # Extract x and y components from nested ndarray
        Vtake = np.vectorize(np.take)
        xShifts = Vtake(all_shifts, [0], axis=0)  # [radians/meter]
        yShifts = Vtake(all_shifts, [1], axis=0)  # [radians/meter]

        fig1 = plt.figure(1)

        ax1 = plt.subplot(1, 2, 1)
        ax1.set_title("x shifts")
        plt.imshow(xShifts, interpolation='None')

        ax2 = plt.subplot(1, 2, 2)
        ax2.set_title("y shifts")
        plt.imshow(yShifts, interpolation='None')

        # TODO: figure a way to display metascreen as well
        # # plt.subplot(2,1,2)
        # N = len(wfs.atmos.scrns)
        # for i in range(N):
        #     scrn = wfs.atmos.scrns[i]
        #     ax = plt.subplot(1,N,i+1)
        #     ax.set_title("MetaScreen" + str(i))
        #     screen = wfs._get_metascreen(scrn)
        #     plt.imshow(screen, interpolation='None')


        plt.show()

    """ How ImgSim Works Methods """

    @staticmethod
    def display_angles(wfs):
        """
        Every pixel in a WFS detector is associated with a viewing angle through the telescope

        Usage:
         wfs = WideFieldSHWFS(0, 16, 128, at, tel)
         SHWFSDemonstrator.display_angles(wfs)
        """
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
        Visualize the vigenetting effect for a single lenslet

        Usage:
         wfs = wfs = WideFieldSHWFS(0, 16, 128, at, tel)
         c_lenslet_pos = (0.2,0.5)
         SHWFSDemonstrator.display_vignette(wfs, c_lenslet_pos)
        """

        img = wfs.ImgSimulator._get_vignette_mask(c_lenslet_pos)

        plt.imshow(img, cmap=plt.cm.gray)
        plt.colorbar()
        plt.show()

    @staticmethod
    def display_all_vignette(wfs):
        """
        Visualize the vignetting effect for all lenslets

        Usage:
         wfs = WideFieldSHWFS(0, 16, 128, at, tel)
         SHWFSDemonstrator.display_all_vignette(wfs)
        """
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
    def display_runWFS_screen(wfs):
        """
        Compare the quality of default runWFS output with true phase screen and "Chop,slopify, and reconstruct"
        Usage:
         wfs = WideFieldSHWFS(0, 16, 128, at, tel)
         SHWFSDemonstrator.display_runWFS_screen(wfs)
        """
        # 1) True phase screen
        screen = wfs._get_metascreen(wfs.atmos.scrns[0])

        # 2) Reconstructed from WFS output
        all_shifts = wfs.runWFS()
        sensed = wfs._reconstruct_WF(all_shifts)

        # 3) Chop, slopify, and reconstruct
        phasescreen = wfs.atmos.scrns[0]
        num = phasescreen.shape[0] / wfs.num_lenslet
        slopes = np.empty((2, wfs.num_lenslet, wfs.num_lenslet))
        for j in range(wfs.num_lenslet):
            for i in range(wfs.num_lenslet):
                oXSlope, oYSlope = _TiltifyMethods.tiltify1(phasescreen[j * num:(j + 1) * num, i * num:(i + 1) * num],
                                                            wfs.tel.wavelength, wfs.conjugated_lenslet_size)
                slopes[0, j, i] = oXSlope
                slopes[1, j, i] = oYSlope
        img = ReconMethods.LeastSquare(slopes[0, :, :], slopes[1, :, :])

        plt.figure(1)

        ax1 = plt.subplot(131)
        ax1.set_title("True phase screen")
        im1 = ax1.imshow(screen)
        plt.colorbar(im1)

        ax2 = plt.subplot(132)
        ax2.set_title("Reconstructed from WFS output")
        im2 = ax2.imshow(sensed)
        plt.colorbar(im2)

        ax3 = plt.subplot(133)
        ax3.set_title("Chop, slopify, and reconstruct")
        im3 = ax3.imshow(img)
        plt.colorbar(im3)

        plt.show()

    """ Evaluation """

    @staticmethod
    def eval_dimg(wfs, c_lenslet_pos):
        """
        Evaluate distortion to test image in a lenslet at particular conjugated position

        dimg vs trueimage

        Usage:
         wfs = wfs = WideFieldSHWFS(0, 16, 128, at, tel)
         c_lenslet_pos = (0.2,0.5)
         SHWFSDemonstrator.eval_dimg(wfs, c_lenslet_pos)
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
    def eval_reconImg(wfs, c_lenslet_pos=(0, 0)):
        """
        Evaluate quality of reconstructed test image.
        Reconstructed test image is used with all_dimg in ImageInterpretation to generate all_shifts

        reconImg vs trueimage

        Usage:
         wfs = WideFieldSHWFS(0, 16, 128, at, tel)
         SHWFSDemonstrator.eval_reconImg(wfs)
        """

        # Compute the stuff
        dimg = wfs.ImgSimulator.dimg(c_lenslet_pos)
        recon_img = wfs.ImgInterpreter.get_recon_img()
        true_img = wfs.ImgSimulator.get_test_img()

        # compute the scores
        from ImageInterpreter import _CorrelationAlgorithms
        dimg_score = _CorrelationAlgorithms._SquaredDifferenceFunction(dimg, true_img, 0, 0)
        reconImg_score = _CorrelationAlgorithms._SquaredDifferenceFunction(recon_img, true_img, 0, 0)
        trueImg_score = _CorrelationAlgorithms._SquaredDifferenceFunction(true_img, true_img, 0, 0)

        print "\n<<<CORRELATION SCORES>>>"

        print "dimg: " + str(dimg_score)
        print "reconImg: " + str(reconImg_score)
        print "trueImg: " + str(trueImg_score)

        # Display
        plt.figure(1)

        ax1 = plt.subplot(1, 3, 1)
        ax1.imshow(dimg)
        ax1.set_title("A distorted lenslet image")

        ax2 = plt.subplot(1, 3, 2)
        ax2.imshow(recon_img)
        ax2.set_title("Reconstructed Image")

        ax3 = plt.subplot(1, 3, 3)
        ax3.imshow(true_img)
        ax3.set_title("True Image")

        plt.show()

    @staticmethod
    def eval_all_shifts(wfs):
        """
        Evaluate quality of image interpretation.

        actual shifts (dmaps) vs measured shift (from dimg)

        Usage:
         wfs = WideFieldSHWFS(0, 16, 128, at, tel)
         SHWFSDemonstrator.eval_all_shifts(wfs)
        """
        all_dmaps = wfs.ImgSimulator.all_dmap()
        all_shifts = wfs.runWFS()

        # Display process
        fig = plt.figure(1)

        # Extract the x component of all dmaps
        Vtake_ndarray = np.vectorize(np.take, otypes=[np.ndarray])
        all_dmaps_x = Vtake_ndarray(all_dmaps, [0], axis=0)

        # Find min and max distortion
        Vmin = np.vectorize(np.min)
        asmin = (Vmin(all_dmaps_x)).min()  # actual shift min
        Vmax = np.vectorize(np.max)
        asmax = (Vmax(all_dmaps_x)).max()  # actual shift max

        # display loop
        plt.subplot(1, 2, 1)
        for j in range(wfs.num_lenslet):
            for i in range(wfs.num_lenslet):
                plt.subplot(wfs.num_lenslet, 2 * wfs.num_lenslet, j * 2 * wfs.num_lenslet + i + 1)
                plt.axis('off')
                im = plt.imshow(all_dmaps_x[j, i], vmin=asmin, vmax=asmax)

        # add all_dmap colorbar
        cbar_ax = fig.add_axes([0.5, 0.15, 0.005, 0.7])
        fig.colorbar(im, cax=cbar_ax)

        # Extract x component of all_shifts
        Vtake = np.vectorize(np.take)
        all_shifts_x = Vtake(all_shifts, [0], axis=0)  # take the x component of the shift

        # Finding min and max of measured shifts
        msmin = all_shifts_x.min()  # measured shift min
        msmax = all_shifts_x.max()

        plt.subplot(1, 2, 2)
        plt.axis('off')

        plt.imshow(all_shifts_x, vmin=msmin, vmax=msmax, interpolation='None')
        plt.colorbar()

        plt.show()

    @staticmethod
    def eval_measuremodes_screen(wfs):
        """
        Evaluate the quality of slopes sensed using different measurement modes.
        Visualise reconstructed phase screen from measured shifts

        Usage:
         wfs = WideFieldSHWFS(0, 16, 128, at, tel)
         SHWFSDemonstrator.eval_measuremodes_surface(wfs)

        """
        # Compute the stuff
        screen = wfs._get_metascreen(wfs.atmos.scrns[0])
        all_dimg = wfs.ImgSimulator.all_dimg()
        all_shifts1 = wfs.ImgInterpreter.average_all_measure(all_dimg)
        all_shifts2 = wfs.ImgInterpreter.spiral_all_measure(all_dimg)

        # Display
        plt.figure(1)

        ax1 = plt.subplot(131)
        ax1.set_title("True phase scrn")
        im1 = ax1.imshow(screen)
        plt.colorbar(im1)

        ax2 = plt.subplot(132)
        ax2.set_title("Average Mode: Averaged dimg as RefImg")
        im2 = ax2.imshow(wfs._reconstruct_WF(all_shifts1))
        plt.colorbar(im2)

        ax3 = plt.subplot(133)
        ax3.set_title("Spiral Mode: Adjacent dimg as RefImg")
        im3 = ax3.imshow(wfs._reconstruct_WF(all_shifts2))
        plt.colorbar(im3)

        plt.show()

    @staticmethod
    def eval_measuremodes_shift(wfs, i, j):
        """
        Evaluate the quality of a single slope sensed using different measurement modes.
        Visualise shifts for a single lenslet

        Usage:
         wfs = WideFieldSHWFS(0, 16, 128, at, tel)
         SHWFSDemonstrator.eval_measuremodes_shifts(wfs, 3,4)
        """
        c_pos = wfs.ImgSimulator._index_to_c_pos(i, j)
        dmap = wfs.ImgSimulator.dmap(c_pos)
        all_dimg = wfs.ImgSimulator.all_dimg()
        shift1 = wfs.ImgInterpreter.average_measure(i, j, all_dimg)
        shift2 = wfs.ImgInterpreter.spiral_measure(i, j, all_dimg)
        xglobal = dmap[0].mean()
        yglobal = dmap[1].mean()

        print "\n<<<SHIFT MEASUREMENT RESULTS>>>"
        print "Mean Appied Shift = " + str((xglobal, yglobal))
        print "Measured Shift (Averaged Mode) = " + str(shift1)
        print "Measured Shift (Spiral Mode) = " + str(shift2)

        plt.imshow(dmap[0])
        plt.show()

    @staticmethod
    def eval_measuremodes_all_shifts(wfs):
        """
        Evaluate the quality of slopes sensed using different measurement modes.
        Visualise all_shifts (which is the measured data)

        Usage:
         wfs = WideFieldSHWFS(0, 16, 128, at, tel)
         SHWFSDemonstrator.eval_measuremodes_all_shifts(wfs)
        """
        # Compute the stuff
        all_dmap = wfs.ImgSimulator.all_dmap()
        all_dimg = wfs.ImgSimulator.all_dimg()
        all_shifts1 = wfs.ImgInterpreter.average_all_measure(all_dimg)
        all_shifts2 = wfs.ImgInterpreter.spiral_all_measure(all_dimg)
        Vtake = np.vectorize(np.take)

        # Display
        plt.figure(1)
        num = all_dmap.shape[0]
        for j in range(num):
            for i in range(num):
                plt.subplot(num, num*3, 3*num*j+i+1)
                # TODO: remove hardcoded min/max
                plt.imshow(all_dmap[j,i][0],vmin = -16, vmax=16)
                plt.axis('off')

        ax2 = plt.subplot(132)
        ax2.set_title("Average Mode: Averaged dimg as RefImg")
        im2 = ax2.imshow(Vtake(all_shifts1,[0],axis=0),interpolation='None')
        plt.colorbar(im2)

        ax3 = plt.subplot(133)
        ax3.set_title("Spiral Mode: Adjacent dimg as RefImg")
        im3 = ax3.imshow(Vtake(all_shifts2,[0],axis=0),interpolation='None')
        plt.colorbar(im3)

        plt.show()

    @staticmethod
    def eval_metascreen(wfs):
        """
        Evaluate how well dmaps fits metascreen

        metascreen vs all_dmaps

        Usage:
         wfs = WideFieldSHWFS(0, 16, 128, at, tel)
         SHWFSDemonstrator.eval_metascreen(wfs)

        """
        all_dmaps = wfs.ImgSimulator.all_dmap()

        # Display process
        fig1 = plt.figure(1)
        plt.subplot(1, 2, 1)
        plt.imshow(wfs._get_metascreen(wfs.atmos.scrns[0]))

        for j in range(wfs.num_lenslet):
            for i in range(wfs.num_lenslet):
                plt.subplot(wfs.num_lenslet, 2 * wfs.num_lenslet, j * 2 * wfs.num_lenslet + i + 1 + wfs.num_lenslet)
                plt.axis('off')
                plt.imshow(all_dmaps[j, i][0])

        plt.show()



if __name__ == '__main__':
    tel = Telescope(2.5)
    at = Atmosphere()
    at.create_default_screen(0, 0.15)
    wfs = WideFieldSHWFS(0, 16, 128, at, tel)
    c_lenslet_pos = (0.2,0.5)
    dimg = wfs.ImgSimulator.dimg(c_lenslet_pos)
    SHWFSDemonstrator.display_dimg(dimg)
