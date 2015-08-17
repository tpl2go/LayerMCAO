import sys
import numpy as np
import matplotlib.pyplot as plt

# [Software Design] I am resisting the implementation of an abstract ImageInterpreter / ImageSimulator class because
# I do not want the contracts to be set in stone. This code is still highly developmental and fluid. And the fact
# that it is written in python means that it is still a highly dynamic and rapidly evolving piece of code.

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

    def all_dimg_to_shifts(self, all_dimg):
        return self.measure_all_shifts(all_dimg)

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

    def _measure_dimg_shifts(self, dimg, refImg):
        c_matrix = _CorrelationAlgorithms.SDF(dimg, refImg, 16)
        try:
            s_matrix = _InterpolationAlgorithms.c_to_s_matrix(c_matrix)
            shifts = _InterpolationAlgorithms.TwoDLeastSquare(s_matrix)
        except MinOnEdgeError as e :
            # TODO: Think - when failure occurs what value is good to return
            xShift, yShift = 0.0, 0.0
            # shifts = np.array((xShift - c_matrix.shape[0] / 2, yShift - c_matrix.shape[1] / 2))
            shifts = np.array((xShift,yShift),'float64')
        return shifts

    def get_ref_img(self):
        all_dimg = self.ImgSimulator.all_dimg()
        ref = _RefImgReconMethods.recon1(all_dimg, self.ImgSimulator.all_vignette_mask())
        return ref

    def measure_all_shifts(self, all_dimg):
        ref = _RefImgReconMethods.recon1(all_dimg, self.ImgSimulator.all_vignette_mask())
        iMax, jMax = all_dimg.shape
        all_shifts = np.empty((jMax, iMax),np.ndarray)
        for j in range(jMax):
            for i in range(iMax):
                sys.stdout.write('\r' + "Now measuring dimg index " + str((i, j)))
                (xShift, yShift) = self._measure_dimg_shifts(all_dimg[j, i], ref)
                all_shifts[j, i] = np.array((xShift,yShift))

        sys.stdout.write(" Done!\n")
        print("Failure rate: ", MinOnEdgeError.count, "/", jMax*iMax)
        return all_shifts


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

        diff = (img[y1t:y2t, x1t:x2t] - imgRef[y1r:y2r, x1r:x2r]) ** 2

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

        if min[0] == 0 or min[0] == c_matrix.shape[0] - 1 \
                or min[1] == 0 or min[1] == c_matrix.shape[1] - 1:
            raise MinOnEdgeError("Minimum is found on an edge",min)

        s_matrix = c_matrix[min[0] - 1:min[0] + 2, min[1] - 1:min[1] + 2]

        return s_matrix

    @staticmethod
    def TwoDLeastSquare(s_matrix):
        a2 = (np.average(s_matrix[:, 2]) - np.average(s_matrix[:, 0])) / 2.0
        a3 = (np.average(s_matrix[:, 2]) - 2 * np.average(s_matrix[:, 1]) + np.average(s_matrix[:, 0])) / 2.0
        a4 = (np.average(s_matrix[2, :]) - np.average(s_matrix[0, :])) / 2.0
        a5 = (np.average(s_matrix[2, :]) - 2 * np.average(s_matrix[1, :]) + np.average(s_matrix[0, :])) / 2.0
        a6 = (s_matrix[2, 2] - s_matrix[2, 0] - s_matrix[0, 2] + s_matrix[0, 0]) / 4.0

        # 1D Minimum (I have no idea what this means)
        # x_min = -a2/(2*a3)
        # y_min = -a4/(2*a5)

        # 2D Minimum
        x_min = (2 * a2 * a5 - a4 * a6) / (a6 ** 2 - 4 * a3 * a5)
        y_min = (2 * a3 * a4 - a2 * a6) / (a6 ** 2 - 4 * a3 * a5)

        return np.array((x_min, y_min))

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

class MinOnEdgeError(RuntimeError):
    count = 0
    def __init__(self,msg, minLocation):
        self.msg = msg
        self.min = minLocation
        MinOnEdgeError.count += 1
