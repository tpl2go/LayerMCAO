__author__ = 'tpl'
import pickle
import os
import matplotlib.pyplot as plt
import random
import time
import numpy as np
import scipy.fftpack as fft


class Atmosphere(object):
    def __init__(self):
        # Screen objects needs to be kept in decreasing order of height
        # if physical beam propagation algorithm were to be used.
        # See WFS.ImageSimulator._stack_lensletscreen for usage.
        self.scrns = []

    def create_screen(self, r_0, N, delta, L_0, l_0, height):
        """
        :param r_0: Fried Parameter [meters]
        :param N: Size of phase screen array [pixels]
        :param delta: Size of each phase screen pixel [meters]
        :param L_0: Outer Scale [meters]
        :param l_0: Inner Scale [meters]
        :param height: Height of phase screen [meters]
        :return:
        """
        screen_id = len(self.scrns)
        new_screen = Screen(r_0, N, delta, L_0, l_0, height, screen_id)
        self.scrns.append(new_screen)
        sorted(self.scrns, key=lambda screen: screen.height)

    def save_screens(self):
        # Create folder to contain collection of Screen objects
        try:
            os.makedirs("ScrnLib")
        except:
            pass

        for scrn in self.scrns:
            path = os.getcwd() + "/ScrnLib/" + "Screen" + scrn.hash_function() + ".scrn"
            f = open(path, 'wb')
            pickle.dump(scrn, f, pickle.HIGHEST_PROTOCOL)
            f.close()

    def load_screen(self, r_0, N, delta, L_0, l_0, height, ID):
        hash = ''.join(map(str, [height, ID, "_", r_0, N, delta, L_0, l_0]))
        path = os.getcwd() + "/ScrnLib/" + "Screen" + hash + ".scrn"
        f = open(path, 'rb')
        screen = pickle.load(f)
        assert isinstance(screen, Screen)
        print "Successfully loaded screen"
        self.add_screen(screen)

    def add_screen(self, scrn):
        assert isinstance(scrn, Screen)
        self.scrns.append(scrn)
        sorted(self.scrns, key=lambda screen: screen.height)

    def create_default_screen(self, height, fried_parameter, screen_id=0):
        """
        Get an instance of screen with default minor parameters and loads it into Atmosphere
        :param height: height to position screen [meters]
        :param fried_parameter: [meters]
        :param screen_id: id of default screen to be added [int]
        
        Usage:
         at = Atmosphere()
         at.create_default_screen(0,0.15) # Phase screen at ground level
        """
        scrn = Screen.get_default_screen(height, fried_parameter, screen_id)
        self.scrns.append(scrn)
        sorted(self.scrns, key=lambda screen: screen.height)

    def create_many_default_screen(self, N, height_list):
        """
        Generates many default screens in the atmospheres
        :param N: Number of screens # [int]
        :param height_list: heights of screens # [int list]
        :return:
        """
        for i in range(N):
            self.create_default_screen(height_list[i], i)

    def display_screen(self, n):
        """
        :param n: screen index number. Ground layer is index 0
        :return:
        """
        plt.imshow((self.scrns[n]).phase_screen)
        plt.colorbar()
        plt.show()


class Screen(object):
    def __init__(self, r_0, N, delta, L_0, l_0, height=0, screen_id=0):
        """
        Usage:
         1) get phase screen: scrn.phase_screen
         2) get height of screen: scrn.height
        :param r_0: Fried Parameter [meters]
        :param N: Size of phase screen array [pixels]
        :param delta: Size of each phase screen pixel [meters]
        :param L_0: Outer Scale [meters]
        :param l_0: Inner Scale [meters]
        :param height: Height of phase screen [meters]
        :param ID: ID number of the
        :return: Screen object [Screen]

        """
        # save the details of the phase screen
        self.r_0 = r_0
        self.N = N
        self.delta = delta
        self.L_0 = L_0
        self.l_0 = l_0
        self.height = height

        # ID is implemented in case we want to differentiate screens with the same parameters
        self.ID = screen_id

        # Generate the phase screen
        self.phase_screen = PhaseScreenGenerator.ft_phase_screen(r_0, N, delta, L_0, l_0)

    @staticmethod
    def get_default_screen(height, fried_parameter, screen_id=0):
        """
        A "get instance" method for Screen. Default values for minor parameters are used.
        Screen is saved.

        :param height: height to position screen [meters]
        :param fried_parameter: [meters]
        :param screen_id: id of default screen to be added [int]
        :return: screen object [Screen]
        """

        # Configuration of default screen
        N = 2048  # Number of pixels
        r_0 = fried_parameter  # Fried parameter [meters]
        delta = 0.01  # Pixel size [meter]
        L_0 = 20  # Outer scale [meters]
        l_0 = 0.01  # Inner scale [meters]

        # Create folder to contain collection of Screen objects
        try:
            os.makedirs("ScrnLib")
        except:
            pass

        # Path to saving location
        # hash_value must be sync with self.hash_function()
        hash_value = ''.join(map(str, [height, "_", screen_id, "_", r_0, "_", N, "_", delta, "_", L_0, "_", l_0]))
        path = os.getcwd() + "/ScrnLib/" + "Screen" + hash_value + ".scrn"

        # Load previously saved screen / Create new screen
        try:
            f = open(path, 'rb')
            screen = pickle.load(f)
            if not isinstance(screen, Screen):
                raise IOError
            print "Found a previously computed default screen"
            f.close()
        except IOError:
            print "Computing a new default screen"
            screen = Screen(r_0, N, delta, L_0, l_0, height, screen_id)
            # Saving
            f = open(path, 'wb')
            pickle.dump(screen, f, pickle.HIGHEST_PROTOCOL)
            f.close()

        return screen

    def hash_function(self):
        return ''.join(map(str, [self.height, "_", self.ID, "_", self.r_0, "_", self.N, "_", self.delta, "_",
                                 self.L_0, "_", self.l_0]))


class PhaseScreenGenerator(object):
    """
    Extracted code from Andrew Reeves' soapy project
    https://github.com/andrewpaulreeves/soapy

    Future Work: Implement various methods for generating phase screens
    """

    @staticmethod
    def ft_phase_screen(r0, N, delta, L0, l0):
        """
        Creates a random phase screen with Von Karmen statistics.
        (Schmidt 2010)
        
        """
        delta = float(delta)
        r0 = float(r0)
        L0 = float(L0)
        l0 = float(l0)

        R = random.SystemRandom(time.time())
        seed = int(R.random() * 100000)
        np.random.seed(seed)

        del_f = 1. / (N * delta)

        fx = np.arange(-N / 2., N / 2.) * del_f

        (fx, fy) = np.meshgrid(fx, fx)
        f = np.sqrt(fx ** 2 + fy ** 2)

        fm = 5.92 / l0 / (2 * np.pi)
        f0 = 1. / L0

        PSD_phi = (0.023 * r0 ** (-5. / 3.) * np.exp(-1 * ((f / fm) ** 2)) /
                   (((f ** 2) + (f0 ** 2)) ** (11. / 6)))

        PSD_phi[(N / 2), (N / 2)] = 0

        cn = ((np.random.normal(size=(N, N)) + 1j * np.random.normal(size=(N, N)))
              * np.sqrt(PSD_phi) * del_f)

        phs = PhaseScreenGenerator.ift2(cn, 1).real

        return phs

    @staticmethod
    def ift2(G, delta_f):
        """
        Wrapper for inverse fourier transform

        Parameters:
            G: data to transform
            delta_f: pixel seperation
        """

        N = G.shape[0]

        g = fft.ifftshift(fft.ifft2(fft.fftshift(G))) * (N * delta_f) ** 2

        return g


class PhaseScreenDemonstrator(object):
    @staticmethod
    def display_fried_param():
        fig1 = plt.figure(1)

        print "Generating Screen 1"
        ax1 = plt.subplot(2, 3, 1)
        sc1 = Screen(0.1, 2048, 0.01, 20, 0.01)
        ax1.set_title("r_0=0.1")
        im1 = ax1.imshow(sc1.phase_screen)
        plt.colorbar(im1)

        print "Generating Screen 2"
        ax2 = plt.subplot(2, 3, 2)
        sc2 = Screen(0.2, 2048, 0.01, 20, 0.01)
        ax2.set_title("r_0=0.2")
        im2 = ax2.imshow(sc2.phase_screen)
        plt.colorbar(im2)

        print "Generating Screen 3"
        ax3 = plt.subplot(2, 3, 3)
        sc3 = Screen(0.4, 2048, 0.01, 20, 0.01)
        ax3.set_title("r_0=0.4")
        im3 = ax3.imshow(sc3.phase_screen)
        plt.colorbar(im3)

        print "Generating Screen 4"
        ax4 = plt.subplot(2, 3, 4)
        sc4 = Screen(0.8, 2048, 0.01, 20, 0.01)
        ax4.set_title("r_0=0.8")
        im4 = ax4.imshow(sc4.phase_screen)
        plt.colorbar(im4)

        print "Generating Screen 5"
        ax5 = plt.subplot(2, 3, 5)
        sc5 = Screen(1.6, 2048, 0.01, 20, 0.01)
        ax5.set_title("r_0=1.6")
        im5 = ax5.imshow(sc5.phase_screen)
        plt.colorbar(im5)

        print "Generating Screen 6"
        ax6 = plt.subplot(2, 3, 6)
        sc6 = Screen(3.2, 2048, 0.01, 20, 0.01)
        ax6.set_title("r_0=3.2")
        im6 = ax6.imshow(sc6.phase_screen)
        plt.colorbar(im6)

        print "Done!"

        plt.show()


if __name__ == "__main__":
    # at = Atmosphere()
    # at.create_default_screen(0,0)
    PhaseScreenDemonstrator.display_fried_param()
