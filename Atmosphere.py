__author__ = 'tpl'
import pickle

import matplotlib.pyplot as plt

from soapy import atmosphere


class Atmosphere(object):
    def __init__(self):
        # Screen objects needs to be kept in decreasing order of height
        # if physical beam propagation algorithm were to be used.
        # See WFS.ImageSimulator._stack_lensletscreen for usage.
        self.scrns = []

    def create_screen(self,r_0,N,delta,L_0,l_0,height):
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
        new_screen = Screen(r_0,N,delta,L_0,l_0,height, screen_id)
        self.scrns.append(new_screen)
        sorted(self.scrns, key=lambda screen: screen.height)

    def add_screen(self,scrn):
        self.scrns.append(scrn)
        sorted(self.scrns, key=lambda screen: screen.height)

    def create_default_screen(self, screen_id, height):
        scrn = Screen.create_default_screen(screen_id, height)
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
            self.create_default_screen(i,height_list[i])

    def display_screen(self,n):
        """
        :param n: screen index number. Ground layer is index 0
        :return:
        """
        plt.imshow((self.scrns[n]).phase_screen)
        plt.colorbar()
        plt.show()

class Screen(object):
    def __init__(self,r_0,N,delta,L_0,l_0, height=0, screen_id=0):
        """
        :param r_0: Fried Parameter [meters]
        :param N: Size of phase screen array [pixels]
        :param delta: Size of each phase screen pixel [meters]
        :param L_0: Outer Scale [meters]
        :param l_0: Inner Scale [meters]
        :param height: Height of phase screen [meters]
        :param ID: ID number of the
        :return: Screen object [Screen]

        Usage:
        1) get phase screen
            scrn.phase_screen
        2) get height of screen
            scrn.height
        """
        # save the details of the phase screen
        self.r_0 = r_0
        self.N = N
        self.delta = delta
        self.L_0 = L_0
        self.l_0 = l_0
        self.height = height

        self.ID = screen_id

        # Generate the phase screen
        self.phase_screen = atmosphere.ft_phase_screen(r_0,N,delta,L_0,l_0)

    @staticmethod
    def create_default_screen(height, screen_id):
        """
        A "get default instance" method

        :param height: height to position screen [number][meters]
        :param screen_id: id of default screen to be added [int]
        :return: screen object [Screen]
        """
        try:
            f = open("DafaultScreen"+str(height)+"_"+str(screen_id)+".scrn",'rb')
            screen = pickle.load(f)
            assert screen.isinstance(Screen)
            print "Found a previously computed default screen"
            return screen
        except IOError:
            print "Computing a new default screen"

            # Configuration of default screen
            N = 2048 # Number of pixels
            r_0 = 1 # Fried parameter [meters]
            delta = 0.01 # Pixel size [meter]
            L_0 = 20 # Outer scale [meters]
            l_0 = 0.01 # Inner scale [meters]

            new_screen = Screen(r_0,N,delta,L_0,l_0,height, screen_id)

            # Saving
            f = open("DafaultScreen"+str(height)+"_"+str(screen_id)+".scrn",'wb')
            pickle.dump(new_screen,f,pickle.HIGHEST_PROTOCOL)

            return new_screen

# TODO: input algorithms to create phase screens here in a class

# #Test Atmosphere
#
# at = Atmosphere ()
# at.create_default_screen(0,0)