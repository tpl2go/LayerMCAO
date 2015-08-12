__author__ = 'tpl'

from Telescope import Telescope
from Atmosphere import Atmosphere
from WFS import WideFieldSHWFS, SHWFS_Demonstrator
import matplotlib.pyplot as plt

def dimg_vs_height(c_pos):
    tel = Telescope(2.5)
    at = Atmosphere()
    at.create_default_screen(2000,0)
    wfs0 = WideFieldSHWFS(0,16,128,at,tel)
    wfs1 = WideFieldSHWFS(1000,16,128,at,tel)
    wfs2 = WideFieldSHWFS(2000,16,128,at,tel)
    wfs3 = WideFieldSHWFS(3000,16,128,at,tel)
    wfs4 = WideFieldSHWFS(4000,16,128,at,tel)

    plt.figure(1)

    ax1 = plt.subplot(2,3,1)
    ax1.imshow(wfs0.ImgSimulator.get_test_img())
    ax1.set_title("true_img")

    ax2 = plt.subplot(2,3,2)
    dimg0 = wfs0.ImgSimulator.dimg(c_pos)
    ax2.imshow(dimg0)
    ax2.set_title("0m")

    ax3 = plt.subplot(2,3,3)
    dimg1 = wfs1.ImgSimulator.dimg(c_pos)
    ax3.imshow(dimg1)
    ax3.set_title("1000m")

    ax4 = plt.subplot(2,3,4)
    dimg2 = wfs2.ImgSimulator.dimg(c_pos)
    ax4.imshow(dimg2)
    ax4.set_title("2000m")

    ax5 = plt.subplot(2,3,5)
    dimg3 = wfs3.ImgSimulator.dimg(c_pos)
    ax5.imshow(dimg3)
    ax5.set_title("3000m")

    ax6 = plt.subplot(2,3,6)
    dimg4 = wfs4.ImgSimulator.dimg(c_pos)
    ax6.imshow(dimg4)
    ax6.set_title("4000m")

    plt.show()

def dmap_vs_height(c_pos, axis=0):
    tel = Telescope(2.5)
    at = Atmosphere()
    at.create_default_screen(2000,0)
    wfs0 = WideFieldSHWFS(0,16,128,at,tel)
    wfs1 = WideFieldSHWFS(1000,16,128,at,tel)
    wfs2 = WideFieldSHWFS(2000,16,128,at,tel)
    wfs3 = WideFieldSHWFS(3000,16,128,at,tel)
    wfs4 = WideFieldSHWFS(4000,16,128,at,tel)
    wfs5 = WideFieldSHWFS(5000,16,128,at,tel)

    fig = plt.figure(1)
    fig.suptitle("Variation of dmap with conjugated height")

    MAX = 0.07
    MIN = -0.07

    ax1 = plt.subplot(2,3,1)
    dmap0 = wfs0.ImgSimulator.dmap(c_pos)
    ax1.imshow(dmap0[axis],vmin=MIN,vmax=MAX)
    ax1.set_title("0m")

    ax2 = plt.subplot(2,3,2)
    dmap1 = wfs1.ImgSimulator.dmap(c_pos)
    ax2.imshow(dmap1[axis],vmin=MIN,vmax=MAX)
    ax2.set_title("1000m")

    ax3 = plt.subplot(2,3,3)
    dmap2 = wfs2.ImgSimulator.dmap(c_pos)
    ax3.imshow(dmap2[axis],vmin=MIN,vmax=MAX)
    ax3.set_title("2000m")

    ax4 = plt.subplot(2,3,4)
    dmap3 = wfs3.ImgSimulator.dmap(c_pos)
    ax4.imshow(dmap3[axis],vmin=MIN,vmax=MAX)
    ax4.set_title("3000m")

    ax5 = plt.subplot(2,3,5)
    dmap4 = wfs4.ImgSimulator.dmap(c_pos)
    ax5.imshow(dmap4[axis],vmin=MIN,vmax=MAX)
    ax5.set_title("4000m")

    ax6 = plt.subplot(2,3,6)
    dmap5 = wfs5.ImgSimulator.dmap(c_pos)
    im = ax6.imshow(dmap5[axis],vmin=MIN,vmax=MAX)
    ax6.set_title("5000m")

    fig.subplots_adjust(bottom=0.2)
    cbar_ax = fig.add_axes([0.15, 0.1, 0.7, 0.05])
    fig.colorbar(im, cax=cbar_ax, orientation = 'horizontal')

    plt.show()

def vignette_vs_height():
    tel = Telescope(2.5)
    at = Atmosphere()
    at.create_default_screen(2000,0)
    # wfs0 = WideFieldSHWFS(0,16,128,at,tel)
    # wfs1 = WideFieldSHWFS(1000,16,128,at,tel)
    # wfs2 = WideFieldSHWFS(2000,16,128,at,tel)
    # wfs3 = WideFieldSHWFS(3000,16,128,at,tel)
    wfs4 = WideFieldSHWFS(8000,16,128,at,tel)

    #TODO: figure out how to make a sub sub plot?
    # SHWFS_Demonstrator.display_all_vignette(wfs0)
    # SHWFS_Demonstrator.display_all_vignette(wfs1)
    # SHWFS_Demonstrator.display_all_vignette(wfs2)
    # SHWFS_Demonstrator.display_all_vignette(wfs3)
    SHWFS_Demonstrator.display_all_vignette(wfs4)

def refImg_vs_height():
    tel = Telescope(2.5)
    at = Atmosphere()
    at.create_default_screen(2000,0)
    wfs0 = WideFieldSHWFS(0,16,128,at,tel)
    wfs1 = WideFieldSHWFS(1000,16,128,at,tel)
    wfs2 = WideFieldSHWFS(2000,16,128,at,tel)
    wfs3 = WideFieldSHWFS(3000,16,128,at,tel)
    wfs4 = WideFieldSHWFS(4000,16,128,at,tel)

    plt.figure(1)

    ax1 = plt.subplot(2,3,1)
    ax1.imshow(wfs0.ImgSimulator.get_test_img())
    ax1.set_title("true_img")

    ax2 = plt.subplot(2,3,2)
    all_dimg0 = wfs0.ImgSimulator.all_dimg()
    recon_truth0 = wfs0.ImgInterpreter.spatialAverage(all_dimg0)
    ax2.imshow(recon_truth0)
    ax2.set_title("0m")

    ax3 = plt.subplot(2,3,3)
    all_dimg1 = wfs1.ImgSimulator.all_dimg()
    recon_truth1 = wfs1.ImgInterpreter.spatialAverage(all_dimg1)
    ax3.imshow(recon_truth1)
    ax3.set_title("1000m")

    ax4 = plt.subplot(2,3,4)
    all_dimg2 = wfs2.ImgSimulator.all_dimg()
    recon_truth2 = wfs2.ImgInterpreter.spatialAverage(all_dimg2)
    ax4.imshow(recon_truth2)
    ax4.set_title("2000m")

    ax5 = plt.subplot(2,3,5)
    all_dimg3 = wfs3.ImgSimulator.all_dimg()
    recon_truth3 = wfs3.ImgInterpreter.spatialAverage(all_dimg3)
    ax5.imshow(recon_truth3)
    ax5.set_title("3000m")

    ax6 = plt.subplot(2,3,6)
    all_dimg4 = wfs4.ImgSimulator.all_dimg()
    recon_truth4 = wfs4.ImgInterpreter.spatialAverage(all_dimg4)
    ax6.imshow(recon_truth4)
    ax6.set_title("4000m")

    plt.show()

def dmap_intensity_vs_lenslet_size():
        fig1 = plt.figure(1)
        at = Atmosphere()
        at.create_default_screen(1000,0)
        plt.suptitle("Standard r_0 = 1.0m")

        print "Generating WFS 1"
        ax1 = plt.subplot(2,3,1)
        tel1 = Telescope(1)
        wfs1 = WideFieldSHWFS(0,16,128,at,tel1)
        ax1.set_title(str(wfs1.conjugated_lenslet_size)+"m")
        im1 = ax1.imshow(wfs1.ImgSimulator.dmap((0,0))[0])
        plt.colorbar(im1)

        print "Generating WFS 2"
        ax2 = plt.subplot(2,3,2)
        tel2 = Telescope(2)
        wfs2 = WideFieldSHWFS(0,16,128,at,tel2)
        ax2.set_title(str(wfs2.conjugated_lenslet_size)+"m")
        im2 = ax2.imshow(wfs2.ImgSimulator.dmap((0,0))[0])
        plt.colorbar(im2)

        print "Generating WFS 3"
        ax3 = plt.subplot(2,3,3)
        tel3 = Telescope(4)
        wfs3 = WideFieldSHWFS(0,16,128,at,tel3)
        ax3.set_title(str(wfs3.conjugated_lenslet_size)+"m")
        im3 = ax3.imshow(wfs3.ImgSimulator.dmap((0,0))[0])
        plt.colorbar(im3)

        print "Generating WFS 4"
        ax4 = plt.subplot(2,3,4)
        tel4 = Telescope(8)
        wfs4 = WideFieldSHWFS(0,16,128,at,tel4)
        ax4.set_title(str(wfs4.conjugated_lenslet_size)+"m")
        im4 = ax4.imshow(wfs4.ImgSimulator.dmap((0,0))[0])
        plt.colorbar(im4)

        print "Generating WFS 5"
        ax5 = plt.subplot(2,3,5)
        tel5 = Telescope(16)
        wfs5 = WideFieldSHWFS(0,16,128,at,tel5)
        ax5.set_title(str(wfs5.conjugated_lenslet_size)+"m")
        im5 = ax5.imshow(wfs5.ImgSimulator.dmap((0,0))[0])
        plt.colorbar(im5)

        print "Generating WFS 6"
        ax6 = plt.subplot(2,3,6)
        tel6 = Telescope(32)
        wfs6 = WideFieldSHWFS(0,16,128,at,tel6)
        ax6.set_title(str(wfs6.conjugated_lenslet_size)+"m")
        im6 = ax6.imshow(wfs6.ImgSimulator.dmap((0,0))[0])
        plt.colorbar(im6)

        print "Done!"

        plt.show()


if __name__ == "__main__":
    dmap_intensity_vs_lenslet_size()