# LayerMCAO

LayerMCAO is a 2015 UROP project in the Cavendish Laboratory, Cambridge University supervised by Dr Aglae Kellerer. Its purpose is to simulate layer-oriented Multi Conjugate Adaptive Optics (MCAO).

Layer-oriented MCAO is a new method of using MCAO. It differs from traditional star-oriented MCAO by optically conjugating not only the Deformable Mirror (DM) but the Wavefront Sensor (WFS) to height. By conjugating the WFS to height, we should be better able to sense the high altitude turbulence and thus correct for it. Because layer-oriented MCAO performs better with large field of view, this technique is ideal for solar astronomy.

LayerMCAO is written in Python and currently implements three AO components: 1) Shack Hartmann Wavefront Sensor 2) Telescope 3) Atmosphere. Most of the work was done on the WFS, implementing the methods to generate and interpret the SH-WFS lenslet images. Telescope and Atmosphere objects largely contains the specifications and turbulence information needed by the lenslet image methods. 

For the sake of modularity, the lenslet image generation methods and lenslet image interpretation methods are kept in two seperate objects and referenced by the SH-WFS object. The only exchange of information between the two classes should be the lenslet images. This keeps physical simulation of atmospheric seeing separate from post-processing methods. 

Multiple methods for interpreting lenslet images are implemented and contained in the ImageInterpretor class. The ```all\_dimg\_to\_shifts``` method specifies the default method. At the moment, only one routine for generating lenslet images is implemented in the ImageGenerator class.

## Dependencies
1. Numpy
2. Scipy
3. Matlab

## Usage
1. ```wfs.runWFS()```
This methods generates all the lenslet images (using ImageGenerator ```all\_dimg``` method) and passes the result into ImageInterpretor's ```all\_dimg_to\_shifts``` method. 
2. Demonstration function
Multiple functions for testing and debugging were written during the course of developing the code. These functions are contained in a static class WFSDemonstrator. Choose one function to run. The docstrings should be self-explanatory.


## Naming Conventions
1. dmap: distortion map \[pixel\]

    - The shifts in units of pixels to be applied to test image to generate distorted image (dimg)
    - ndarray.shape = (2, pixel_lenslet, pixel_lenslet)
    - ndarray.dtype = float

2. all_dmap: all lenslets' distortion map

    - A collection of dmaps
    - ndarray.shape = (num_lenslet, num_lenslet)
    - ndarray.dtype = ndarray

3. dimg: distorted image

    - The image behind a SH lenslet as distorted by atmosphere
    - ndarray.shape = (pixel_lenslet, pixel_lenslet)
    - ndarray.dtype = float

4. all_dimg: all lenslets' distorted image

    - A collection of dimgs
    - ndarray.shape = (num_lenslet, num_lenslet)
    - ndarray.dtype = ndarray

5. Shifts / Tilts / Gradient
    - Shift: the number of pixels the point IMAGE is shifted by in detector plane \[pixel\]
    - Tilt: the overage angle the WAVEFRONT is tilted by the phase screens \[radians\]
    - Gradient: the average gradient over a portion of the PHASE SCREEN   \[radians/meter\]

6. ReconImg / RefImg
    - ReconImg: Reconstructed test image. Given many dimg, we try to reconstruct the original image without distortion. To be used as RefImg.
    - RefImg: Reference image as which image shift is measured.







