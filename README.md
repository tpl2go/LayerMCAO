# LayerMCAO

LayerMCAO is a 2015 UROP project in the Cavendish Laboratory, Cambridge University supervised by Dr Aglae Kellerer. Its purpose is to simulate layer-oriented Multi Conjugate Adaptive Optics (MCAO).

Layer-oriented MCAO is a new method of using MCAO. It differs from traditional star-oriented MCAO by optically conjugating not only the *Deformable Mirror* (DM) but the *Wavefront Sensor* (WFS) to height. Because the WFS is conjugated to height, ideally it should be able to sense high altitude turbulence without influence from the lower altitude turbulence. THe large field of view necessary on MCAO makes this technique ideal for solar astronomy.

LayerMCAO is written in Python and currently implements three AO components: 1) Shack Hartmann Wavefront Sensor 2) Telescope 3) Atmosphere. Most of the work was done on the WFS, which contains the simulation methods to generate and interpret *lenslet images*. Telescope and Atmosphere objects largely contains the specifications and turbulence information needed by the simulation methods. 

Lenslet image generation and interpretation are kept as two seperate objects and referenced by the WFS object. The only information exchanged between the two classes should be the lenslet images. This keeps physical simulation of atmospheric seeing separate from post-processing methods. 

Multiple methods for interpreting lenslet images are implemented and contained in the ImageInterpretor class. The ```all\_dimg\_to\_shifts``` method specifies the default method. At the moment, only one routine for generating lenslet images is implemented in the ImageGenerator class.

## Dependencies
1. MATLAB
2. Scipy
3. Numpy
4. Matplotlib

In ubuntu:

``` sudo apt-get install python-scipy python-matplotlib ```

## Getting Started

**Importing LayerMCAO classes:**
```
from Telescope import Telescope
from Atmosphere import Atmosphere
from WFS import WideFieldSHWFS
```
**Constructing LayerMCAO objects:**
```
tel = Telescope(2.5)
at = Atmosphere() 
at.create_default_screen(0, 0.15) # screens can be done later
wfs = WideFieldSHWFS(0, 16, 128, at, tel) 
```
**Run simulation:**
```
# This method generates SH-WFS lenslet-images and interpretes global pixel shift in each image
all_shifts = wfs.runWFS()
```

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

5. shifts: \[pixel\]

    - the overall 2D shift of a whole dimg 
    - format: (xShift, yShift)
    - type: pixel tuple

6. all_shifts: all lenslets' interpreted shift value

    - A collection of shift values
    - ndarray.shape = (num_lenslet, num_lenslet)
    - ndarray.dtype = ndarray (tuple)

7. Shifts / Tilts / Gradient
    - Shift: the number of pixels the whole IMAGE is shifted by in detector plane \[pixel\]
    - Tilt: the average angle the WAVEFRONT is tilted by the phase screens \[radians\]
    - Gradient: the average gradient over a portion of the PHASE SCREEN   \[radians/meter\]

8. ReconImg / RefImg
    - ReconImg: Reconstructed test image. Given many dimg, we try to reconstruct the original image without distortion. To be used as RefImg.
    - RefImg: Reference image as which image shift is measured.

9. c\_lenslet\_pos \[meters\]
    - position of conjugated lenslet
    - format: (x,y)


## Usage
**Display dmaps / dimgs**

More display methods in SHWFSDemonstrator class.
```
c_lenslet_pos = (0.2,0.5)
dmap = wfs.ImgSimulator.dmap(c_lenslet_pos)
SHWFSDemonstrator.display_dmap(dmap)
```

**Display the obscure values used in intermediate computations**

More display methods in SHWFSDemonstrator class.
```
wfs = wfs = WideFieldSHWFS(0, 16, 128, at, tel)
c_lenslet_pos = (0.2,0.5)
SHWFSDemonstrator.display_vignette(wfs, c_lenslet_pos)
```

**Evaluate quality of distortion / reconstruction / measurement**

More display methods in SHWFSDemonstrator class.
```
wfs = WideFieldSHWFS(0, 16, 128, at, tel)
SHWFSDemonstrator.eval_all_shifts(wfs)
```









