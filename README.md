# LayerMCAO
This project aims to

1. Demonstrate the principles of layer-oriented MCAO 
2. Test methods and algorithms for use in layer-oriented MCAO

## Usage
1. Choose a function from "Evaluation.py" module to run. Docstring should explain the principles behind the demonstration
2. Choose a function from SHWFSDemonstrator to run. 
2. Choose a function from PhaseScreenDemonstrator to run. 

## Naming Conventions
1. dmap: distortion map \[pixel\]

	- The shifts in units of pixels to be applied to test image to generate the distorted image (dimg)
    - ndarray.shape = (2, pixel_lenslet, pixel_lenslet)
    - ndarray.dtype = float

2. all_dmap: all lenslets' distortion map

	- A collection of dmaps
    - ndarray.shape = (num_lenslet, num_lenslet)
    - ndarray.dtype = ndarray

3. dimg: distorted image \[pixel\]

	- The image behind a SH lenslet as distorted by atmosphere
    - ndarray.shape = (pixel_lenslet, pixel_lenslet)
    - ndarray.dtype = float

4. all_dimg: all lenslets' distorted image

	- A collection of dimgs
    - ndarray.shape = (num_lenslet, num_lenslet)
    - ndarray.dtype = ndarray

5. Shifts / Tilts / Gradient
    - Shift: the number of pixels the point IMAGE is shifted by on detector plane \[pixel\]
    - Tilt: the overage angle the WAVEFRONT is tilted by the phase screens \[radians\]
    - Gradient: the average gradient over a portion of the PHASE SCREEN   \[radians/meter\]






