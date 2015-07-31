__author__ = 'tpl'
import numpy as np

class Telescope(object):
    def __init__(self, dia):
        self.pupil_diameter = dia # [meters]
        self.primary_lens_f = 0.8 # [meters]
        self.secondary_lens_f = 0.2 # [meters]
        self.Mfactor = self.primary_lens_f/self.secondary_lens_f
        """primary_lens_f/self.secondary_lens_f"""
        self.field_of_view = 0.02 # [radians]
        self._field_diaphragm = np.tan(self.field_of_view/2.0)*self.primary_lens_f*2.0 # [meters]
