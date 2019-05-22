""" Defines the Const class for GCE. """

from pycuda import gpuarray as ga
from gce.data import Data
import numpy as np

class Const(Data):
    """ Const class for GCE. 
    
    Used to store globally accessible, but unchangeable data.

    Derives from the Data class.

    New methods:
    __init__ -- Store an array as a Const on the GPU. 
    """

    def __init__(self, array):
        """ Create a Const.

        Input variables
        array -- a numpy array of valid dtype.
        """

        self._set_gce_type('const')
        if type(array) is not np.ndarray: # Make sure we actually got an array.
                raise TypeError('Array must be a numpy ndarray.')

        self._get_dtype(array.dtype.type) # Validate the array's dtype.
        self.to_gpu(array) # Load onto device.

