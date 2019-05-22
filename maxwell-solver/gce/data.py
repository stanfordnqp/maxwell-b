""" Defines the Data class for GCE. """

from gce.space import get_space_info
import numpy as np
from pycuda import gpuarray as ga

class Data:
    """ Generic data class for GCE. 
    
    The Grid, Const, and Out classes are derived from this class.

    Only supports datatypes: np.float32, np.float64, np.complex64, 
        and np.complex128.

    Functions:
    to_gpu -- Load a numpy array on to the GPU.
    get -- Transfer data back to host memory.

    Variables:
    data -- GPUArray instance.
    dtype -- Numpy datatype of the data.
    cuda_type -- Corresponding cuda type of the data.
    """

    def _get_dtype(self, dtype):
        """ Certify that the dtype is valid, and find the cuda datatype. """
        if dtype not in (np.int32, np.float32, np.float64, np.complex64, np.complex128):
            raise TypeError('Array is of an unsupported dtype.')
        
        self.dtype = dtype # The numpy datatype.

        cuda_dict = {np.float32: 'float', np.float64: 'double', \
                    np.int32: 'int', \
                    np.complex64: 'pycuda::complex<float>', \
                    np.complex128: 'pycuda::complex<double>'}

        self.cuda_type = cuda_dict[self.dtype] # Corresponding cuda datatype.

    def _set_gce_type(self, type):
        """ Set whether we have a Grid, Const, or Out. """
        if type in ('grid', 'const', 'out'):
            self.gce_type = type
        else:
            raise TypeError('Invalid gce type.')

    def to_gpu(self, array):
        """ Load data to the gpu. """
        self.data = ga.to_gpu(array)

    def get(self):
        """ Get data from the gpu. """
        return self.data.get()
