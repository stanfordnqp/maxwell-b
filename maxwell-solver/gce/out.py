""" Defines the Out class for GCE. """

from pycuda import gpuarray as ga
from pycuda.reduction import ReductionKernel
from gce.space import get_space_info
from gce.data import Data
import numpy as np
from mpi4py.MPI import COMM_WORLD as comm


class Out(Data):
    """ Out class for GCE. 
    
    Outs store reduction operations. Outs allow for reduction operations in
    the GCE framework by storing intermediary (y,z) values during a kernel
    operation, which are then reduced into a single value. See the Kernel
    class for additional information.

    Currently only the "sum" operation is supported.

    Derives from the Data class.

    New methods:
    __init__ -- Create an Out of a particular dtype and operation.
    get -- Redefined to retrieve the result of the reduction.

    """

    def __init__(self, dtype, op='sum'):
        """ Create an Out.

        Input variables
        dtype -- numpy dtype.

        Keyword variables
        op -- type of reduction operation to perform. Default='sum'.
            At this time, only the "sum" operation is supported.
        """

        self._set_gce_type('out')
        self._get_dtype(dtype) # Validate dtype.

        if op not in ('sum','prod'): # Validate op.
            raise TypeError('Invalid op.')
        self.op = op

        # Obtain the neutral value and store it in the result variable.
        neutral_val = {'sum': 0, 'prod': 1}

        # Create the intermediary values.
        shape = get_space_info()['shape']
        self.to_gpu((neutral_val[op] * \
                        np.ones((1, shape[1], shape[2]))).astype(self.dtype))

    def reduce(self):
        """ Compute the result. """
        self.result = comm.allreduce(ga.sum(self.data).get())

    def get(self):
        """ Redefine get() to return the result of the operation. """
        return self.result


def batch_reduce(*outs):
    """ Optimal (compared to self.reduce) when communication cost is latency bound. """
    results = comm.allreduce(np.array([ga.sum(out.data).get() for out in outs]))
    for k in range(len(outs)):
        outs[k].result = results[k]

