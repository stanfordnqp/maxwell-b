""" Defines the Grid class for GCE. """

from pycuda import gpuarray as ga
from pycuda import driver as drv
from gce.space import get_space_info
from gce.data import Data
import numpy as np
from mpi4py.MPI import COMM_WORLD as comm
import threading



class Grid(Data):
    """ Grid class for GCE. 
    
    Grids store modifiable information on a 3D rectangular grid.

    Grids may be split up along the x-axis for parallel processing. 
    If a particular Grid requires adjacent values in the x-direction,
    then the needed adjacent cells can be synchronized through use of the 
    x_overlap option and the synchronize(), synchronize_start(), and
    synchronize_isdone() functions.

    Derives from the Data class.

    New methods:
    __init__ -- Loads a (possibly empty) array onto the GPU.
    synchronize -- Used to synchronize Grids with non-zero x_overlap (blocking).
    synchronize_start -- Initiate non-blocking synchronization.
    synchronize_isdone -- Advance and complete non-blocking synchronization.

    New variables:
    none
    
    """


    def __init__(self, array_or_dtype, x_overlap=0):
        """ Create a spatial grid on the GPU(s).

        Input variables
        array_or_dtype -- can either be a numpy array of the same shape as
            the global space, or a numpy dtype. If a valid array is passed, 
            it will be loaded on to the GPU. If a dtype is passed, then
            an array of zeros, of that dtype will be loaded onto the GPU.

        Optional variables
        x_overlap -- the number of adjacent cells in either the negative or
            positive x-direction that need to simultaneously be accessed along
            with the current cell. Must be a non-negative integer. Default
            value is 0.

        """

        shape = get_space_info()['shape'] # Get the shape of the space.
        xr = get_space_info()['x_range'] # Get the local x_range.
        all_x_ranges = get_space_info()['all_x_ranges'] # Get the local x_range.
        local_shape = (xr[1]-xr[0], shape[1], shape[2])

        self._set_gce_type('grid') # Set the gce type to grid.

        # Make sure overlap option is valid.
        if type(x_overlap) is not int:
            raise TypeError('x_overlap must be an integer.')
        elif x_overlap < 0:
            raise TypeError('x_overlap must be a non-negative integer.')

        if comm.rank == 0:
            # Process the array_or_dtype input variable.
            if type(array_or_dtype) is np.ndarray: # Input is an array.
                array = array_or_dtype

                # Make sure the array is of the correct shape.
                if array.shape != shape:
                    raise TypeError('Shape of array does not match shape of space.')

                # Make sure the array is of a valid datatype.
                self._get_dtype(array.dtype.type)


            elif type(array_or_dtype) is type: # Input is a datatype.
                self._get_dtype(array_or_dtype) # Validate the dtype.
                array = np.zeros(shape, dtype=self.dtype) # Make a zeros array.

            else: # Invalid input.
                raise TypeError('Input variable must be a numpy array or dtype')

            # Prepare array to be scattered.
            array = [array[r[0]:r[1],:,:] for r in all_x_ranges]

        else:
            array = None

        array = comm.scatter(array)
        self._get_dtype(array.dtype.type)

#         # Narrow down the array to local x_range.
#         array = array[xr[0]:xr[1],:,:]

        # Add padding to array, if needed.
        self._xlap = x_overlap
        if self._xlap is not 0:
            padding = np.empty((self._xlap,) + shape[1:3], dtype=array.dtype)
            array = np.concatenate((padding, array, padding), axis=0)

        self.to_gpu(array) # Load onto device.

        # Determine information needed for synchronization.
        if self._xlap is not 0:
            # Calculates the pointer to the x offset in a grid.
            ptr_dx = lambda x_pos: self.data.ptr + self.data.dtype.itemsize * \
                                        x_pos * shape[1] * shape[2]
            
            # Pointers to different sections of the grid that are relevant
            # for synchronization.
            self._sync_ptrs = { 'forw_src': ptr_dx(xr[1]-xr[0]), \
                                'back_dest': ptr_dx(0), \
                                'back_src': ptr_dx(self._xlap), \
                                'forw_dest': ptr_dx(xr[1]-xr[0] + self._xlap)}

            # Buffers used during synchronization.
            self._sync_buffers = [drv.pagelocked_empty( \
                                    (self._xlap, shape[1], shape[2]), \
                                    self.dtype) for k in range(4)]

            # Streams used during synchronization.
            self._sync_streams = [drv.Stream() for k in range(4)]

            # Used to identify neighboring MPI nodes with whom to synchronize.
            self._sync_adj = get_space_info()['mpi_adj']

            # Offset in bytes to the true start of the grid.
            # This is used to "hide" overlap areas from the kernel.
            self._xlap_offset = self.data.dtype.itemsize * \
                                self._xlap * shape[1] * shape[2]

            self.synchronize() # Synchronize the grid.
            comm.Barrier() # Wait for all grids to synchronize before proceeding.

    def get(self):
        """ Redefined so that we don't get overlap data. """
        # Get our section of the grid (excluding overlap).
        if self._xlap is 0:
            data = self.data.get()
        else:
            data = self.data.get()[self._xlap:-self._xlap,:,:]
        
#         return np.concatenate(comm.allgather(data), axis=0) # Super-simple.

        result = comm.gather(data) # Gather all peices to root.
        if comm.Get_rank() == 0:
            # Root node glues everything together.
            return np.concatenate(result, axis=0) 
        else: 
            return None

    def _get_raw(self):
        """ Output even the overlap data. Just for debugging/testing. """
        return self.data.get()

    def synchronize(self):
        """ Blocking synchronization.  """

        if self._xlap is 0:
            raise TypeError('No need to synchronize Grid with no overlaps.')

        self.synchronize_start()
        while not self.synchronize_isdone():
            pass

    def synchronize_start(self):
        """ Start the synchronization process. """

        # Use shorter, easier names for class variables.
        bufs = self._sync_buffers
        ptrs = self._sync_ptrs
        streams = self._sync_streams
        adj = self._sync_adj

        # Start the transfer operations needed.
        self._sync_tags = [mpi_tag() for k in range(2)] # Mpi message tags.

        # Forward send.
        drv.memcpy_dtoh_async(bufs[0], ptrs['forw_src'], stream=streams[0]) 

        # Backward send.
        drv.memcpy_dtoh_async(bufs[1], ptrs['back_src'], stream=streams[1]) 

        # Forward receive.
        self._sync_req_forw = comm.Irecv(bufs[2], source=adj['back'], \
                                            tag=self._sync_tags[0])

        # Backward receive.
        self._sync_req_back = comm.Irecv(bufs[3], source=adj['forw'], \
                                            tag=self._sync_tags[1])

        # Signalling variables needed to complete transfers.
        self._sync_part2_start = [False, False, False, False]


    def synchronize_isdone(self):
        """ Complete synchronization process. """

        # Use shorter, easier names for class variables.
        bufs = self._sync_buffers
        ptrs = self._sync_ptrs
        streams = self._sync_streams
        adj = self._sync_adj
        part2_start = self._sync_part2_start 
        is_done = [False, False, False, False]

        # Forward send.
        if streams[0].is_done(): # Device-to-host copy completed.
            if not part2_start[0]: # Initialize MPI send.
                comm.Isend(bufs[0], dest=adj['forw'], tag=self._sync_tags[0])
                part2_start[0] = True
                is_done[0] = True
            else: # No more work to do.
                is_done[0] = True

        # Backward send.
        if streams[1].is_done(): # Device-to-host copy completed.
            if not part2_start[1]: # Initialize MPI send.
                comm.Isend(bufs[1], dest=adj['back'], tag=self._sync_tags[1])
                part2_start[1] = True
                is_done[1] = True
            else: # No more work to do.
                is_done[1] = True

        # Forward receive.
        if self._sync_req_forw.Test(): # MPI receive completed.
            if not part2_start[2]: # Initialize host-to-device copy.
                drv.memcpy_htod_async(ptrs['back_dest'], bufs[2], \
                                        stream=streams[2]) # Host-to-device.
                part2_start[2] = True
            elif streams[2].is_done(): # Host-to-device copy completed.
                is_done[2] = True

        # Backward receive.
        if self._sync_req_back.Test(): # MPI receive completed.
            if not part2_start[3]: # Initialize host-to-device copy.
                drv.memcpy_htod_async(ptrs['forw_dest'], bufs[3], \
                                        stream=streams[3]) # Host-to-device.
                part2_start[3] = True
            elif streams[3].is_done(): # Host-to-device copy completed.
                is_done[3] = True
        # print '~', is_done[0:4],
        # Return true only when all four transfers are complete.
        return all(is_done) 


__MPI_TAG_NUM = 0 # Global variable used to generate unique mpi tags.

def mpi_tag():
    """ Get a new, unique mpi tag number. """
    global __MPI_TAG_NUM # Get the global variable.
    tag = __MPI_TAG_NUM # The variable to return.
    __MPI_TAG_NUM += 1 
    return tag
