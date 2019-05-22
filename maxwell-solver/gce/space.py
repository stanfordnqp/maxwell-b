""" Used to setup the global space for GCE. """

from mpi4py import MPI
from mpi4py.MPI import COMM_WORLD as comm
from pycuda import driver 


def _init_gpu(comm):
    """ Chooses a gpu and creates a context on it. """
    # Find out how many GPUs are available to us on this node.
    driver.init()
    num_gpus = driver.Device.count()

    # Figure out the names of the other hosts.
    rank = comm.Get_rank() # Find out which process I am.
    name = MPI.Get_processor_name() # The name of my node.
    hosts = comm.allgather(name) # Get the names of all the other hosts

    # Find out which GPU to take (by precedence).
    gpu_id = hosts[0:rank].count(name)
    if gpu_id >= num_gpus:
        raise TypeError('No GPU available.')

    
    # Create a context on the appropriate device.
    for k in range(num_gpus):
        try:
            device = driver.Device((gpu_id + k) % num_gpus)
            context = device.make_context()
        except:
            continue
        else:
#             print "On %s: process %d taking gpu %d of %d.\n" % \
#                 (name, rank, gpu_id+k, num_gpus)
            break

    return device, context # Return device and context.

# Global variable for the global space.
# The leading double underscore should prevent outside modules from accessing
# this variable.
__GLOBAL_SPACE = None 

# Upon module initialization, claim a GPU and create a context on it.
__DEVICE, __CONTEXT = _init_gpu(comm)

import atexit
atexit.register(__CONTEXT.pop)

def initialize_space(shape):
    """ Form the space. """
    global __GLOBAL_SPACE, __DEVICE, __CONTEXT 
    __GLOBAL_SPACE = __Space(shape, __DEVICE, __CONTEXT)

def get_space_info():
    """ Returns all the info needed about a space. """
    if __GLOBAL_SPACE is None: # Global space not yet initialized.
        raise TypeError('The global space is not initialized.')
    else:
        return __GLOBAL_SPACE.get_info()

def print_space_info():
    """ Prints out information about the space. """
    if __GLOBAL_SPACE is None: # Global space not yet initialized.
        raise TypeError('The global space is not initialized.')
    info = __GLOBAL_SPACE.get_info()
    for name, val in info.iteritems():
        print(name, val)

# def destroy_space():
#     """ Set global space to none. """
#     global __GLOBAL_SPACE 
#     __GLOBAL_SPACE.__del__()
#     __GLOBAL_SPACE = None

class __Space():
    """ Space forms the 3D context for Grid and Kernel objects. 
    
    As of the current implementation, it is assumed that only one space
    will be created, and that all Const, Grid, and Kernel objects will
    operate on that space.

    """

    def __init__(self, shape, device, context):
        """ Constructor for the Space class. 

        Input variables
        shape -- Three-element tuple of positive integers defining the size of
            the space in the x-, y-, and z-directions.

        """

        # Make sure shape has exactly three elements.
        if len(shape) is not 3:
            raise TypeError('Shape must have exactly three elements.')

        # Make sure they are all integers.
        if any([type(s) is not int for s in shape]):
            raise TypeError('Shape must have only integer elements.')

        # Make sure all elements are positive.
        if any([s < 1 for s in shape]):
            raise TypeError('Shape must have only integer elements.')

#         # Make sure stencil is a single, non-negative integer.
#         if (type(stencil) is not int) or (stencil < 0):
#             raise TypeError('Stencil must be a non-negative scalar integer.')
# 
        # Initialize the space.
        self.shape = shape

        # Get MPI information.
        rank = comm.Get_rank()
        size = comm.Get_size()

        # Nodes to pass forward and backward (along x) to.
        self.mpi_adj = {'forw': (rank+1)%size, 'back': (rank-1)%size}	

        # Grid is too small to be partitioned.
        if (size > self.shape[0]): 
            raise TypeError('Shape is too short along x to be partitioned.')

        # Create the context on the appropriate GPU.
        # self.device, self.context = self._init_gpu(comm)
        self.device = device
        self.context = context

        # Partition the space.
        # Each space is responsible for field[x_range[0]:x_range[1],:,:].
        get_x_range = lambda r: (int(self.shape[0] * (float(r) / size)), \
                                int(self.shape[0] * (float(r+1) / size)))
        self.x_range = get_x_range(rank)

        self.all_x_ranges = [get_x_range(r) for r in range(size)]


#     def __del__(self):
#         """ Pop the cuda context on cleanup. """
#         # Make sure the space was actually initialized.
#         if hasattr(self, 'context'): 
#             self.context.pop()

    def get_info(self):
        """ Return information about the space as a dict. """
        return {'shape': self.shape, \
                'x_range': self.x_range, \
                'all_x_ranges': self.all_x_ranges, \
                'mpi_adj': self.mpi_adj, \
                'max_shared_mem': self.device.max_shared_memory_per_block, \
                'max_block_z': self.device.max_block_dim_x, \
                'max_block_y': self.device.max_block_dim_y, \
                'max_threads': self.device.max_threads_per_block, \
                'mem_bandwidth': 1000 * self.device.memory_clock_rate/8 * \
                    self.device.global_memory_bus_width * 2, \
                'max_registers': self.device.max_registers_per_block, \
                'async_engine_count': self.device.async_engine_count, \
                'ecc_enabled': self.device.ecc_enabled, \
                }
