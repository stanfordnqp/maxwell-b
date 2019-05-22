""" Defines the Kernel class for GCE. """
from pycuda import compiler
from pycuda import driver as drv
from jinja2 import Environment, PackageLoader
from gce.space import get_space_info
from gce.out import batch_reduce
import numpy as np
from mpi4py.MPI import COMM_WORLD as comm

# Load the jinja environment when the module is loaded.
_template_file = 'kernel.cu'
_jinja_env = Environment(loader=PackageLoader(__name__, '.'))


class Kernel:
    """ Create an executable kernel for GCE.

    A Kernel executable allows for the modification of Grid objects and the 
    computation of Outs. Kernels accept Grid, Const, Out, and certain numpy
    scalar objects as their input. 

    Kernels work by traversing the 3D space in the x-direction and executing
    user-specified cuda code at every grid point. For more information on the
    conventions and available tools for defining Kernels, please see the 
    KERNEL_DOC file.

    Additionally, Kernels will self-optimize runtime parameters. Such parmaters
    include only block size for now though.

    Methods:
    __init__ -- Define the executable kernel.
    __call__ -- Execute the kernel.

    Example usage:
    fun = Kernel(((x, 'x'), (y, 'y')), code)
    fun()
    """

    def __init__(self, code, *vars, **kwargs):
        """ Prepare a cuda function that will execute on the GCE space.

        Input variables:
        code -- The looped cuda code to be executed.
        vars -- (name, gce_type, numpy_type) of the input arguments.

        Keyword variables:
        pre_loop -- Cuda code that is executed before the loop code.
        shape_filter -- Can be either 'all', 'skinny', or 'square'.
        padding -- (yn, yp, zn, zp), describes the number of "extra" threads
            to be run on the border of each thread block.
        smem_per_thread -- Number of bytes of shared memory needed by a thread.
        """

        # Make sure there are no extraneous keyword arguments.
        if any([key not in \
                ('pre_loop', 'shape_filter', 'padding', 'smem_per_thread')
                for key in kwargs.keys()]):
            raise TypeError('Invalid key used.')

        # Process keyword arguments.
        pre_code = kwargs.get('pre_loop', '')
        shape_filter = kwargs.get('shape_filter', 'skinny')
        padding = kwargs.get('padding', (0, 0, 0, 0))
        smem_per_thread = kwargs.get('smem_per_thread', 0)

        # Dictionary for conversion from numpy to cuda types.
        cuda_types = {np.float32: 'float', np.float64: 'double', \
                        np.int32: 'int', \
                        np.complex64: 'pycuda::complex<float>', \
                        np.complex128: 'pycuda::complex<double>'}
        # Dictionary for conversion from numpy to alternate type for Consts.
        alt_types = {np.float32: 'float', np.float64: 'double', \
                        np.complex64: 'float2', np.complex128: 'double2'}

        # Process vars.
        params = [{'name': v[0], \
                'gce_type': v[1], \
                'dtype': v[2], \
                'cuda_type': cuda_types[v[2]]} for v in vars]

        # Get the template and render it using jinja2.
        shape = get_space_info()['shape']  # Shape of the space.
        template = _jinja_env.get_template(_template_file)
        cuda_source = template.render(  params=params, \
                                        padding=padding, \
                                        dims =get_space_info()['shape'], \
                                        x_range=get_space_info()['x_range'], \
                                        preloop_code=pre_code, \
                                        loop_code=code, \
                                        flat_tag='_f')

        # Compile the code into a callable cuda function.
        mod = compiler.SourceModule(cuda_source)
        # mod = compiler.SourceModule(cuda_source, options=['-Xptxas', '-dlcm=cg']) # Global skips L1 cache.
        self.fun = mod.get_function('_gce_kernel')

        # Prefer 48KB of L1 cache when possible.
        self.fun.set_cache_config(drv.func_cache.PREFER_L1)

        # Get address of global variable in module.
        # Note: contains a work-around for problems with complex types.
        my_get_global = lambda name: mod.get_global('_' + name + '_temp')

        # Useful information about the kernel.
        self._kernel_info = {'max_threads': self.fun.max_threads_per_block, \
                            'const_bytes': self.fun.const_size_bytes, \
                            'local_bytes': self.fun.local_size_bytes, \
                            'num_regs': self.fun.num_regs}

        # Get some valid execution configurations.
        self.exec_configs = self._get_exec_configs( \
                                        self.fun.max_threads_per_block, \
                                        padding, smem_per_thread, shape_filter)

        # Prepare the function by telling pycuda the types of the inputs.
        arg_types = []
        for p in params:
            if p['gce_type'] is 'number':
                arg_types.append(p['dtype'])
#             elif p['gce_type'] is 'const':
#                 arg_types.append(p['dtype'])
#                 # pass # Consts don't actually get passed in.
            else:
                arg_types.append(np.intp)
        self.fun.prepare([np.int32, np.int32] + arg_types)

        # Define the function which we will use to execute the kernel.
        # TODO: Make a shortcut version with lower overhead.
        # Used for asynchronous execution and timing.
        stream = drv.Stream()
        start, start2, pad_done, sync_done, comp_done, all_done = \
            [drv.Event() for k in range(6)]

        # Kernel execution over a range of x-values.
        def execute_range(x_start, x_end, gpu_params, cfg, stream):
            """ Defines asynchronous kernel execution for a range of x. """
            self.fun.prepared_async_call( \
                cfg['grid_shape'][::-1], \
                cfg['block_shape'][::-1] + (1,), \
                stream, \
                *([np.int32(x_start), np.int32(x_end)] + gpu_params), \
                shared_size=cfg['smem_size'])

        x_start, x_end = get_space_info()['x_range']  # This node's range.

        def execute(cfg, *args, **kwargs):

            # Parse keyword arguments.
            post_sync_grids = kwargs.get('post_sync', None)

            # Parse the inputs.
            gpu_params = []
            for k in range(len(params)):
                if params[k]['gce_type'] is 'number':
                    gpu_params.append(params[k]['dtype'](args[k]))
                elif params[k]['gce_type'] is 'const':  # Load Const.
                    gpu_params.append(args[k].data.ptr)
                    # Const no longer actually "const" in cuda code.


#                     d_ptr, size_in_bytes = my_get_global(params[k]['name'])
#                     drv.memcpy_dtod(d_ptr, args[k].data.gpudata, size_in_bytes)
                elif params[k]['gce_type'] is 'grid':
                    if args[k]._xlap is 0:
                        gpu_params.append(args[k].data.ptr)
                    else:
                        gpu_params.append(args[k].data.ptr + \
                                            args[k]._xlap_offset)
                elif params[k]['gce_type'] is 'out':
                    args[k].data.fill(args[k].dtype(0))  # Initialize the Out.
                    gpu_params.append(args[k].data.ptr)
                else:
                    raise TypeError('Invalid input type.')

            # See if we need to synchronize grids after kernel execution.
            if post_sync_grids is None:
                sync_pad = 0
            else:
                sync_pad = max([g._xlap for g in post_sync_grids])

            start2.record(stream)
            comm.Barrier()
            start.record(stream)

            # Execute kernel in padded regions first.
            execute_range(x_start, x_start + sync_pad, gpu_params, cfg, stream)
            execute_range(x_end - sync_pad, x_end, gpu_params, cfg, stream)
            pad_done.record(stream)  # Just for timing purposes.
            stream.synchronize()  # Wait for execution to finish.

            # Begin kernel execution in remaining "core" region.
            execute_range(x_start + sync_pad, x_end - sync_pad, gpu_params,
                          cfg, stream)
            comp_done.record(stream)  # Timing only.

            # While core kernel is executing, perform synchronization.
            if post_sync_grids is not None:  # Synchronization needed.
                for grid in post_sync_grids:
                    grid.synchronize_start()  # Start synchronization.

                # Keep on checking until everything is done.
                while not (all([grid.synchronize_isdone() \
                                for grid in post_sync_grids]) and \
                        stream.is_done()):
                    pass

            else:  # Nothing to synchronize.
                stream.synchronize()  # Just wait for execution to finish.

            sync_done.record()  # Timing.

            # Obtain the result for all Outs.
            batch_reduce(*[args[k] for k in range(len(params)) \
                                if params[k]['gce_type'] is 'out'])
            all_done.record()  # Timing.
            all_done.synchronize()

            return comp_done.time_since(
                start)  # Return time needed to execute the function.

        self.execute = execute  # Save execution function in Kernel instance.
        self.min_exec_time = float('inf')  # Stores the fastest execution time.

    def __call__(self, *args, **kwargs):
        """ Execute the kernel. 

        Each valid execution configuration will be tried once, and then the
        fastest configuration will be used for all remaining calls.
        """
        if self.exec_configs:  # As long as list is not empty, choose from list.
            cfg = self.exec_configs.pop()  # Choose execution configuration.

            # Execute.
            exec_time = self.execute(cfg, *args, **kwargs)

            # Check if this was the fastest execution to-date.
            if exec_time < self.min_exec_time:  # Found a new fastest config.
                self.min_exec_time = exec_time
                self.fastest_cfg = cfg

        else:  # If config list empty, go with the fastest configuration found.
            cfg = self.fastest_cfg
            exec_time = self.execute(cfg, *args, **kwargs)

        # Return results.
        return exec_time, cfg

    def _get_exec_configs(self, threads_max, padding, smem_per_thread, \
                        shape_filter):
        """ Find all valid execution configurations. """

        # Padding of the kernel.
        y_pad = sum(padding[0:2])
        z_pad = sum(padding[2:4])

        # Shared memory requirements.
        smem_size = lambda b_shape: smem_per_thread * \
                                            (b_shape[0] * b_shape[1])

        # The kind of shapes that we are interested in.
        if shape_filter is 'skinny':  # Only z-dominant shapes.
            my_filter = lambda b_shape: (b_shape[0] < b_shape[1]) and \
     (b_shape[1] > 8) and ((b_shape[1] % 16) == 0)
        elif shape_filter is 'square':  # Only square-ish shapes.
            my_filter = lambda b_shape: (b_shape[0] < 2 * b_shape[1]) and \
                                        (b_shape[1] < 2 * b_shape[0]) and \
     (b_shape[0] > 8) and \
     (b_shape[1] > 8)
        elif shape_filter is 'all':  # All shapes okay.
            my_filter = lambda b_shape: b_shape[1] > 1  # Must be greater than 1.
        else:
            raise TypeError('Unrecognized shape filter.')

        # Function defining valid block shapes.
        smem_max = get_space_info()['max_shared_mem']
        is_valid_shape = lambda b_shape: (smem_size(b_shape) < smem_max) and \
                                            my_filter(b_shape) and \
                                            (b_shape[0] * b_shape[1]) <= \
                                                threads_max

        # Create a list of all valid block shapes.
        valid_block_shapes = []
        z_max = get_space_info()['max_block_z']
        y_max = get_space_info()['max_block_y']
        for j in range(y_pad + 1, y_max + 1):
            for k in range(z_pad + 1, z_max + 1):
                if is_valid_shape((j, k)):
                    valid_block_shapes.append((j,
                                               k))  # Block shape is (yy,zz).

        # A hack for profiling
        # valid_block_shapes = ((31,16),)
        # valid_block_shapes = ((17,22),)

        if not valid_block_shapes:  # Make sure the list is not empty.
            raise TypeError('No valid shapes found.')

        # Create a list of all possible execution configurations.
        # Note that the convention for both block_shape and grid_shape is
        # (yy,zz). Among other things, this leads to the (slightly)
        # tricky computation of grid_shape.
        sp_shape = get_space_info()['shape']  # Shape of the space.
        return [{   'block_shape': vbs, \
                    'grid_shape': (int((sp_shape[1]-1)/(vbs[0]-y_pad)) + 1, \
                                    int((sp_shape[2]-1)/(vbs[1]-z_pad)) + 1), \
                    'smem_size': smem_size(vbs)}
                for vbs in valid_block_shapes]
