""" Implements the operations needed to solve Maxwell's equations in 3D. """

import numpy as np
import copy
from jinja2 import Environment, PackageLoader, Template
from gce.space import initialize_space, get_space_info
from gce.grid import Grid
from gce.const import Const
from gce.out import Out
from gce.kernel import Kernel
from typing import List
from mpi4py.MPI import COMM_WORLD as comm

# Execute when module is loaded.
# Load the jinja environment.
jinja_env = Environment(loader=PackageLoader(__name__, 'kernels'))


def conditioners(params, dtype):
    """ Form the functions for both the preconditioner and postconditioner. """

    #
    #     # Code for the post step function.
    #     code = """
    #         if (_in_global) {
    #             Ex(0,0,0) *= tx1(_X) * ty0(_Y) * tz0(_Z);
    #             Ey(0,0,0) *= tx0(_X) * ty1(_Y) * tz0(_Z);
    #             Ez(0,0,0) *= tx0(_X) * ty0(_Y) * tz1(_Z);
    #         } """
    def reshaper(f):
        for k in range(3):
            new_shape = [1, 1, 1]
            new_shape[k] = f[k].size
            f[k] = f[k].reshape(new_shape)
        return f

    # Consts that are used.
    sqrt_sc_pml_0 = reshaper([dtype(np.sqrt(s)**1) for s in params['s']])
    sqrt_sc_pml_1 = reshaper([dtype(np.sqrt(t)**1) for t in params['t']])
    inv_sqrt_sc_pml_0 = reshaper([dtype(np.sqrt(s)**-1) for s in params['s']])
    inv_sqrt_sc_pml_1 = reshaper([dtype(np.sqrt(t)**-1) for t in params['t']])

    # Define the actual functions.

    def apply_cond(x, t0, t1):
        x[0] *= t1[0] * t0[1] * t0[2]
        x[1] *= t0[0] * t1[1] * t0[2]
        x[2] *= t0[0] * t0[1] * t1[2]
        return x

    def pre_step(x):
        return apply_cond(x, sqrt_sc_pml_0, sqrt_sc_pml_1)

    def post_step(x):
        return apply_cond(x, inv_sqrt_sc_pml_0, inv_sqrt_sc_pml_1)

    return pre_step, post_step


def _get_cuda_type(dtype):
    """ Convert numpy type into cuda type. """
    if dtype is np.complex64:
        return 'pycuda::complex<float>'
    elif dtype is np.complex128:
        return 'pycuda::complex<double>'
    else:
        raise TypeError('Invalid dtype.')


# GPU operations
#---------
def make_gpu_copy(dtype):
    """ Returns a function that does B=A """
    # Code for the rho step function.
    code = Template("""
        if (_in_global) {
            Bx(0,0,0) = Ax(0,0,0);
            By(0,0,0) = Ay(0,0,0);
            Bz(0,0,0) = Az(0,0,0);
        } """).render(type=_get_cuda_type(dtype))

    # Compile the code using gce.Kernel
    grid_names = [A + i for A in ['A', 'B'] for i in ['x', 'y', 'z']]
    copy_fun = Kernel(code, \
                    *[(name, 'grid', dtype) for name in grid_names], \
                    shape_filter='skinny')

    # Define the actual function.
    def gpu_copy(A, B):
        copy_fun( \
                         *( A + B), \
                         post_sync=B) # r must be post-synced for upcoming alpha step.

    return gpu_copy


def make_gpu_norm(dtype):
    """ Returns a function c=vec_norm(A) that does c=sqrt(A'A) """
    # GPU Code in gce.kernel.
    code = Template("""
        if (_in_global) {
            norm_a += conj(Ax(0,0,0))*Ax(0,0,0);
            norm_a += conj(Ay(0,0,0))*Ay(0,0,0);
            norm_a += conj(Az(0,0,0))*Az(0,0,0);
        } """).render(type=_get_cuda_type(dtype))

    # Compile the code using gce.Kernel
    grid_names = [A + i for A in ['A'] for i in ['x', 'y', 'z']]
    prod_fun = Kernel(code, \
                    ('norm_a', 'out', dtype), \
                    *[(name, 'grid', dtype) for name in grid_names], \
                    shape_filter='skinny')
    norm_a = Out(dtype)

    # Define the actual function.
    def gpu_norm(A):
        prod_fun( norm_a,\
                  *( A )) # remove the post_sync
        return np.sqrt(norm_a.get())

    return gpu_norm


def make_gpu_scale(dtype):
    """ Returns a function scale(A, a) that does A=aA """
    # Code for the rho step function.
    code = Template("""
        if (_in_global) {
            Ax(0,0,0) = a*Ax(0,0,0);
            Ay(0,0,0) = a*Ay(0,0,0);
            Az(0,0,0) = a*Az(0,0,0);
        } """).render(type=_get_cuda_type(dtype))

    # Compile the code using gce.Kernel
    grid_names = [A + i for A in ['A'] for i in ['x', 'y', 'z']]
    Sum_fun = Kernel(code, \
                    ('a', 'number', dtype), \
                    *[(name, 'grid', dtype) for name in grid_names], \
                    shape_filter='skinny')

    # Define the actual function.
    def gpu_scale(A, a):
        Sum_fun(dtype(a), \
                *( A ), \
                 post_sync=A) # r must be post-synced for upcoming alpha step.

    return gpu_scale


def make_gpu_conj(dtype):
    """ Returns a function that does B=conj(A) """
    # Code for the rho step function.
    code = Template("""
        if (_in_global) {
            Bx(0,0,0) = conj(Ax(0,0,0));
            By(0,0,0) = conj(Ay(0,0,0));
            Bz(0,0,0) = conj(Az(0,0,0));
        } """).render(type=_get_cuda_type(dtype))

    # Compile the code using gce.Kernel
    grid_names = [A + i for A in ['A', 'B'] for i in ['x', 'y', 'z']]
    conj_fun = Kernel(code, \
                    *[(name, 'grid', dtype) for name in grid_names], \
                    shape_filter='skinny')

    # Define the actual function.
    def gpu_conj(A, B):
        conj_fun( \
                         *( A + B), \
                         post_sync=B) # r must be post-synced for upcoming alpha step.

    return gpu_conj


def make_gpu_dot(dtype):
    """ Returns a function c=vec_dot(A, B) that does c=A'B """
    # GPU Code in gce.kernel.
    code = Template("""
        if (_in_global) {
            dot_ab += conj(Ax(0,0,0))*Bx(0,0,0);
            dot_ab += conj(Ay(0,0,0))*By(0,0,0);
            dot_ab += conj(Az(0,0,0))*Bz(0,0,0);
        } """).render(type=_get_cuda_type(dtype))

    # Compile the code using gce.Kernel
    grid_names = [A + i for A in ['A', 'B'] for i in ['x', 'y', 'z']]
    prod_fun = Kernel(code, \
                    ('dot_ab', 'out', dtype), \
                    *[(name, 'grid', dtype) for name in grid_names], \
                    shape_filter='skinny')

    dot_ab = Out(dtype)

    # Define the actual function.
    def gpu_dot(A, B):
        prod_fun( dot_ab,\
                  *( A + B))
        return dot_ab.get()

    return gpu_dot


def make_gpu_addvec(dtype):
    """ Returns a function vec_addvec(A, b, B) that does A=A+bB """
    # GPU Code in gce.Kernel
    code = Template("""
        if (_in_global) {
            Ax(0,0,0) = Ax(0,0,0) + b*Bx(0,0,0);
            Ay(0,0,0) = Ay(0,0,0) + b*By(0,0,0);
            Az(0,0,0) = Az(0,0,0) + b*Bz(0,0,0);
        } """).render(type=_get_cuda_type(dtype))

    # Compile the code using gce.Kernel
    grid_names = [A + i for A in ['A', 'B'] for i in ['x', 'y', 'z']]
    Sum_fun = Kernel(code, \
                    ('b', 'number', dtype), \
                    *[(name, 'grid', dtype) for name in grid_names], \
                    shape_filter='skinny')

    # Define the actual function.
    def gpu_addvec(A, b, B):
        Sum_fun( dtype(b), \
                *( A + B ), \
                 post_sync=A) # r must be post-synced for upcoming alpha step.

    return gpu_addvec


def make_gpu_scaled_copy(dtype):
    """ Returns a function vec_scaled_copy(A, a, B) that does B=aA """
    # GPU code for the Kernel
    code = Template("""
        if (_in_global) {
            Bx(0,0,0) = a*Ax(0,0,0);
            By(0,0,0) = a*Ay(0,0,0);
            Bz(0,0,0) = a*Az(0,0,0);
        } """).render(type=_get_cuda_type(dtype))

    # Compile the code using gce.Kernel
    grid_names = [A + i for A in ['A', 'B'] for i in ['x', 'y', 'z']]
    Sum_fun = Kernel( code, \
                      ('a', 'number', dtype), \
                      *[(name, 'grid', dtype) for name in grid_names], \
                      shape_filter='skinny')

    # Define the actual function.
    def gpu_scaled_copy(A, a, B):
        Sum_fun(dtype(a), \
                *( A + B ), \
                post_sync=B)

    return gpu_scaled_copy


def make_gpu_sum(dtype):
    """ Returns a function that does aA+bB=C """
    # Code for the rho step function.
    code = Template("""
        if (_in_global) {
            Cx(0,0,0) = a*Ax(0,0,0) + b*Bx(0,0,0);
            Cy(0,0,0) = a*Ay(0,0,0) + b*By(0,0,0);
            Cz(0,0,0) = a*Az(0,0,0) + b*Bz(0,0,0);
        } """).render(type=_get_cuda_type(dtype))

    # Compile the code using gce.Kernel
    grid_names = [A + i for A in ['A', 'B', 'C'] for i in ['x', 'y', 'z']]
    Sum_fun = Kernel(code, \
                    ('a', 'number', dtype), \
                    ('b', 'number', dtype), \
                    *[(name, 'grid', dtype) for name in grid_names], \
                    shape_filter='skinny')

    # Define the actual function.
    def gpu_sum(a, b, A, B, C):
        Sum_fun(dtype(a), dtype(b), \
                         *( A + B + C), \
                         post_sync=C) # r must be post-synced for upcoming alpha step.

    return gpu_sum


def make_gpu_weighted_sum(dtype):
    """ Return weighted sum function """
    # returns function vec_weighted_sum(V,y,U) that will do:
    #   U = y1*V1 + y2*V2 + ... + yn*Vn
    # Note: you can not have U in V !!!
    gpu_scaled_copy = make_gpu_scaled_copy(dtype)
    gpu_addvec = make_gpu_addvec(dtype)

    def gpu_weighted_sum(L: List[Grid], y: np.ndarray, A):
        gpu_scaled_copy(L[0], y[0], A)
        for i in range(1, len(y)):
            gpu_addvec(A, y[i], L[i])

    return gpu_weighted_sum


def make_gpu_fdfd_residual(params, dtype):
    """ Return function get_residual(X, B, R) that will do R = B - AX """

    ### this will be wrong !!! this code is not adapted to the needed changes to make the
    ### biCGstab work

    num_shared_banks = 6  # TODO Dries: does this need to be increased?

    # Render the pre-loop and in-loop code.
    cuda_type = _get_cuda_type(dtype)
    code_allpre = jinja_env.get_template('fdfd_residual_pec_pmc.cu').\
                    render(dims=params['shape'], \
                           type=cuda_type, \
                           mu_equals_1=False, \
                           full_operator=True)

    # Grid input parameters.
    grid_params = [(A + i, 'grid', dtype) for A in ['X', 'B', 'R', 'e', 'm'] \
                                            for i in ['x', 'y', 'z']]

    # Const input parameters.
    const_names = ('sx0', 'sy0', 'sz0', 'sx1', 'sy1', 'sz1') + \
                    ('sqrt_sx0', 'sqrt_sy0', 'sqrt_sz0', \
                    'sqrt_sx1', 'sqrt_sy1', 'sqrt_sz1') + \
                    ('bloch_x', 'bloch_y', 'bloch_z')
    const_sizes = params['shape'] * 4 + tuple([3]) * 3
    const_params = [(const_names[k], 'const', dtype, const_sizes[k]) \
                        for k in range(len(const_sizes))]
    const_params.append(('pemc', 'const', params['pemc'].dtype.type, 6))

    # Compile. (note shape_filter = 'square')
    residual_fun = Kernel('', \
                    *(grid_params + const_params), \
                    pre_loop=code_allpre, \
                    padding=(1,1,1,1), \
                    smem_per_thread=num_shared_banks*16, \
                    shape_filter='square')

    # Temporary variables.

    # Grid variables.
    # !!!!! here eps is scattered over the GPUs when e intitialised
    e = [Grid(dtype(f), x_overlap=1) for f in params['e']]
    m = [Grid(dtype(f), x_overlap=1) for f in params['m']]  # Optional.

    # Constant variables.
    sc_pml_0 = [Const(dtype(s**-1)) for s in params['s']]
    sc_pml_1 = [Const(dtype(t**-1)) for t in params['t']]
    sqrt_sc_pml_0 = [Const(dtype(np.sqrt(s**-1))) for s in params['s']]
    sqrt_sc_pml_1 = [Const(dtype(np.sqrt(t**-1))) for t in params['t']]
    bloch_x = [Const(dtype(params['bloch_phase'][0]))]
    bloch_y = [Const(dtype(params['bloch_phase'][1]))]
    bloch_z = [Const(dtype(params['bloch_phase'][2]))]
    pemc = [Const(params['pemc'])]

    # Define the function
    def gpu_fdfd_residual(X, B, R):
        # Execute cuda code.
        residual_fun( \
                    *(X + B + R + e + m + \
                        sc_pml_0 + sc_pml_1 + sqrt_sc_pml_0 + sqrt_sc_pml_1 + \
                        bloch_x + bloch_y + bloch_z + pemc), \
                    post_sync = R)

    return gpu_fdfd_residual


def make_gpu_fdfd_matrix_multiplication(params, dtype):
    """ Return function vec_matrix_multiplication(X, B) that will do AX=B """

    num_shared_banks = 6

    # Render the pre-loop and in-loop code.
    cuda_type = _get_cuda_type(dtype)
    code_allpre = jinja_env.get_template('fdfd_matrix_multiplication_pec_pmc.cu').\
                    render(dims=params['shape'], \
                            type=cuda_type, \
                            mu_equals_1=False, \
                            full_operator=True)

    # Grid input parameters.
    grid_params = [(A + i, 'grid', dtype) for A in ['X', 'B', 'e', 'm'] \
                                            for i in ['x', 'y', 'z']]

    # Const input parameters.
    const_names = ('sx0', 'sy0', 'sz0', 'sx1', 'sy1', 'sz1') + \
                    ('sqrt_sx0', 'sqrt_sy0', 'sqrt_sz0', \
                    'sqrt_sx1', 'sqrt_sy1', 'sqrt_sz1') + \
                    ('bloch_x', 'bloch_y', 'bloch_z')
    const_sizes = params['shape'] * 4 + tuple([3]) * 3
    const_params = [(const_names[k], 'const', dtype, const_sizes[k]) \
                        for k in range(len(const_sizes))]
    const_params.append(('pemc', 'const', params['pemc'].dtype.type, 6))

    # Compile. (note shape_filter = 'square')
    A_multiplication_fun = Kernel('', \
                                  *(grid_params + const_params), \
                                  pre_loop=code_allpre, \
                                  padding=(1,1,1,1), \
                                  smem_per_thread=num_shared_banks*16, \
                                  shape_filter='square')

    # Temporary variables.

    # Grid variables.
    # !!!!! here eps is scattered over the GPUs when e intitialised
    e = [Grid(dtype(f), x_overlap=1) for f in params['e']]
    m = [Grid(dtype(f), x_overlap=1) for f in params['m']]  # Optional.

    # Constant variables.
    sc_pml_0 = [Const(dtype(s**-1)) for s in params['s']]
    sc_pml_1 = [Const(dtype(t**-1)) for t in params['t']]
    sqrt_sc_pml_0 = [Const(dtype(np.sqrt(s**-1))) for s in params['s']]
    sqrt_sc_pml_1 = [Const(dtype(np.sqrt(t**-1))) for t in params['t']]
    bloch_x = [Const(dtype(params['bloch_phase'][0]))]
    bloch_y = [Const(dtype(params['bloch_phase'][1]))]
    bloch_z = [Const(dtype(params['bloch_phase'][2]))]
    pemc = [Const(params['pemc'])]

    # Define the function
    def gpu_fdfd_matrix_multiplication(X, B):
        # Execute cuda code.
        A_multiplication_fun( \
                    *(X + B + e + m + \
                        sc_pml_0 + sc_pml_1 + sqrt_sc_pml_0 + sqrt_sc_pml_1 + \
                        bloch_x + bloch_y + bloch_z + pemc), \
                    post_sync = B)

    return gpu_fdfd_matrix_multiplication


def make_gpu_cond(dtype, cond):
    """ Returns a function gpu_cond(A) that does A=A*C """
    # GPU Code in gce.Kernel
    code = Template("""
        if (_in_global) {
            Ax(0,0,0) = Ax(0,0,0)*Cx(0,0,0);
            Ay(0,0,0) = Ay(0,0,0)*Cy(0,0,0);
            Az(0,0,0) = Az(0,0,0)*Cz(0,0,0);
        } """).render(type=_get_cuda_type(dtype))

    # Compile the code using gce.Kernel
    grid_names = [A + i for A in ['A', 'C'] for i in ['x', 'y', 'z']]
    Sum_fun = Kernel(code, \
                    *[(name, 'grid', dtype) for name in grid_names], \
                    shape_filter='skinny')

    C = cond

    # Define the actual function.
    def gpu_cond(A):
        Sum_fun(*( A + C ), \
                 post_sync=A) # r must be post-synced for upcoming alpha step.

    return gpu_cond


def make_DB_get_vec(dtype):
    """ Returns a function that does aA+bB=C """
    # Code for the rho step function.
    temp = [Grid(dtype, x_overlap=1) for k in range(3)]
    gpu_scaled_copy = make_gpu_scaled_copy(dtype)

    # Define the actual function.
    def DB_get_vec(A):
        gpu_scaled_copy(A, 1, temp)
        out = [E.get() for E in temp]
        return out

    return DB_get_vec
