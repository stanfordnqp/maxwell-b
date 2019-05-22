import copy
import numpy as np
from typing import List

from jinja2 import Template
from mpi4py.MPI import COMM_WORLD as comm

from gce.space import initialize_space, get_space_info
from gce.grid import Grid
from gce.const import Const
from gce.out import Out
from gce.kernel import Kernel
from maxwell_ops_lumped import _get_cuda_type
from maxwell_ops_lumped import *


# CG
#----
def ops(params):
    """ Define the operations that specify the symmetrized, lumped problem. """

    # Initialize the space.
    initialize_space(params['shape'])

    dtype = np.complex128

    if comm.rank == 0:
        pre_cond, post_cond = conditioners(params, dtype)
        params['j'] = pre_cond(params['j'])  # Precondition b.
        params['x'] = pre_cond(params['x'])
    else:
        post_cond = None

    b = [Grid(dtype(f), x_overlap=1) for f in params['j']]
    x = [Grid(dtype(f), x_overlap=1) for f in params['x']]

    # Return b, the lumped operations needed for the bicg algorithm, and
    # the postconditioner to obtain the "true" x.
    return  b, x, \
            {'zeros': lambda: [Grid(dtype, x_overlap=1) for k in range(3)], \
             'rho_step': rho_step(dtype), \
             'alpha_step': alpha_step(params, dtype)}, \
            post_cond, \
            {'gpu_copy': make_gpu_copy(dtype)}


def rho_step(dtype):
    """ Return the function to execute the rho step of the bicg algorithm. """

    # Code for the rho step function.
    code = Template("""
        if (_in_global) {
            x0(0,0,0) = x0(0,0,0) + alpha * p0(0,0,0);
            x1(0,0,0) = x1(0,0,0) + alpha * p1(0,0,0);
            x2(0,0,0) = x2(0,0,0) + alpha * p2(0,0,0);
            {{ type }} s0 = r0(0,0,0) - alpha * v0(0,0,0);
            {{ type }} s1 = r1(0,0,0) - alpha * v1(0,0,0);
            {{ type }} s2 = r2(0,0,0) - alpha * v2(0,0,0);
            rho += (s0 * s0) + (s1 * s1) + (s2 * s2);
            err +=  (real(s0) * real(s0)) + \
                    (imag(s0) * imag(s0)) + \
                    (real(s1) * real(s1)) + \
                    (imag(s1) * imag(s1)) + \
                    (real(s2) * real(s2)) + \
                    (imag(s2) * imag(s2));
            r0(0,0,0) = s0;
            r1(0,0,0) = s1;
            r2(0,0,0) = s2;
        } """).render(type=_get_cuda_type(dtype))

    # Compile the code.
    grid_names = [A + i for A in ['p', 'r', 'v', 'x'] for i in ['0', '1', '2']]
    rho_fun = Kernel(code, \
                    ('alpha', 'number', dtype), \
                    ('rho', 'out', dtype), \
                    ('err', 'out', dtype), \
                    *[(name, 'grid', dtype) for name in grid_names], \
                    shape_filter='skinny')

    # Temporary values that are needed.
    rho_out = Out(dtype)
    err_out = Out(dtype)

    # Define the actual function.
    def rho_step(alpha, p, r, v, x):
        rho_fun(dtype(alpha), rho_out, err_out, *(p + r + v + x), \
                post_sync=r) # r must be post-synced for upcoming alpha step.
        return rho_out.get(), np.sqrt(err_out.get())

    return rho_step


def alpha_step(params, dtype):
    """ Define the alpha step function needed for the bicg algorithm. """
    num_shared_banks = 6

    # Render the pre-loop and in-loop code.
    cuda_type = _get_cuda_type(dtype)
    code_allpre = jinja_env.get_template('alpha_allpre.cu').\
                    render(dims=params['shape'], \
                            type=cuda_type, \
                            mu_equals_1=False, \
                            full_operator=True)

    # Grid input parameters.
    grid_params = [(A + i, 'grid', dtype) for A in ['P', 'P1', 'R', 'V', 'e', 'm'] \
                                            for i in ['x', 'y', 'z']]

    # Const input parameters.
    const_names = ('sx0', 'sy0', 'sz0', 'sx1', 'sy1', 'sz1') + \
                    ('sqrt_sx0', 'sqrt_sy0', 'sqrt_sz0', \
                    'sqrt_sx1', 'sqrt_sy1', 'sqrt_sz1')
    const_sizes = params['shape'] * 4
    const_params = [(const_names[k], 'const', dtype, const_sizes[k]) \
                        for k in range(len(const_sizes))]

    # Compile.
    alpha_fun = Kernel('', \
                    ('beta', 'number', dtype), \
                    ('alpha_denom', 'out', dtype), \
                    *(grid_params + const_params), \
                    pre_loop=code_allpre, \
                    padding=(1,1,1,1), \
                    smem_per_thread=num_shared_banks*16, \
                    shape_filter='square')

    # Temporary variables.
    alpha_denom_out = Out(dtype)
    p_temp = [Grid(dtype, x_overlap=1) for k in range(3)]  # Used to swap p.

    # Grid variables.
    e = [Grid(dtype(f), x_overlap=1) for f in params['e']]
    m = [Grid(dtype(f), x_overlap=1) for f in params['m']]  # Optional.

    # Constant variables.
    sc_pml_0 = [Const(dtype(s**-1)) for s in params['s']]
    sc_pml_1 = [Const(dtype(t**-1)) for t in params['t']]
    sqrt_sc_pml_0 = [Const(dtype(np.sqrt(s**-1))) for s in params['s']]
    sqrt_sc_pml_1 = [Const(dtype(np.sqrt(t**-1))) for t in params['t']]

    # Define the function
    def alpha_step(rho_k, rho_k_1, p, r, v, compute_alpha=True):
        # Execute cuda code.
        # Notice that p_temp and v are post_synced.
        alpha_fun(dtype(rho_k/rho_k_1), alpha_denom_out, \
                    *(p + p_temp + r + v + e + m + \
                        sc_pml_0 + sc_pml_1 + sqrt_sc_pml_0 + sqrt_sc_pml_1), \
                    post_sync=p_temp+v)
        p[:], p_temp[:] = p_temp[:], p[:]  # Deep swap.

        # TODO(logansu): Remove compute_alpha solve_symm_lumped does not use
        # alpha_step to solve for the matrix. Because solve_symm_lumped sets
        # r to zero vector to compute the matrix multiplication, alpha_denom_out
        # comes out to be zero. The if-statement stops DivisionByZero when this
        # happens (which is important for us to catch real DivisionByZero
        # errors).
        if compute_alpha:
            return rho_k / alpha_denom_out.get()  # The value of alpha.

    return alpha_step


# biCGSTAB
#----------
def ops_biCGSTAB(params):
    """ Define the operations for biCGSTAB. """
    '''
    This function initializes the space and preconditions b and x(needed should you have a specific 
    initial x)
    it returns
        b: preconditioned j
        x: preconditioned x
        biCGSTAB_functions:
            zeros: a function that will create [gce.grid, gce.grid, gce.grid], with grid having the 
                    shape of the space and initialized  with zeros
            rho_step: the rho_step function
            alpha_step: the alpha_step function
            omega_step: the omega-step function
        post_cond: the post conditioning function
        gpu_functions: Extra functions for debugging
    When making space, b, x and rho_step and alpha_step the memory on the GPU is allocated.(all these variables
    inherite from gce.data)
    '''

    # Initialize the space.
    # this creates the global variables __GLOBAL_SPACE, __DEVICE, __CONTEXT
    # Every MPI node gets a device and creates its context.
    # With the device and context a __Space() object is created called
    # __GLOBAL_SPACE.
    # The __SPACE object will has one function: get_info. You can call it just by
    # using get_space_info. It gives a dict with all the info of the device
    # and context. Also, it containt how the space has to devided over the
    # MPI devices: 'x_range'.
    initialize_space(params['shape'])

    # type used for the arrays
    dtype = np.complex128

    # get pre and post conditioners and apply on pre_cond on j and b
    # (only params of rank 0 has the j and x)
    if comm.rank == 0:
        pre_cond, post_cond = conditioners(params, dtype)
        params['j'] = pre_cond(params['j'])  # Precondition b.
        params['x'] = pre_cond(params['x'])
    else:
        post_cond = None

    # save the data j->b, x->x in the format [gce.Grid, gce.Grid, gce.Grid]
    # gce.grid is a child from data, so it puts the data on the GPU, it also has the
    # required functions to synch the boundaries of the data between different MPI nodes
    r_hatH = [Grid(dtype(f), x_overlap=1) for f in params['j']]  # r_hatH = b
    x = [Grid(dtype(f), x_overlap=1) for f in params['x']]
    # r_hatH = [Grid(dtype(np.conj(f)), x_overlap=1) for f in params['j']]

    # calculate the residual
    gpu_matrix_mult = make_gpu_fdfd_matrix_multiplication(params, dtype)
    gpu_sum = make_gpu_sum(dtype)
    gpu_conj = make_gpu_conj(dtype)
    r = [Grid(dtype, x_overlap=1) for k in range(3)]
    gpu_matrix_mult(x, r)  # r=Ax
    gpu_sum(1, -1, r_hatH, r, r)  # r=r_hatH-r=b-Ax
    gpu_conj(r, r_hatH)  # r_hatH=conj(r)

    # Return b, the lumped operations needed for the bicg algorithm, and
    # the postconditioner to obtain the "true" x.
    return  r, x, r_hatH, \
            {'zeros': lambda: [Grid(dtype, x_overlap=1) for k in range(3)], \
             'rho_step': rho_biCGSTAB_step(dtype), \
             'alpha_step': alpha_biCGSTAB_step(params, dtype), \
             'omega_step': omega_biCGSTAB_step(params, dtype)}, \
            post_cond, \
            {}


def rho_biCGSTAB_step(dtype):
    """ Return the function to execute the rho step of the bicg algorithm. """
    '''
    This returns a function that will perform the rho_step, i.e. a part of the 
    biCGSTAB algorithm
        x=x+alpha*p+omega*s
        r=s-omega*t
        rho[k+1]=r_hatH*r
        err=conj(r)*r
    the function returns rho[k+1] and err (it is returned to the CPU where it is 
    gathered and summed!)
    '''
    # Code for the rho step function.
    code = Template("""
        if (_in_global) {
            X1x(0,0,0) = Xx(0,0,0) + alpha*Px(0,0,0) + omega * Sx(0,0,0);
            X1y(0,0,0) = Xy(0,0,0) + alpha*Py(0,0,0) + omega * Sy(0,0,0);
            X1z(0,0,0) = Xz(0,0,0) + alpha*Pz(0,0,0) + omega * Sz(0,0,0);
            {{ type }} R_tmpx = Sx(0,0,0) - omega * Tx(0,0,0);
            {{ type }} R_tmpy = Sy(0,0,0) - omega * Ty(0,0,0);
            {{ type }} R_tmpz = Sz(0,0,0) - omega * Tz(0,0,0);
            rho += (R_hatHx(0,0,0) * R_tmpx) + (R_hatHy(0,0,0) * R_tmpy) + (R_hatHz(0,0,0) * R_tmpz);
            err +=  (real(R_tmpx) * real(R_tmpx)) + \
                    (imag(R_tmpx) * imag(R_tmpx)) + \
                    (real(R_tmpy) * real(R_tmpy)) + \
                    (imag(R_tmpy) * imag(R_tmpy)) + \
                    (real(R_tmpz) * real(R_tmpz)) + \
                    (imag(R_tmpz) * imag(R_tmpz));
            Rx(0,0,0) = R_tmpx;
            Ry(0,0,0) = R_tmpy;
            Rz(0,0,0) = R_tmpz;
        } """).render(type=_get_cuda_type(dtype))

    # Compile the code using gce.Kernel
    grid_names = [
        A + i for A in ['S', 'X', 'X1', 'P', 'T', 'R', 'R_hatH']
        for i in ['x', 'y', 'z']
    ]
    rho_biCGSTAB_fun = Kernel(code, \
                    ('alpha', 'number', dtype), \
                    ('omega', 'number', dtype), \
                    ('rho', 'out', dtype), \
                    ('err', 'out', dtype), \
                    *[(name, 'grid', dtype) for name in grid_names], \
                    shape_filter='skinny')

    # Temporary values that are needed. CG-variable that are stored on the gpu in a gce.Out type (child of gce.data)
    rho_out = Out(dtype)
    err_out = Out(dtype)
    X_temp = [Grid(dtype, x_overlap=1) for k in range(3)]  # Used to swap p.

    # Define the actual function.
    def rho_biCGSTAB_step(alpha, omega, S, X, P, T, R, R_hatH):
        rho_biCGSTAB_fun(dtype(alpha), dtype(omega), rho_out, err_out, \
                         *( S + X + X_temp + P + T + R + R_hatH ), \
                         post_sync= X_temp + R ) # r must be post-synced for upcoming alpha step.
        X[:], X_temp[:] = X_temp[:], X[:]  # deep swap X

        return rho_out.get(), np.sqrt(err_out.get())

    return rho_biCGSTAB_step


def alpha_biCGSTAB_step(params, dtype):
    """ Define the alpha step function needed for the bicg algorithm. """
    '''
    This returns a function that will perform the alpha_step, i.e. a part of the biCGSTAB algorithm
        p=r+rho[k]/rho[k+1]*alpha/omega*(p-omega*v)
        v=A*p
        alpha=rho/(p*v)
    the function returns alpha
    note that in alpha fun does not calculate alpha since p and v are scattered over the different MPI nodes
    however alpha_denom is calculated, or at least the part that the MPInode can calculate., and then put together
    by alpha_step.
    '''

    num_shared_banks = 6

    # Render the pre-loop and in-loop code.
    cuda_type = _get_cuda_type(dtype)
    #code_allpre = jinja_env.get_template('alpha_biCGSTAB.cu').\
    code_allpre = jinja_env.get_template('alpha_bloch_pmc_pec.cu').\
                    render(dims=params['shape'], \
                            type=cuda_type, \
                            mu_equals_1=False, \
                            full_operator=True)

    # Grid input parameters.
    grid_params = [(A + i, 'grid', dtype) for A in ['P', 'P1', 'R', 'R_hatH', 'V', 'V1', 'e', 'm'] \
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
    alpha_fun = Kernel('', \
                    ('beta', 'number', dtype), \
                    ('omega', 'number', dtype), \
                    ('alpha_denom', 'out', dtype), \
                    *(grid_params + const_params), \
                    pre_loop=code_allpre, \
                    padding=(1,1,1,1), \
                    smem_per_thread=num_shared_banks*16, \
                    shape_filter='square')

    # Temporary variables.
    alpha_denom_out = Out(dtype)
    P_temp = [Grid(dtype, x_overlap=1) for k in range(3)]  # Used to swap p.
    V_temp = [Grid(dtype, x_overlap=1) for k in range(3)]  # Used to swap v.

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
    def alpha_biCGSTAB_step(rho_k,
                            rho_k_1,
                            alpha,
                            omega,
                            P,
                            R,
                            R_hatH,
                            V,
                            compute_alpha=True):
        # Execute cuda code.
        # Notice that p_temp and v are post_synced.
        alpha_fun(dtype((rho_k*alpha)/(rho_k_1*omega)), dtype(omega), alpha_denom_out, \
                    *(P + P_temp + R + R_hatH + V + V_temp + e + m + \
                        sc_pml_0 + sc_pml_1 + sqrt_sc_pml_0 + sqrt_sc_pml_1 + \
                        bloch_x + bloch_y + bloch_z + pemc), \
                    post_sync = P_temp + V_temp)
        P[:], P_temp[:] = P_temp[:], P[:]  # Deep swap.
        V[:], V_temp[:] = V_temp[:], V[:]  # Deep swap

        # TODO(logansu): Remove compute_alpha solve_symm_lumped does not use
        # alpha_step to solve for the matrix. Because solve_symm_lumped sets
        # r to zero vector to compute the matrix multiplication, alpha_denom_out
        # comes out to be zero. The if-statement stops DivisionByZero when this
        # happens (which is important for us to catch real DivisionByZero
        # errors).
        if compute_alpha:
            return rho_k / alpha_denom_out.get()  # The value of alpha.

    return alpha_biCGSTAB_step


def omega_biCGSTAB_step(params, dtype):
    """ Define the alpha step function needed for the bicg algorithm. """
    '''
    This returns a function that will perform the alpha_step, i.e. a part of the CG algorithm
        s = r - alpha * v
        t = A*s
        omega=(t*s)/(t*t)
    the function returns alpha
    note that in omega fun does not calculate calculate since t and s are scattered over the different MPI nodes
    however omega_num and omega_denom are calculated, or at least the part that the MPInode can calculate.,
    and then put together by omega_step.
    '''
    num_shared_banks = 6

    # Render the pre-loop and in-loop code.
    cuda_type = _get_cuda_type(dtype)
    code_allpre = jinja_env.get_template('omega_bloch_pmc_pec.cu').\
                    render(dims=params['shape'], \
                            type=cuda_type, \
                            mu_equals_1=False, \
                            full_operator=True)

    # Grid input parameters.
    grid_params = [(A + i, 'grid', dtype) for A in ['V', 'S', 'R', 'T', 'e', 'm'] \
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
    omega_fun = Kernel('', \
                    ('alpha', 'number', dtype), \
                    ('omega_num', 'out', dtype), \
                    ('omega_denom', 'out', dtype), \
                    *(grid_params + const_params), \
                    pre_loop=code_allpre, \
                    padding=(1,1,1,1), \
                    smem_per_thread=num_shared_banks*16, \
                    shape_filter='square')

    # Temporary variables.
    omega_num_out = Out(dtype)
    omega_denom_out = Out(dtype)

    # Grid variables.
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
    def omega_step(alpha, V, S, R, T, compute_omega=True):
        # Execute cuda code.
        # Notice that H, S and T are post_synced.
        omega_fun(dtype(alpha), omega_num_out, omega_denom_out, \
                    *( V + S + R + T + e + m + \
                        sc_pml_0 + sc_pml_1 + sqrt_sc_pml_0 + sqrt_sc_pml_1 + \
                        bloch_x + bloch_y + bloch_z + pemc), \
                    post_sync=  S + T )

        if compute_omega:
            return omega_num_out.get() / omega_denom_out.get(
            )  # The value of omega.

    return omega_step


def _axby(a, x, b, y):
    """ Default axby routine. """
    y[:] = a * x + b * y


def _nochange(x, y):
    """ Default multA and multAT routine, simply sets y = x. """
    _axby(1, x, 0, y)


def solve_asymm(b, x=None, x_hat=None, multA=_nochange, multAT=_nochange,
                norm=np.linalg.norm, dot=np.dot, axby=_axby, \
                zeros=None, \
                eps=1e-6, max_iters=1000):
    """ Bi-conjugate gradient solve of square, non-symmetric system.

    Input variables:
    b -- the problem to be solved is A * x = b.

    Keyword variables:
    x -- initial guess of x, default value is 0.
    x_hat -- initial guess for x_hat, default value is 0.
    multA -- multA(x) calculates A * x and returns the result,
             default: returns x.
    multAT -- multAT(x) calculates A^T * x and returns the result.
              default: returns x.
    dot -- dot(x, y) calculates xT * y and returns the result, 
           default: numpy.dot().
    axby -- axby(a, x, b, y) calculates a * x + b * y and 
            stores the result in y, default y[:] = a * x + b * y.
    copy -- copy(x) returns a copy of x, default numpy.copy().
    eps -- the termination error is determined by eps * ||b|| (2-norm),
           default 1e-6.
    max_iters -- maximum number of iterations allowed, default 1000.

    Output variables:
    x -- the apprximate answer of A * x = b.
    err -- a numpy array with the error value at every iteration.
    success -- True if convergence was successful, False otherwise.
    """

    # TODO
    # ----
    # *   Check for breakdown condition.
    # *   Notify user if we weren't able to beat term_err.

    if zeros is None:  # Default version of the zeros operation.

        def zeros():
            return np.zeros_like(b)

    if x is None:  # Default value of x is 0.
        x = zeros()
        axby(0, b, 0, x)

    if x_hat is None:  # Default value for x_hat is 0.
        x_hat = zeros()
        axby(0, b, 0, x_hat)

    # r = b - A * x.
    r = zeros()
    multA(x, r)
    axby(1, b, -1, r)

    # r_hat = b - A * x_hat
    r_hat = zeros()
    multAT(x, r_hat)
    axby(1, b, -1, r_hat)

    # p = r, p_hat = r_hat.
    p = zeros()
    axby(1, r, 0, p)
    p_hat = zeros()
    axby(1, r_hat, 0, p_hat)

    # Initialize v, v_hat. Used to store A * p, AT * p_hat.
    v = zeros()  # Don't need the values, this is an "empty" copy.
    v_hat = zeros()

    rho = np.zeros(max_iters).astype(np.complex128)  # Related to error.
    err = np.zeros(max_iters).astype(np.float64)  # Error.
    term_err = eps * norm(b)  # Termination error value.

    for k in range(max_iters):

        # Compute error and check termination condition.
        err[k] = norm(r)
        print(k, err[k])

        if err[k] < term_err:  # We successfully converged!
            return x, err[:k + 1], True
        elif np.isnan(err[k]):  # Hopefully this will never happen.
            return x, err[:k + 1], False

        # rho = r_hatT * r.
        rho[k] = dot(r_hat, r)

        #         if abs(rho[k]) < 1e-15: # Breakdown condition.
        #             raise ArithmeticError('Breakdown')

        multA(p, v)
        multAT(p_hat, v_hat)

        # alpha = rho / (p_hatT * v).
        alpha = rho[k] / dot(p_hat, v)

        # x += alpha * p, x_hat += alpha * p_hat.
        axby(alpha, p, 1, x)
        axby(alpha, p_hat, 1, x_hat)

        # r -= alpha * v, r -= alpha * v_hat.
        axby(-alpha, v, 1, r)
        axby(-alpha, v_hat, 1, r_hat)

        # beta = (r_hatT * r) / rho.
        beta = dot(r_hat, r) / rho[k]

        # p = r + beta * p, p_hat = r_hat + beta * p_hat.
        axby(1, r, beta, p)
        axby(1, r_hat, beta, p_hat)

    # Return the answer, and the progress we made.
    return x, err, False


def solve_symm(b, x=None, multA=_nochange,
                norm=np.linalg.norm, dot=np.dot, \
                axby=_axby, zeros=None, eps=1e-6, max_iters=1000):
    """ Bi-conjugate gradient solve of square, symmetric system.

    Input variables:
    b -- the problem to be solved is A * x = b.

    Keyword variables:
    x -- initial guess of x, default value is 0.
    multA -- multA(x) calculates A * x and returns the result,
             default: returns x.
    dot -- dot(x, y) calculates xT * y and returns the result, 
           default: numpy.dot().
    axby -- axby(a, x, b, y) calculates a * x + b * y and 
            stores the result in y, default y[:] = a * x + b * y.
    zeros -- zeros() creates a zero-initialized vector. 
    eps -- the termination error is determined by eps * ||b|| (2-norm),
           default 1e-6.
    max_iters -- maximum number of iterations allowed, default 1000.

    Output variables:
    x -- the approximate answer of A * x = b.
    err -- a numpy array with the error value at every iteration.
    success -- True if convergence was successful, False otherwise.
    """

    # TODO
    # ----
    # *   Check for breakdown condition.
    # *   Notify user if we weren't able to beat term_err.

    if zeros is None:  # Default version of the zeros operation.

        def zeros():
            return np.zeros_like(b)

    if x is None:  # Default value of x is 0.
        x = zeros()

    # r = b - A * x.
    r = zeros()
    multA(x, r)
    axby(1, b, -1, r)

    # p = r.
    p = zeros()
    axby(1, r, 0, p)

    # Initialize v. Used to store A * p.
    v = zeros()  # Don't need the values, this is an "empty" copy.
    v_hat = zeros()

    rho = np.zeros(max_iters).astype(np.complex128)  # Related to error.
    err = np.zeros(max_iters).astype(np.float64)  # Error.
    term_err = eps * norm(b)  # Termination error value.

    for k in range(max_iters):

        # Compute error and check termination condition.
        err[k] = norm(r)
        print(k, err[k])

        if err[k] < term_err:  # We successfully converged!
            return x, err[:k + 1], True

        # rho = r^T * r.
        rho[k] = dot(r, r)

        #         if abs(rho[k]) < 1e-15: # Breakdown condition.
        #             raise ArithmeticError('Breakdown')

        multA(p, v)

        # alpha = rho / (p^T * v).
        alpha = rho[k] / dot(p, v)

        # x += alpha * p.
        axby(alpha, p, 1, x)

        # r -= alpha * v.
        axby(-alpha, v, 1, r)

        # beta = (r^T * r) / rho.
        beta = dot(r, r) / rho[k]

        # p = r + beta * p.
        axby(1, r, beta, p)

    # Return the answer, and the progress we made.
    return x, err, False


def solve_symm_lumped(r, x=None, rho_step=None, alpha_step=None, zeros=None, \
                    err_thresh=1e-6, max_iters=1000, reporter=lambda err: None,
                     gpu_func=None):
    # Note: r is used instead of b in the input parameters of the function.
    # This is in order to initialize r = b, and to inherently disallow access to
    # b from within this function.
    """ Lumped bi-conjugate gradient solve of a symmetric system.

    Input variables:
    b -- the problem to be solved is A * x = b.

    Keyword variables:
    x -- initial guess of x, default value is 0.
    rho_step -- rho_step(alpha, p, r, v, x) updates r and x, and returns
        rho and the error. Specifically, rho_step performs:
            x = x + alpha * p
            r = r - alpha * v
            rho_(k+1) = (r dot r)
            err = (conj(r) dot r)
    alpha_step -- alpha_step(rho_k, rho_(k-1), p, r, v) updates p and v, and 
        returns alpha. Specifically, alpha_step performs:
            p = r + (rho_k / rho_(k-1)) * p
            v = A * p
            alpha = rho_k / (p dot v)
    zeros -- zeros() creates a zero-initialized vector. 
    err_thresh -- the relative error threshold, default 1e-6.
    max_iters -- maximum number of iterations allowed, default 1000.
    reporter -- function to report progress.

    Output variables:
    x -- the approximate answer of A * x = b.
    err -- a numpy array with the error value at every iteration.
    success -- True if convergence was successful, False otherwise.
    """

    # TODO
    # ----
    # *   Check for breakdown condition.

    # Initialize variables.
    # Note that r = b was "initialized" in the function declaration.

    # Initialize x = 0, if defined.
    if x is None:  # Default value of x is 0.
        x = zeros()

    # Initialize v = Ax.
    v = zeros()
    # TODO(logansu): Remove compute_alpha once refactored.
    alpha_step(
        1, 1, x, zeros(), v, compute_alpha=False)  # Used to calculate v = Ax.

    p = zeros()  # Initialize p = 0.
    alpha = 1  # Initial value for alpha.

    rho = np.zeros(max_iters).astype(np.complex128)
    rho[-1] = 1  # Silly trick so that rho[k-1] for k = 0 is defined.

    err = np.zeros(max_iters).astype(np.float64)  # Error.

    temp, b_norm = rho_step(0, p, r, v, x)  # Calculate norm of b, ||b||.

    #prepare variables for residual calculation
    b = zeros()
    Ax = zeros()
    zero_vec = zeros()
    temp = zeros()
    gpu_func['gpu_copy'](r, b)

    print('b_norm check: ',
          b_norm)  # Make sure this isn't something bad like 0.

    for k in range(max_iters):

        print('rho  ', k, end=" ")
        #if not (k+1)%50 and gpu_func is not None:
        #    gpu_func['gpuSum_vec'](1, alpha, x, p, temp)
        #    gpu_func['gpuCopy_vec'](temp, x)
        #    gpu_func['gpuA_multiplication'](x, Ax)
        #    gpu_func['gpuSum_vec'](1, -1, b, Ax, r)
        #    rho[k], err0 = rho_step(0, zero_vec, r, zero_vec, x)
        #    err[k] = np.absolute(err0 / b_norm)
        #    print('    Calculating residual')
        #else:
        rho[k], err0 = rho_step(alpha, p, r, v, x)
        err[k] = np.absolute(err0 / b_norm)

        # Check termination condition.
        reporter(err[k])
        if err[k] < err_thresh:  # We successfully converged!
            return x, err[:k + 1], True, k

        print('alpha', k, end=" ")
        alpha = alpha_step(rho[k], rho[k - 1], p, r, v)
        print('err: ' + str(err[k]))

    # Return the answer, and the progress we made.
    return x, err, False, max_iters


def solve_asymm_biCGSTAB(r,
                         r_hatH,
                         x=None,
                         rho_step=None,
                         omega_step=None,
                         alpha_step=None,
                         zeros=None,
                         err_thresh=1e-6,
                         max_iters=1000,
                         reporter=lambda err: None,
                         gpu_func=None):
    """  solve of asymmetric system.

    Input variables:
        - r=b residual of the problem to be solved is A * x = b.
        - x= initial x, type:[gce.grid, gce.grid, gce.grid]
        - rho_step= rho_step function created by maxwell_ops_lumped.ops(params)
        - omega_step= omega_step function created by maxwell_ops_lumped.ops(params)
        - alpha_step= alpha_step function created by maxwell_ops_lumped.ops(params)
        - max_iters
        - reporter
        - gpu-func: additional functions for debugging

    Keyword variables:
    x -- initial guess of x, default value is 0.
    rho_step -- rho_step(alpha, omega, s, x, p, t, r, r_hatH) updates r and x, and returns
        rho and the error. Specifically, rho_step performs:
            x=x+alpha*p+omega*s
            r=r-omega*t
            rho[k+1]=r_hatH*r
            err=conj(r)*r
    alpha_step -- alpha_step(rho_k, rho_(k-1), alpha, omega, p, r, r_hatH, v) updates p and v, and 
        returns alpha. Specifically, alpha_step performs:
            p=r+rho[k]/rho[k+1]*alpha/omega*(p-omega*v)
            v=A*p
            alpha=rho/(p*v)
    omega_step -- omega_step(alpha, v, s, r, t) updates s and t, and returns 
        omega. Specifically, omega_step performs:
            s = r - alpha * v
            t = A*s
            omega=(t*s)/(t*t)
    max_iters -- maximum number of iterations allowed, default 1000.
    reporter -- function to report progress.
    gpu_func -- functions for debugging

    NOTE(Dries): in Nocedal every 50 steps r is calculeted as b-A*x to avoid the 
                buildup of numerical errors

    Output variables:
    x -- the approximate answer of A * x = b.
    err -- a numpy array with the error value at every iteration.
    success -- True if convergence was successful, False otherwise.
    """

    # TODO
    # ----
    # *   Check for breakdown condition.

    # Initialize variables.
    # Note that r has to be b-Ax and r_harH=conj(r). This is done in maxwell_ops_lumpled...

    # Initialize x and v,
    v = zeros()
    if x is None:  # Default value of x is 0.
        x = zeros()
    else:
        alpha_step(0, 1, 0, 1, x, zeros(), zeros(), v, compute_alpha=False)

    p = zeros()  # Initialize p = 0.
    t = zeros()  # Initialize t = 0.
    s = zeros()  # Initialize s = 0.

    alpha = 1  # Initial value for alpha.
    rho = np.zeros(max_iters + 1).astype(np.complex128)
    rho[-1] = 1  # Silly trick so that rho[k-1] for k = 0 is defined.
    omega = np.zeros(max_iters).astype(np.complex128)
    omega[-1] = 1  # Silly trick so that rho[k-1] for k = 0 is defined.

    err = np.zeros(max_iters).astype(np.float64)  # Error.

    rho[0], b_norm = rho_step(0.0, 0.0, r, x, zeros(), t, r,
                              r_hatH)  # Calculate norm of b, ||b||.

    print('b_norm check: ' +
          str(b_norm))  # Make sure this isn't something bad like 0.
    print('rho[0] check: ' + str(
        rho[0]))  # Make sure this isn't something bad like 0.

    for k in range(max_iters):

        print('alpha ', k, end=" ")
        # alpha step
        # beta = rho_k/rho_k_1 * alpha/omega_i_1
        # p_i = r_i_1 + beta(p_i_1-omega_i_1*v_i_1)
        # v_i = A*p_i
        # alpha = rho_i/(r0H,v_i)
        alpha = alpha_step(rho[k], rho[k - 1], alpha, omega[k - 1], p, r,
                           r_hatH, v)
        print('    alpha=' + str(alpha))
        print('omega', k, end=" ")
        # omega step
        # s = r_i_1 - alpha*v_i
        # t = A*s
        # omega_k = (tH,s)/(tH,t)
        omega[k] = omega_step(alpha, v, s, r, t)
        print('    omega=' + str(omega[k]))
        print('rho  ', k, end=" ")
        # rho step
        # x_i = x_i_1 + alpha*p_i + omega_i*s
        # r_i = s - omega * t
        # rho_ix1 = (r0H,r_i)
        # err = (r_iH, r_i)
        rho[k + 1], err0 = rho_step(alpha, omega[k], s, x, p, t, r, r_hatH)
        err[k] = np.absolute(err0 / b_norm)
        print('    rho=' + str(rho[k + 1]))
        print('    error = ' + str(err[k]) + '\n')
        # Check termination condition.
        reporter(err[k])
        if err[k] < err_thresh:  # We successfully converged!
            return x, err[:k + 1], True, k

    # Return the answer, and the progress we made.
    return x, err, False, max_iters
