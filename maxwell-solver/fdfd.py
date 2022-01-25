import h5py
import numpy as np
import maxwell_ops_lumped
from solvers import bicg
from gce.grid import Grid
from mpi4py.MPI import COMM_WORLD as comm
import time, sys, tempfile, os

from pycuda import driver


def simulate(name, check_success_only=False):
    """ Read simulation from input file, simulate, and write out results. """
    print_comm0('starting simulate')

    # Reset the environment variables pointing to the temporary directory.
    tempfile.tempdir = '/tmp'

    # Create the reporter function.
    write_status = lambda msg: open(name + '.status', 'a').write(msg)
    if comm.Get_rank() == 0:
        # write_status('EXEC initializing\n')
        def rep(err, info: str = None):
            if info is None:
                write_status('%e\n' % np.abs(err))
            else:
                write_status('%s %e\n' % (info, np.abs(err)))
    else:  # No reporting needed for non-root nodes.

        def rep(err, info: str = None):
            pass

    # Get input parameters.
    params = get_parameters(name)
    solver = params['solver']
    if solver == 'CG':
        # Define operations needed for the CG operation.(bicg is CG)
        b, x, ops, post_cond, gpu_func = bicg.ops(params)
        # Solve!
        start_time = time.time()
        x, err, success, iters = bicg.solve_symm_lumped(b, x=x, \
                                                max_iters=params['max_iters'], \
                                                reporter=rep, \
                                                err_thresh=params['err_thresh'], \
                                                gpu_func=gpu_func, \
                                                **ops)
        stop_time = time.time()

    elif solver == 'biCGSTAB':
        # Define operations needed for the biCGSTAB operation.
        b, x, r_hatH, ops, post_cond, gpu_func = bicg.ops_biCGSTAB(params)

        # Solve!
        start_time = time.time()
        x, err, success, iters = bicg.solve_asymm_biCGSTAB( b, r_hatH, x=x, \
                                                    max_iters=params['max_iters'], \
                                                    reporter=rep, \
                                                    err_thresh=params['err_thresh'], \
                                                    gpu_func=gpu_func,\
                                                    **ops)
        stop_time = time.time()

    elif solver == 'lgmres':
        from solvers import lgmres
        # Define operations needed for the lumped bicg operation.
        b, x, lgmres_functions, post_cond, gpu_func = lgmres.ops_lgmres(params)

        options = {
            'maxiters': params['max_iters'],
            'inner_m': 15,
            'outer_k': 2,
            'tol': params['err_thresh']
        }
        # Solve!
        start_time = time.time()
        x, err, success, iters = lgmres.solve_asymm_lgmres( b, x=x, \
                                                            reporter=rep, \
                                                            lgmres_func=lgmres_functions,\
                                                            options=options, \
                                                            gpu_func=gpu_func)
        stop_time = time.time()

    elif solver == 'Jacobi-Davidson':
        from solvers import lgmres, JacDav
        # Check if x is zeros and do a simulation with biCGSTAB if so
        if not np.any(params['x']):
            b, x, lgmres_functions, post_cond, gpu_func = lgmres.ops_lgmres(
                params)
            options = {
                'maxiters': 300,  #params['max_iters'],
                'inner_m': 15,
                'outer_k': 2,
                'tol': 10 * params['err_thresh']
            }
            print_comm0('zero E0 - initial simulation needed')
            x_start, err, success, iters = lgmres.solve_asymm_lgmres( b, x=x, \
                                                                reporter=rep, \
                                                                lgmres_func=lgmres_functions,\
                                                                options=options, \
                                                                gpu_func=gpu_func)
            #b, x0, r_hatH, ops, post_cond, gpu_func = maxwell_ops_lumped.ops_lgmres(params)
            #x_start, err, success, iters = bicg.solve_asymm_biCGSTAB( b, r_hatH, x=x0, \
            #                                            max_iters=params['max_iters'], \
            #                                            reporter=rep, \
            #                                            err_thresh=10*params['err_thresh'], \
            #                                            gpu_func=gpu_func,\
            #                                            **ops)
            params['x'] = [
                E.get() for E in x_start
            ]  #.get() will get the data from the gpu and gather it to the root
            if comm.Get_rank() == 0:
                params['x'] = post_cond(params['x'])  # Apply postconditioner
            #shp = params['x'][0].shape
            #params['x']=[np.random.rand(shp[0], shp[1], shp[2]) for i in range(3)]
            del x_start
        else:
            print_comm0('none zero E0 - No initial simulation needed')

        # Change the precompution that was done on the permitivity (j and m is kept)
        # So, undo eps = omega**2*eps
        if comm.Get_rank() == 0:
            for k in range(3):
                params['e'][k] = (params['omega']**(-2) * params['e'][k])

        # Define operations needed for the JacDav operation.
        print_comm0('preparing solver')
        t0, gpu_post_cond_eps_norm, post_cond, JacDav_func, gpu_func = \
            JacDav.ops_JacDav(params)

        # Set the solver options
        options_JacDav = {
            'maxiters': 100,
            'n_eig': params['n_eig'],
            'target': params['omega']**2,
            'm_max': 40,
            'm_min': 2,
            'tol': params['err_thresh']
        }
        options_lgmres = {'maxiters': 25, 'inner_m': 15, 'outer_k': 3}

        # Solve!
        start_time = time.time()
        print_comm0('start solver')  # t can not be 0

        q, Q, success, err, iters = \
                        JacDav.solve_eig_JacDav( t0 = t0, \
                                                 reporter = rep, \
                                                 JacDav_func = JacDav_func, \
                                                 options_lgmres = options_lgmres, \
                                                 options_JacDav = options_JacDav,
                                                 gpu_func = gpu_func)
        stop_time = time.time()
        print_comm0('time: ' + str(stop_time - start_time))

        # remove the eps_norm
        for Qi in Q:
            gpu_post_cond_eps_norm(Qi)

    if check_success_only:  # Don't write output, just see if we got a success.
        return success

    # Gather results onto root's host memory.
    if solver == 'Jacobi-Davidson':
        Q_result = [[E.get() for E in x] for x in Q]
        result = {  'Q': Q_result, \
                    'q': q,
                    'err': err, \
                    'success': success, \
                    'iters': iters, \
                    'time': (stop_time-start_time)}
    else:
        result = {  'E': [E.get() for E in x], \
                    'err': err, \
                    'success': success, \
                    'iters': iters, \
                    'time': (stop_time-start_time)}
    print_comm0(result['time'])

    # Write results to output file.
    if comm.Get_rank() == 0:
        if solver == 'Jacobi-Davidson':
            for i in range(len(result['Q'])):
                result['Q'][i] = post_cond(
                    result['Q'][i])  # Apply postconditioner
        else:
            result['E'] = post_cond(result['E'])  # Apply postconditioner.
        write_results(name, result)

    return success


def get_parameters(name):
    """ Reads the simulation parameters from the input hdf5 file. """

    if comm.rank == 0:
        f = h5py.File(name + '.grid', 'r')
        files_to_delete = [name + '.grid']

        omega = np.complex128(f['omega_r'][0] + 1j * f['omega_i'][0])
        shape = tuple([int(s) for s in f['shape'][:]])
        n_eig = int(f['n_eig'][0])

        # bloch boundary conditions
        bloch_phase = f['bloch_phase'][...]

        # PEC or PMC boundary conditions
        pemc = f['pemc'][...].astype('int32')

        # get solver
        EM_solvers = ['CG', 'biCGSTAB', 'lgmres', 'Jacobi-Davidson']
        solver = EM_solvers[f['solver'][...]]

        # Function used to read in a 1D complex vector fields.
        get_1D_fields = lambda a: [(f[a+'_'+u+'r'][:] + 1j * f[a+'_'+u+'i'][:]).\
                                astype(np.complex128) for u in 'xyz']

        # Read in s and t vectors.
        s = get_1D_fields('sp')
        t = get_1D_fields('sd')

        # Read in max_iters and err_thresh.
        max_iters = int(f['max_iters'][0])
        err_thresh = float(f['err_thresh'][0])

        # Function used to read in 3D complex vector fields.
        def get_3D_fields(a):
            field = []
            # Check if field data all in one HDF5 file.
            if (a + '_xr') in f:
                for k in range(3):
                    key = a + '_' + 'xyz' [k]
                    field.append(
                        (f[key + 'r'][:] + 1j * f[key + 'i'][:]).astype(
                            np.complex128))
                return field

            for k in range(3):
                key = name + '.' + a + '_' + 'xyz' [k]
                field.append((h5py.File(key + 'r')['data'][:] + \
                        1j * h5py.File(key + 'i')['data'][:]).astype(np.complex128))
                files_to_delete.append(key + 'r')
                files_to_delete.append(key + 'i')
            return field

        e = get_3D_fields('e')  # Permittivity (eps).
        j = get_3D_fields('J')  # Current source.
        m = get_3D_fields('m')  # Permeability (mu).
        x = get_3D_fields('A')  # Initial fields (E0).

        f.close()  # Close file.

        # Delete input files.
        for filename in files_to_delete:
            os.remove(filename)

        # Do some simple pre-computation.
        for k in range(3):
            m[k] = m[k]**-1
            e[k] = omega**2 * e[k]
            j[k] = -1j * omega * j[k]

        params = {'omega': omega, 'shape': shape, 'n_eig': n_eig,\
                  'max_iters': max_iters, 'err_thresh': err_thresh, \
                  's': s, 't': t, 'bloch_phase': bloch_phase, \
                  'pemc': pemc, 'solver': solver}
    else:
        params = None

    params = comm.bcast(params)

    if comm.rank == 0:
        params['e'] = e
        params['m'] = m
        params['j'] = j
        params['x'] = x
    else:
        for field_name in 'emjx':
            params[field_name] = [None] * 3

    return params


def write_results(name, result):
    """ Write out the results to an hdf5 file. """

    my_write = lambda fieldname, data: h5py.File(name + '.' + fieldname, 'w').\
                                            create_dataset('data', data=data)

    if 'q' in list(result.keys()):
        my_write('iter_info', np.array([result['iters']]).astype(np.float32))
        my_write('time_info', np.array([result['time']]).astype(np.float32))
        my_write('qr', np.real(np.array([result['q']])).astype(np.float32))
        my_write('qi', np.imag(np.array([result['q']])).astype(np.float32))

        # Write out the datasets.
        for i in range(len(result['q'])):
            for k in range(3):
                my_write('Q' + str(i) + '_' + 'xyz'[k] + 'r', \
                        np.real(result['Q'][i][k]).astype(np.float32))
                my_write('Q' + str(i)+ '_' + 'xyz'[k] + 'i', \
                        np.imag(result['Q'][i][k]).astype(np.float32))
        my_write = lambda fieldname, data: h5py.File(name + '.' + fieldname, 'w').\
                                                create_dataset('data', data=data)
    else:
        my_write('iter_info', np.array([result['iters']]).astype(np.float32))
        my_write('time_info', np.array([result['time']]).astype(np.float32))

        # Write out the datasets.
        for k in range(3):
            my_write('E_' + 'xyz'[k] + 'r', \
                    np.real(result['E'][k]).astype(np.float32))
            my_write('E_' + 'xyz'[k] + 'i', \
                    np.imag(result['E'][k]).astype(np.float32))


def print_comm0(txt: str):
    if comm.Get_rank() == 0:
        print(txt)


if __name__ == '__main__':  # Allows calls from command line.
    if comm.rank == 0:
        print('start in main')
    simulate(sys.argv[1])  # Specify name of the job.
