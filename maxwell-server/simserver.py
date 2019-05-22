""" Simulation server for Maxwell.

    Executes uploaded jobs.

    Consists of an infinite loop which does the following:
    0.  Check that GCE is not running.
    1.  Find the oldest job.
    2.  Run the solver on it.
    3.  Repeat.

"""

import logging
import os
import os.path
import shlex
import subprocess
import sys
import time

import maxwell_config
import pycuda.driver
import unbuffered

LOG_FORMAT = '[%(asctime)-15s][%(levelname)s][%(module)s][%(funcName)s] %(message)s'
logging.basicConfig(format=LOG_FORMAT)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def get_num_gpus():
    pycuda.driver.init()
    return pycuda.driver.Device.count()

def check_process_running(pname):
    # Check if process with name 'pname' is running
    p1 = subprocess.Popen(['ps', 'ax'], 
                          stdout=subprocess.PIPE)
    p2 = subprocess.Popen(['grep', pname],
                          stdin=p1.stdout, stdout=subprocess.PIPE)
    p3 = subprocess.Popen(['grep', '-v', 'grep'],
                           stdin=p2.stdout, stdout=subprocess.PIPE)
    return p3.communicate()[0].find(b'\n') > -1

def find_oldest_job():
    req = maxwell_config.list_requests() # Get the requests.
    if not req:
        return None

    req_with_time = {}
    for r in req:
        req_with_time[r] = os.stat(os.path.join(maxwell_config.path, r)).st_ctime

    oldest_req = min(req_with_time) # Run this job.
    os.remove(os.path.join(maxwell_config.path, oldest_req))
    return oldest_req[:-len('.request')]

def main():
    sys.stdout = unbuffered.Unbuffered(sys.stdout)

    path_to_solver_dir = os.path.abspath(__file__).replace(
                            __file__.split('/')[-1], 'maxwell-solver') + '/'
    logger.info('Solver directory set to {0}'.format(path_to_solver_dir))

    # Determine number of GPUs on system
    num_gpus = get_num_gpus()
    logger.info('Number of GPUs detected on system: {0}'.format(num_gpus))

    # Determine number of GPUs to use per solve (user input)
    if len(sys.argv) > 1:
        gpus_per_solve = int(sys.argv[1])
    else:
        gpus_per_solve = 2
    logger.info('Number of GPUs used per solve: {0}'.format(gpus_per_solve))

    if gpus_per_solve > num_gpus:
        raise ValueError('Number of GPUs is {0}, but number of GPUs requested '
                         'per solve is {1}, exceeding the number of available '
                         'GPUs.'.format(num_gpus, gpus_per_solve))

    # Generate GPU groupings.
    # Groupings take the form gpu1,gpu2,gpu3...
    # e.g. with 8 GPUs and 3 GPUs per solve we have:
    # solve_gpus = ['0,1,2','3,4,5']
    num_solves  = num_gpus // gpus_per_solve
    solve_gpus  = []
    for i in range(0, num_solves):
        start_num = i * gpus_per_solve
        solve_gpus.append(','.join(
             str(j) for j in range(start_num, start_num + gpus_per_solve)))

    # Managing loop
    solve_obj   = [None]*num_solves
    solve_paths = ['']*num_solves
    out_files   = [None]*num_solves

    logger.info('Ready to accept simulations.')

    while True:
        time.sleep(1)

        # Check for solve completion
        for i in range(len(solve_obj)):
            if solve_obj[i] and solve_obj[i].poll() is not None:
                logger.info('Simulation {0} ended with code {1}'.format(
                    i, solve_obj[i].returncode))

                # Close output log file
                out_files[i].close()

                # Used to let user know that files can be downloaded.
                time.sleep(0.5)
                filepath = os.path.join(maxwell_config.path, solve_paths[i]
                                        + '.finished')
                f = open(filepath, 'w')
                logger.debug('Writing finished file at {0}'.format(filepath))
                f.write('{}'.format(solve_obj[i].returncode))
                f.close()

                # Delete old job
                solve_obj[i] = None
                out_files[i] = None
                solve_paths[i] = ''

        # Ensure that GCE is not running
        if check_process_running('job_manager'):
            continue

        # Check for and start new solves
        for i in range(len(solve_obj)):
            if solve_obj[i]:
                continue
            solve_paths[i] = find_oldest_job()
            if solve_paths[i]:
                logger.info('Solving {0} as simulation {1}'.format(
                    solve_paths[i], i))

                tmp_env = os.environ.copy()
                tmp_env['CUDA_VISIBLE_DEVICES'] = solve_gpus[i]
                #logger.debug('Environment provided: {0}'.format(tmp_env))

                out_file_log = os.path.join(maxwell_config.path,
                                            solve_paths[i] + '.log')
                out_files[i] = open(out_file_log, 'w')
                logger.debug('Outputing to log file: {0}'.format(out_file_log))

                command = ('mpirun -n ' + str(gpus_per_solve) + ' python ' +
                           path_to_solver_dir + 'fdfd.py ' +
                           os.path.join(maxwell_config.path, solve_paths[i]))
                logger.debug('Running command {0}'.format(command))

                solve_obj[i] = subprocess.Popen(shlex.split(command),
                                   stdout=out_files[i],
                                   stderr=subprocess.STDOUT,
                                   env=tmp_env)

if __name__ == '__main__':
    main()
