#!/usr/bin/python
""" Removes the temporary files for Maxwell automatically.

The temporary files in Maxwell tend to build up over time and these files
eventually take up enough space to warrant removing them. Rather than manually
performing this task, this script automatically removes that are older than
a certain date (default: 7 days). This script is intended to be placed in
cron folder (be sure to give the script executable permissions).
"""
import datetime
import logging
import os

# Temporary maxwell server files directory to sweep.
MAXWELL_SERVER_FILES_DIR = os.environ["MAXWELL_SERVER_FILES"]
# Number of days to retain temporary files.
DELETE_THRESHOLD_DAY = 7
# Logging format to use.
LOG_FORMAT = '[%(asctime)-15s][%(levelname)s][%(module)s][%(funcName)s] %(message)s'
# Place to store logs.
LOG_LOCATION = '/home/maxwell/maxwell-sweeper.log'

# Append to log file so that script can be run multiple times.
logging.basicConfig(filename=LOG_LOCATION, filemode='a',
                    format=LOG_FORMAT, level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    logger.info('Beginning sweep...')
    # Keep track of number of deleted files.
    deleted_files = 0
    # List all the files
    for filename in os.listdir(MAXWELL_SERVER_FILES_DIR):
        fullpath = os.path.join(MAXWELL_SERVER_FILES_DIR, filename)
        try:
            # Get the last modified time and compare against current time.
            # Note that it is safer to retrieve the last modified timestamp
            # before obtaining the current timestamp to avoid the situation
            # where `last_modified > now`.
            last_modified = datetime.datetime.fromtimestamp(
                os.path.getmtime(fullpath))
            now = datetime.datetime.today()

            delta_time = now - last_modified
            if delta_time.days > DELETE_THRESHOLD_DAY:
                logging.debug('Removing {0} ({1} days old)...'.format(
                    fullpath, delta_time.days))
                os.remove(fullpath)
                deleted_files += 1
        except:
            logger.exception('Error handling {0}'.format(fullpath))
    logger.info('Sweep finished. Removed {0} files.'.format(deleted_files))

if __name__ == '__main__':
    main()
