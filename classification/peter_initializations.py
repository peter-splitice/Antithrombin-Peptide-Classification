
import sys
import logging

# Creating a logger to record and save information.
def log_files(logname):
    """
    Create the meachanism for which we log results to a .log file.

    Parameters
    ----------
    logname:

    Returns
    -------
    logger:  The logger object we create to call on in other functions. 
    """

    # Instantiate the logger and set the formatting and minimum level to DEBUG.
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s')

    # Display the logs in the output
    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setLevel(logging.DEBUG)
    stdout_handler.setFormatter(formatter)

    # Write the logs to a file
    file_handler = logging.FileHandler(logname)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)

    # Adding the file and output handlers to the logger.
    logger.addHandler(file_handler)
    logger.addHandler(stdout_handler)

    return logger
