import threading
import sys
import subprocess
import logging
import time

logger = logging.getLogger(__name__)

def _stream_output(pipe, log_func):
    """Helper function to stream output from a pipe to a log function."""
    try:
        for line in iter(pipe.readline, ''):
            line = line.strip()
            if line:
                log_func(line)
    except (ValueError, IOError):
        pass  # Pipe has been closed


def _log_stderr_line(line):
    """Log a subprocess stderr line at a level inferred from its content.

    COLMAP/glog (and many CLI tools) write their routine INFO/progress output
    to stderr, so logging all stderr at ERROR buries real failures in noise.
    glog lines start with a severity letter + date, e.g. "I0609 22:57:46...".
    """
    head = line[:1]
    if head in "IWEF" and line[1:5].isdigit():
        log_func = {"I": logger.info, "W": logger.warning}.get(head, logger.error)
    else:
        lowered = line.lstrip().lower()
        if lowered.startswith(("error", "fatal", "traceback")):
            log_func = logger.error
        elif lowered.startswith("warn"):
            log_func = logger.warning
        else:
            log_func = logger.info
    log_func(line)


def run_command(command, log_info_output=True, kill_check=None, check_interval=60.0, pipe_stderr=True, pipe_stdout=True):
    """
    Runs a shell command and checks for errors.
    Periodically calls kill_check to determine if the process should be terminated.
    
    Args:
        command: The command to run as a list of strings
        log_info_output: Whether to log stdout as info (default: True)
        kill_check: A function that returns True if the process should be terminated
        check_interval: How often to call kill_check in seconds (default: 1.0)
        
    Returns:
        True if the process completed normally, False if it was terminated by kill_check
    """
    try:
        logger.info(f"Running command: {' '.join(command)}")
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE if pipe_stdout else subprocess.DEVNULL,
            stderr=subprocess.PIPE if pipe_stderr else subprocess.DEVNULL,
            universal_newlines=True,
            bufsize=1  # Line buffered
        )
        
        # Create threads to stream output in real-time
        stdout_thread = None
        stderr_thread = None
        if pipe_stdout:
            stdout_thread = threading.Thread(
                target=_stream_output, 
                args=(process.stdout, logger.info if log_info_output else lambda x: None)
            )
            stdout_thread.daemon = True
            stdout_thread.start()
        if pipe_stderr:
            stderr_thread = threading.Thread(
                target=_stream_output,
                args=(process.stderr, _log_stderr_line)
            )
            stderr_thread.daemon = True
            stderr_thread.start()
        
        # Wait for process to complete or kill_check to return True
        killed = False
        process_check_interval = 0.1  # 100ms check for process completion
        kill_check_counter = 0
        kill_check_frequency = int(check_interval / process_check_interval)  # How many small intervals equal one kill check

        while process.poll() is None:  # While process is still running
            kill_check_counter += 1
            if kill_check is not None and kill_check_counter >= kill_check_frequency:
                kill_check_counter = 0
                if kill_check():
                    logger.info(f"Kill check returned True, terminating process (PID: {process.pid})")
                    try:
                        process.terminate()
                        # Give it a few seconds to terminate gracefully
                        for _ in range(5):
                            if process.poll() is not None:
                                break
                            time.sleep(0.5)
                        
                        # If still running, force kill
                        if process.poll() is None:
                            logger.info(f"Process didn't terminate gracefully, killing (PID: {process.pid})")
                            process.kill()
                    except Exception as e:
                        logger.error(f"Error killing process: {e}")
                    
                    killed = True
                    break
            
            time.sleep(process_check_interval)  # Use small sleep to detect process completion quickly
        
        # Wait for output threads to finish
        if stdout_thread:
            stdout_thread.join(timeout=1)
        if stderr_thread:
            stderr_thread.join(timeout=1)
        
        if killed:
            logger.info(f"Process was terminated by kill_check")
            return False
            
        if process.returncode != 0:
            raise subprocess.CalledProcessError(process.returncode, command)
        
        return True
            
    except subprocess.CalledProcessError as e:
        logger.error(f"Error running command: {e}")
        raise Exception(f"Error running command: {e}")