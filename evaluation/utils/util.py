import re
import signal

class TimeoutException(Exception):
    pass

def timeout_handler(signum, frame):
    raise TimeoutException()

def safe_search(pattern, string, timeout=1):
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(timeout)
    try:
        result = re.search(pattern, string)
    except TimeoutException:
        result = None
    finally:
        signal.alarm(0)
    return result