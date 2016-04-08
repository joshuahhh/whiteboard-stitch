import sys
import cProfile
from contextlib import contextmanager
from time import time


@contextmanager
def profile(filename=None):
    if filename:
        pr = cProfile.Profile()
        pr.enable()
    now = time()
    yield
    print (time() - now), 'seconds'
    sys.stdout.flush()
    if filename:
        pr.disable()
        pr.dump_stats(filename)
