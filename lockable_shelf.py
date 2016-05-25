import shelve
from threading import Lock
from contextlib import contextmanager


# This wraps a Python 'shelf' (key-value store in a file) in a lock, so that
# multiple threads can use it in parallel. Note that the threads need to use the
# same LockableShelf object, since that's where the lock lives!

class LockableShelf:
    def __init__(self, filename):
        self.filename = filename
        self.lock = Lock()

    @contextmanager
    def open(self):
        self.lock.acquire()
        self.current_shelf = shelve.open(self.filename)
        yield self.current_shelf
        self.current_shelf.close()
        self.lock.release()
