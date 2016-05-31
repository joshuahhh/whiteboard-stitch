import os
import errno
import shutil
import tempfile
import cPickle as pickle


# TODO: Right now, Libraries don't provide any sort of locking. That means that,
# if two users request the same nonexistent resource at around the same time,
# they may both end up computing it. Bummer.

class Library:
    def __init__(self, root_path):
        self.root_path = os.path.realpath(root_path)

    # Gets the value at `paths` out of the library. If it's not found, runs
    # `thunk()` to create it, then stores that at `paths` (and returns it).
    def get(self, thunk, *paths):
        full_path = os.path.join(self.root_path, *paths)
        # print 'GETTING', paths
        try:
            with open(full_path, 'rb') as f:
                # print '  CACHE HIT; LOADING'
                value = pickle.load(f)
                # print '  LOADED'
                return value
        except IOError as exc:
            if exc.errno != errno.ENOENT:
                raise

            # print '  CACHE MISS; CALCULATING'
            value = thunk()
            # print '  CALCULATED; SAVING'

            full_path_dirname = os.path.dirname(full_path)
            if not os.path.exists(full_path_dirname):
                try:
                    os.makedirs(full_path_dirname)
                except OSError as exc:
                    if exc.errno != errno.EEXIST:
                        raise
            with open(full_path, 'wb') as f:
                pickle.dump(value, f)
            # print '  SAVED'
            return value

    # Copies the directory located at `paths` out of the library. If it's not
    # found, runs `thunk(temp_dir)`, which should populate the directory
    # `temp_dir`, then stores that at `paths` (and returns it).
    def get_dir(self, ultimate_destination, thunk, *paths):
        full_path = os.path.join(self.root_path, *paths)

        if not os.path.exists(full_path):
            temp_dir = tempfile.mkdtemp()
            thunk(temp_dir)

            full_path_dirname = os.path.dirname(full_path)
            if not os.path.exists(full_path_dirname):
                try:
                    os.makedirs(full_path_dirname)
                except OSError as exc:
                    if exc.errno != errno.EEXIST:
                        raise

            shutil.copytree(temp_dir, full_path)

        shutil.copytree(full_path, ultimate_destination)
