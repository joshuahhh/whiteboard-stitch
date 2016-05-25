import os
import cPickle as pickle


class Library:
    def __init__(self, root_path):
        self.root_path = os.path.realpath(root_path)

    def get(self, thunk, *paths):
        full_path = os.path.join(self.root_path, *paths)
        try:
            with open(full_path, 'rb') as f:
                return pickle.load(f)
        except OSError:
            value = thunk()
            with open(full_path, 'wb') as f:
                pickle.dump(value, f)
            return value
