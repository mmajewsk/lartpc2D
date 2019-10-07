import scipy.sparse
import numpy as np
from pathlib import Path
from collections import  OrderedDict


class LartpcData:
    def __init__(self, data_filepath):
        self.data_filepath = Path(data_filepath)
        self.source_files_list = [x for x in self.data_filepath.iterdir() if 'image' in x.name]
        self.source_files_range = [ int(x.name[len('image'):].split('.')[0]) for x in self.source_files_list]
        self.source_dict = OrderedDict(sorted(zip(self.source_files_range,self.source_files_list), key=lambda x:x[0]))
        self.target_files_list = [x for x in self.data_filepath.iterdir() if 'label' in x.name]
        self.target_files_range = [ int(x.name[len('label'):].split('.')[0]) for x in self.target_files_list]
        self.target_dict = OrderedDict(sorted(zip(self.target_files_range,self.target_files_list), key=lambda x:x[0]))
        self.index = 0
        self.length = len(self.source_files_range)

    def __len__(self):
        return self.length

    def _read_array(self, npz_path):
        return scipy.sparse.load_npz(npz_path).todense()

    def __getitem__(self, item):
        s_path, s_target = self.source_dict[item], self.target_dict[item]
        source, target = self._read_array(s_path), self._read_array(s_target)
        return source, target

    def random(self):
        return self[np.random.randint(0, len(self))]

    def current(self):
        return self[self.index]

    def __next__(self):
        if self.index > self.length:
            raise StopIteration
        else:
            self.index += 1
            return self[self.index-1]

