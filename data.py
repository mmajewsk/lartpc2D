import scipy.sparse
import numpy as np
from pathlib import Path
from collections import  OrderedDict


class LartpcData:

    def __init__(self, source_data: OrderedDict, target_data: OrderedDict):
        self.source_dict = source_data
        self.target_dict = target_data
        self.index = 0
        self.length = len(source_data)

    @staticmethod
    def from_path(data_filepath):
        data_filepath = Path(data_filepath)
        source_files_list = [x for x in data_filepath.iterdir() if 'image' in x.name]
        source_files_range = [ int(x.name[len('image'):].split('.')[0]) for x in source_files_list]
        source_dict = OrderedDict(sorted(zip(source_files_range,source_files_list), key=lambda x:x[0]))
        target_files_list = [x for x in data_filepath.iterdir() if 'label' in x.name]
        target_files_range = [ int(x.name[len('label'):].split('.')[0]) for x in target_files_list]
        target_dict = OrderedDict(sorted(zip(target_files_range,target_files_list), key=lambda x:x[0]))
        return LartpcData(source_dict, target_dict)

    def __len__(self):
        return self.length

    def _read_array(self, npz_path):
        return scipy.sparse.load_npz(npz_path).todense()

    def __getitem__(self, item):
        s_path, s_target = self.source_dict[item], self.target_dict[item]
        source, target = self._read_array(s_path), self._read_array(s_target)
        return source, target

    def get_range(self, min, max):
        return LartpcData(self.source_dict[min:max], self.target_dict[min:max])

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

