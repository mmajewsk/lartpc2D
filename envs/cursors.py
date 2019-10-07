from envs.regions import BaseRegion, Region2D, Region3D
import numpy as np

class Cursor:
    def __init__(self, output_size, input_size, movement_size=3):
        """

		:param output_size: output of the network size (the categorisation result, like one 1)
		:param input_size:  input to the network size (usually 3x3x3)
		"""
        self.__center = None
        self.region_output = self.__class__.region_cls(output_size)
        self.region_input = self.__class__.region_cls(input_size)
        self.region_movement =self.__class__.region_cls(movement_size)

    @property
    def current_center(self) -> np.ndarray:
        return self.__center

    @current_center.setter
    def current_center(self, val: np.ndarray):
        self.__center = val
        self.__update_indeces()

    @property
    def current_indeces(self) -> np.ndarray:
        if self.__center is None:
            self.__update_indeces()
        return self.__indeces

    def __update_indeces(self):
        self.__indeces = (self.current_center + self.region_movement.range).T

    def get_range(self, arr: np.ndarray, center: np.ndarray = None, region='input') -> np.ndarray:
        if center is None:
            center = self.current_center
        region = self.region_input if region == 'input' else self.region_output
        boundaries_low, boundaries_high = (region.r_low, region.r_high)
        return self.__class__.region_cls.get_range(arr, center, boundaries_low, boundaries_high)

    def set_range(self, arr: np.ndarray, value: np.ndarray, center: np.ndarray = None, region='output'):
        if center is None:
            center = self.current_center
        region = self.region_output if region == 'output' else self.region_input
        boundaries_low, boundaries_high = (region.r_low, region.r_high)
        return self.__class__.region_cls.set_range(arr, value, center, boundaries_low, boundaries_high)

def _cursor_factory_by_region(RegionClass):
    class TmpCursor(Cursor):
        region_cls = RegionClass
    return TmpCursor

Cursor2D = _cursor_factory_by_region(Region2D)
Cursor3D = _cursor_factory_by_region(Region3D)
