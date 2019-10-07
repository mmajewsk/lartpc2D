from envs.cursors import Cursor2D
from collections import OrderedDict
import numpy as np
import pandas as pd
from abc import abstractmethod, ABC

class BaseEnvironment(ABC):
    def __init__(self):
        self.source_map = None
        self.target_map = None
        self.result_map = None

    def read_source_nonzero_indeces(self):
        self.nonzero_indeces = np.where(self.source_map>0.)

    @property
    @abstractmethod
    def dimension_list(self):
        #eg. return ['x','y','z']
        pass

    def create_nonzero_df(self, nonzero_indeces):
        df_dict = {}
        for dim, nonzero_ax in zip(self.dimension_list, nonzero_indeces):
            df_dict[dim] = nonzero_ax
        self.nonzero_df = pd.DataFrame(df_dict)
        self.nonzero_df['touched'] = 0

    def get_random_position(self):
        row = self.nonzero_df.sample(1)
        return row[self.dimension_list].values[0]

    def set_map(self, source, target, result=None):
        self.source_map = source
        self.target_map = target
        if result is not None:
            self.result_map = result
        else:
            self.result_map = np.zeros(self.target_map.shape)

    def get_map(self):
        return self.source_map, self.target_map, self.result_map


class Environment2D(BaseEnvironment):

    @property
    def dimension_list(self):
        return ['x','y']

class Action2D:
    def __init__(self, movement_vector, put_data):
        self.movement_vector = movement_vector
        self.put_data = put_data

class Action2DFactory:
    def __init__(self, cursor: Cursor2D ):
        self.cursor = cursor
        mov_range = cursor.region_movement.range
        self.possible_movement = mov_range
        self.possible_data = cursor.region_input.range
        self.movement_size = cursor.region_movement.basic_block_size-1
        self.data_size =  cursor.region_input.basic_block_size

    def from_flat(self, flat_array: np.ndarray) -> Action2D:
        assert len(flat_array)==self.data_size+self.movement_size, "Incorrect array length"
        flat_movement, flat_data = flat_array[:self.movement_size], flat_array[self.movement_size:]
        assert np.sum(flat_movement) == 1, "ambigiuos movement choice"
        middle_index = self.cursor.region_movement.r_low*(self.cursor.region_movement.size+1)
        a,  = np.where(flat_movement==1.0)
        if a>middle_index:
            a+=1
        unflat_data = flat_data.reshape(self.cursor.region_input.shape)
        return Action2D(self.possible_movement[a], unflat_data)

    def to_flat(self, action: Action2D) -> np.ndarray:
        flat_movement = np.zeros(self.movement_size, self.possible_movement.dtype)
        a, = np.where(self.possible_movement==action.movement_vector)
        flat_movement[a] = 1.0
        flat_data = action.put_data.flatten()
        return np.concatenate([flat_movement, flat_data])


class Observation2D:
    def __init__(self, source_oservation, result_observation):
        self.source_obsercation = source_oservation
        self.result_observation = result_observation

def rewards(source_cursor, result_cursor):
    # assumes result cursor is binary
    discovered = np.sum((source_cursor != 0.0)-result_cursor)
    empty = np.sum((source_cursor==0.0))*-1.1
    # 2:
    # + Discovers undiscovered
    # - Doesn't go to empty
    return OrderedDict(discovered=discovered, empty=empty)

class Game2D:

    def __init__(self, env: Environment2D):
        self.env = env
        self.cursor = Cursor2D(output_size=3, input_size=3, movement_size=3)


    def start(self):
        self.cursor.current_center = self.env.get_random_position()

    def act(self, action: Action2D):
        self.cursor.set_range(self.env.result_map, action.put_data)
        self.cursor.current_center += action.movement_vector

    def get_observation(self) -> Observation2D:
        source_curs = self.cursor.get_range(self.env.source_map)
        result_curs = self.cursor.get_range(self.env.result_map)
        obs = Observation2D(source_curs, result_curs)
        return obs

    def reward(self):
        rewards_dict = rewards(
            source_cursor=self.cursor.get_range(self.env.source_map),
            result_cursor=self.cursor.get_range(self.env.result_map),
        )
        rewards_list = list(rewards_dict.values())
        return np.sum(rewards_list)


