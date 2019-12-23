from dataclasses import  dataclass

@dataclass
class GameConfig:
    max_step_number : int = 12
    batch_size : int = 128
    maps_iterations: int = 4000
    trials: int = 8
    trace_length: int = 1
    gamma: float = 0.8
    epsilon_initial_value = 1.0
    epsilon_decay: float = 0.9987
    epsilon_min: float = 0.5


@dataclass
class ClassicConfConfig:
    #input_params
    source_feature_size : int = 9
    input_window_size : int = 3
    #output_params
    output_window_size : int = 1
    result_output : int = 1 * 3
    # output_window_size : int = 1
    # result_output : int = 1 * 3
    #other_params
    dense_size : int = ((9) ** 2)
    dropout_rate : float = 0.0
    batch_size : int = 128
    extended_neighbours : bool = True
