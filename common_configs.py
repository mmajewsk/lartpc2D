from dataclasses import  dataclass

@dataclass
class TrainerConfig:
    max_step_number : int = 12
    batch_size : int = 128
    maps_iterations: int = 4000
    trials: int = 8
    trace_length: int = 1
    gamma: float = 0.8
    epsilon_initial_value = 1.0
    epsilon_decay: float = 0.9987
    epsilon_min: float = 0.5
    conv_model_path: str = "assets/model_dumps/categorisation/model00000030.h5"
    movement_model_path: str = "assets/model_dumps/target_model.h5"
    conv_trainable = False
    mov_trainable = False
    network_type='movement'
    #categorisation_mode = 'network'
    #decision_mode = 'network'
    decision_mode = 'network'
    categorisation_mode = 'random'


@dataclass
class ClassicConfConfig:
    #input_params
    source_feature_size : int = 25
    input_window_size : int = 5
    #output_params
    output_window_size : int = 1
    result_output : int = 1 * 3
    # output_window_size : int = 1
    # result_output : int = 1 * 3
    #other_params
    dense_size : int = ((25) ** 2)
    dropout_rate : float = 0.0
    batch_size : int = 128
    extended_neighbours : bool = True
