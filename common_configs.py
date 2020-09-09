from dataclasses import  dataclass

@dataclass
class TrainerConfig:
    max_step_number : int = 4
    batch_size : int = 512
    maps_iterations: int = 4000
    trials: int = 30
    trace_length: int = 1
    gamma: float = 0.4
    epsilon_initial_value: float = 1.0
    epsilon_decay: float = 0.99987
    epsilon_min: float = 0.3
    conv_model_path: str = "lightning_logs/version_1/checkpoints/240_checkpoint.ckpt"
    movement_model_path: str = None
    conv_trainable: bool = False
    mov_trainable: bool= True
    agent_type: str ='torch'
    mov_type: str = None
    try_gpu: bool = True

@dataclass
class ReplayConfig(TrainerConfig):
    conv_model_path: str = "assets/model_dumps/categorisation/model00000030.h5"
    movement_model_path: str = "mlruns/0/b43f6d400f904918a679401d08045577/artifacts/target_models/data/model.h5"
    network_type: str ='read_movement'
    decision_mode: str= 'network'
    categorisation_mode: str = 'random'

@dataclass
class TrainerA2C:
    batch_size : int = 32
    max_step_number : int = 8
    maps_iterations: int = 4000
    trials: int = 8
    agent_type: str = 'a2c'
    conv_model_path: str = "assets/model_dumps/categorisation/model00000030.h5"
    mov_trainable: bool= True
    conv_trainable: bool = False
    trace_length: int = 4 # lower than max_step number
    gamma: float = 0.8
    actor_lr = 0.001
    critic_lr = 0.001

@dataclass
class ClassicConfConfig:
    source_feature_size : int = 25
    input_window_size : int = 5
    output_window_size : int = 1
    result_output : int = 1 * 3
    dense_size : int = ((25) ** 2)
    dropout_rate : float = 0.3
    batch_size : int = 128
    extended_neighbours : bool = True
