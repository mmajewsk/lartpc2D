from dataclasses import  dataclass

@dataclass
class LConfig:
    max_step_number : int = 12
    batch_size : int = 128
    maps_iterations: int = 4000
    trials: int = 8
    trace_length: int = 1
    gamma: float = 0.8
    epsilon_initial_value = 1.0
    epsilon_decay: float = 0.9987
    epsilon_min: float = 0.5