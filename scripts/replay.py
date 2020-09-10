from pathlib import  Path
import numpy as np
from lartpc_game.viz import MixedModelVisualisation
from lartpc_game.game.game_ai import Lartpc2D
from lartpc_game.data import LartpcData
from reinforcement_learning.torch_agents import TorchAgent, StateToObservables, ToFlat1D, ToTorchTensorTuple
from reinforcement_learning.torch_agents import ToNumpy, ModelOutputToAction
from common_configs import ReplayConfig, ClassicConfConfig
import torch

def simple_replay(data_path):
    config = ReplayConfig()
    classic_config = ClassicConfConfig()
    data_generator = LartpcData.from_path(data_path)
    result_dimensions = 3
    env = Lartpc2D(result_dimensions , max_step_number=config.max_step_number)
    policy = torch.load(config.model_path/"policy")
    target = torch.load(config.model_path/"target")
    agent = TorchAgent(env)
    agent.set_models(policy, target)
    vis = MixedModelVisualisation(env)

    for iterate_maps in range(config.maps_iterations):
        map_number = np.random.randint(0, len(data_generator))
        env.detector.set_maps(*data_generator[map_number])
        for iterate_tries in range(config.trials):
            env.start()
            for model_run_iteration in range(env.max_step_number):
                current_observation = env.get_observation()
                model_state = StateToObservables()(current_observation)
                model_state = ToFlat1D()(model_state)
                model_state_tensor = ToTorchTensorTuple()(model_state)
                model_action = agent.create_action(model_state_tensor, use_epsilon=False)
                model_action = ToNumpy()(model_action)
                game_action = ModelOutputToAction()(model_action, agent.action_settings)
                state = env.step(game_action)
                vis.obs_action(current_observation, game_action)
                vis.update(0)
                if state.done:
                    break

if __name__ == "__main__":
    data_path = '../assets/dump'
    #data_path = '/home/mwm/repositories/lartpc/lartpc_notebooks/Blog/content/dump'
    simple_replay(data_path)
