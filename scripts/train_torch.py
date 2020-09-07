from lartpc_game.game.game_ai import Lartpc2D
from pathlib import Path
import copy
from lartpc_game.agents.settings import Observation2DSettings
import lartpc_game.data
import numpy as np
# from reinforcement_learning.agents import AgentFactory
from lartpc_game.agents.settings import Action2DSettings
#from viz import  Visualisation
from common_configs import TrainerA2C, TrainerConfig, ClassicConfConfig
from logger import Logger, MLFlowLoggerTorch, NeptuneLogger

from reinforcement_learning.torch_agents import TorchAgent, ModelOutputToAction, StateToObservables, ToTorchTensorTuple, ToNumpy, ToFlat1D
from reinforcement_learning.torch_networks import CombinedNetworkTorch, MovementTorch
from scripts.classic_conv_lt import CatLt
from tqdm.auto import tqdm, trange
import warnings

# warnings.simplefilter("error")

def create_model_params(env: Lartpc2D):
    action_settings = env.action_settings
    observation_settings = env.observation_settings
    input_parameters = dict(
        source_feature_size =observation_settings.cursor.region_source_input.basic_block_size, # size of input window
        result_feature_size = np.prod(observation_settings.result_shape), # this is a space where we
    )
    output_parameters= dict(
        possible_moves = action_settings.movement_size, #where it can move
    )
    other_params = dict(dense_size=32, dropout_rate=0.2)
    model_params = dict(
        input_parameters=input_parameters,
        output_parameters = output_parameters,
        other_params = other_params,
    )
    return model_params


def simple_learn(data_path):
    config = TrainerConfig()
    logger = Logger()
    mlf_logger = MLFlowLoggerTorch(config)
    mlf_logger.start()
    mlf_logger.log_config(config)
    nep_logger = NeptuneLogger(config)
    nep_logger.start()
    classic_config = ClassicConfConfig()
    data_generator = lartpc_game.data.LartpcData.from_path(data_path)
    result_dimensions = 3
    # @TODO env.set_maps(*data_generator[3])
    env = Lartpc2D(result_dimensions , max_step_number=config.max_step_number)
    model_settings = create_model_params(env)
    agent = TorchAgent(env)
    mov_net = MovementTorch(
        source_in_size=model_settings['input_parameters']['source_feature_size'],
        result_in_size=model_settings['input_parameters']['result_feature_size'],
        moves_out_size=model_settings['output_parameters']['possible_moves'],
        dense_size = model_settings['other_params']['dense_size'],
        dropout_rate = model_settings['other_params']['dropout_rate']
    )
    cat_net = CatLt(
        dense_size=classic_config.dense_size,
        dropout_rate=classic_config.dropout_rate,
        result_feature_size=result_dimensions
    )
    policy = CombinedNetworkTorch(mov_net, cat_net)
    target = CombinedNetworkTorch(copy.deepcopy(mov_net), copy.deepcopy(cat_net))
    agent.set_models(policy, target)
    for iterate_maps in range(config.maps_iterations):
        map_number = np.random.randint(1000, len(data_generator))
        env.detector.set_maps(*data_generator[map_number])
        iterations = []
        for iterate_tries in trange(config.trials):
            env.start()
            trial_run_history = []
            for model_run_iteration in range(env.max_step_number):
                current_state = env.get_state()
                model_state = StateToObservables()(current_state.obs)
                model_state = ToFlat1D()(model_state)
                model_state_tensor = ToTorchTensorTuple()(model_state)
                model_action = agent.create_action(model_state_tensor)
                model_action = ToNumpy()(model_action)
                game_action = ModelOutputToAction()(model_action, agent.action_settings)
                # game_action.type_check()
                new_state = env.step(game_action)
                # new_state.type_check()
                trial_run_history.append((current_state, game_action, new_state))
                if new_state.done:
                    break
            agent.memory.add(trial_run_history)
            iterations.append(trial_run_history.copy())
            if agent.enough_samples_to_learn():
                h1 = agent.train_agent()
                logger.add_train_history(h1)
                mlf_logger.log_history(h1)
                nep_logger.log_history(h1)
                nep_logger.log_metrics('reward', [state.reward for _, _, state in trial_run_history])
                agent.target_train()
        # logger.game_records(dict(map=map_number, data=iterations))
        # mlf_logger.log_game(map_number, iterations)
        # if agent.enough_samples_to_learn() and iterate_maps%4==0:
            # logger.plot()
        # agent.dump_models(Path('assets/model_dumps'))
        # logger.dump_log()
        # mlf_logger.log_model(agent)


if __name__ == "__main__":
    data_path = '../assets/dump'
    #data_path = '/home/mwm/repositories/content/dump'  # home cluster
    #ftest_draw_random_cursor(data_path)
    simple_learn(data_path)
