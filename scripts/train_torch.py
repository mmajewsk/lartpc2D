import os
import multiprocessing
from lartpc_game.game.game_ai import Lartpc2D
from pathlib import Path
import copy
from lartpc_game.agents.settings import Observation2DSettings
import lartpc_game.data
import numpy as np
import tempfile
# from reinforcement_learning.agents import AgentFactory
from lartpc_game.agents.settings import Action2DSettings
#from viz import  Visualisation
from common_configs import TrainerA2C, TrainerConfig, ClassicConfConfig
from logger import Logger, MLFlowLoggerTorch, NeptuneLogger, get_neptune_logger
import pytorch_lightning as pl

from reinforcement_learning.torch_agents import TorchAgent
from reinforcement_learning.torch_agents import ModelOutputToAction, StateToObservables, ToTorchTensorTuple
from reinforcement_learning.torch_agents import ToNumpy, ToFlat1D, ToDevice
from reinforcement_learning.torch_networks import CombinedNetworkTorch, MovementTorch, MovementBinarised
from scripts.classic_conv_lt import CatLt
from tqdm.auto import tqdm, trange
import warnings

import torch
import dataclasses

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
    other_params = dict(dense_size=64, dropout_rate=0.4)
    model_params = dict(
        input_parameters=input_parameters,
        output_parameters = output_parameters,
        other_params = other_params,
    )
    return model_params

def get_networks(env, config, classic_config, result_dimensions):
    model_settings = create_model_params(env)
    if config.movement_model_path is None:
        if config.mov_type is None:
            CLS = MovementTorch
        elif config.mov_type == 'binarised':
            CLS = MovementBinarised
        mov_net = CLS(
            source_in_size=model_settings['input_parameters']['source_feature_size'],
            result_in_size=model_settings['input_parameters']['result_feature_size'],
            moves_out_size=model_settings['output_parameters']['possible_moves'],
            dense_size = model_settings['other_params']['dense_size'],
            dropout_rate = model_settings['other_params']['dropout_rate']
        )
    else:
        mov_net = torch.load(config.movement_model_path, map_location=torch.device('cpu'))


    cat_net = CatLt(
        dense_size=classic_config.dense_size,
        dropout_rate=classic_config.dropout_rate,
        result_feature_size=result_dimensions
    )
    if config.conv_model_path is not None:
        data = torch.load(
            config.conv_model_path,
            map_location=torch.device('cpu')
        )
        cat_net.load_state_dict(data['state_dict'])
    if not config.conv_trainable:
        for param in cat_net.parameters():
            param.requires_grad = False

    if not config.mov_trainable:
        for param in mov_net.parameters():
            param.requires_grad = False
    # __import__("pdb").set_trace()

    return mov_net, cat_net

def simple_learn(data_path):
    config = TrainerConfig()
    classic_config = ClassicConfConfig()
    DEBUG = os.environ.get('DEBUG')
    if DEBUG is not None and DEBUG == '1':
        config.batch_size=8
        config.trials=4
        config.max_step_number=4
        config.trace_length = 1
    if config.try_gpu:
        device_str = "cuda:0" if torch.cuda.is_available() else "cpu"
    else:
        device_str = "cpu"
    device = torch.device(device_str)
    print("Using device: {} ".format(device))
    logger = Logger()
    mlf_logger = MLFlowLoggerTorch(dataclasses.asdict(config))
    mlf_logger.start()
    mlf_logger.log_config(config)
    params = {**dataclasses.asdict(config), **dataclasses.asdict(classic_config)}
    nep_logger = NeptuneLogger(params)
    nep_logger.start()
    nep_logger.package.append_tag(device_str)
    data_generator = lartpc_game.data.LartpcData.from_path(data_path)
    result_dimensions = 3
    # @TODO env.set_maps(*data_generator[3])
    env = Lartpc2D(result_dimensions , max_step_number=config.max_step_number)
    agent = TorchAgent(env)
    mov_net, cat_net = get_networks(env, config,  classic_config, result_dimensions)
    mov_net.to(device)
    cat_net.to(device)
    policy = CombinedNetworkTorch(mov_net, cat_net, cat_trainable=config.conv_trainable)
    target = CombinedNetworkTorch(copy.deepcopy(mov_net), copy.deepcopy(cat_net))
    agent.set_models(policy, target)
    for iterate_maps in range(config.maps_iterations):
        map_number = np.random.randint(1000, len(data_generator))
        env.detector.set_maps(*data_generator[map_number])
        iterations = []
        epsilon_hist = []
        rewards = []
        for iterate_tries in trange(config.trials):
            env.start()
            trial_run_history = []
            for model_run_iteration in range(env.max_step_number):
                current_state = env.get_state()
                epsilon_hist.append(agent.epsilon.value)
                model_state = StateToObservables()(current_state.obs)
                model_state = ToFlat1D()(model_state)
                model_state_tensor = ToTorchTensorTuple()(model_state)
                model_state_tensor = ToDevice(device)(model_state_tensor)
                model_action = agent.create_action(model_state_tensor)
                model_action = ToDevice('cpu')(model_action)
                model_action = ToNumpy()(model_action)
                game_action = ModelOutputToAction()(model_action, agent.action_settings)
                # game_action.type_check()
                new_state = env.step(game_action)
                rewards.append(new_state.reward)
                # new_state.type_check()
                agent.memory.add(current_state, model_action,new_state)
                if new_state.done:
                    break

            # iterations.append(trial_run_history.copy())
            if agent.enough_samples_to_learn() and iterate_tries % 2 ==0:
                h1 = agent.train_agent(device)
                logger.add_train_history(h1)
                mlf_logger.log_history(h1)
                nep_logger.log_history(h1)
                agent.target_train()
        nep_logger.log_metrics('reward', rewards)
        nep_logger.log_metrics('epsilon', epsilon_hist)
        if iterate_maps % 30 == 0:
            nep_logger.log_model(policy, 'policy')
            nep_logger.log_model(target, 'target')
        # logger.game_records(dict(map=map_number, data=iterations))
        # mlf_logger.log_game(map_number, iterations)
        # if agent.enough_samples_to_learn() and iterate_maps%4==0:
            # logger.plot()
        # agent.dump_models(Path('assets/model_dumps'))
        # logger.dump_log()
        # mlf_logger.log_model(agent)



if __name__ == "__main__":
    data_path = os.environ['DATA_PATH']
    #data_path = '/home/mwm/repositories/content/dump'  # home cluster
    #ftest_draw_random_cursor(data_path)
    simple_learn(data_path)
