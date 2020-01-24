from rl_environments.game import Environment2D, Game2D
from pathlib import  Path
from actors.observations import Observation2DFactory
import data
import numpy as np
from actors.models import Actor
from actors.actions import Action2DFactory
from rl_environments.dims import neighborhood2d
from actors.networks import ParameterBasedNetworks, load_model, create_network_factory
#from viz import  Visualisation
from common_configs import TrainerConfig
from logger import Logger, MLFlowLogger


def ftest_draw_random_cursor(data_path):
    data_generator = data.LartpcData.from_path(data_path)
    env = Environment2D()
    env.set_map(*data_generator[3])
    game = Game2D(env)
    game.cursor.current_center = np.array([60,120])
    vis = Visualisation(game)
    vis.update()
    possible_choices = neighborhood2d(3)
    for i in range(400):
        choice = np.random.randint(0,len(possible_choices))
        vector = possible_choices[choice]
        print(vector)
        prev = game.cursor.current_center

        game.cursor.current_center = game.cursor.current_center + vector
        if game.cursor.get_range(game.env.target_map)[1,1] == 0:
               game.cursor.current_center = prev
        else:
            game.cursor.set_range(game.env.result_map, game.cursor.get_range(game.env.target_map))
        vis.update(100)


def create_model_params(action_factory, observation_factory):
    input_parameters = dict(
        source_feature_size =observation_factory.cursor.region_source_input.basic_block_size, # size of input window
        result_feature_size = np.prod(observation_factory.result_shape), # this is a space where we
    )
    output_parameters= dict(
        possible_moves = action_factory.movement_size, #where it can move
    )
    other_params = dict(dense_size=32, dropout_rate=0.2)
    model_params = dict(
        input_parameters=input_parameters,
        output_parameters = output_parameters,
        other_params = other_params,
    )
    return model_params

def prepare_game(data_path, config: TrainerConfig, network_type='empty'):

    data_generator = data.LartpcData.from_path(data_path)
    result_dimensions = 3
    env = Environment2D(result_dimensions=result_dimensions)
    env.set_map(*data_generator[3])
    game = Game2D(env, max_step_number=config.max_step_number)
    #vis = Visualisation(game)
    #vis.update()
    action_factory = Action2DFactory(game.cursor.copy(), categories=result_dimensions)
    observation_factory = Observation2DFactory(game.cursor.copy(), categories=result_dimensions)
    epsilon_kwrgs = dict(
        value=config.epsilon_initial_value,
        decay=config.epsilon_decay,
        min=config.epsilon_min
    )
    model_params = create_model_params(action_factory, observation_factory)
    network_builder = ParameterBasedNetworks(**model_params, action_factory=action_factory, observation_factory=observation_factory)
    network_factory = create_network_factory(network_type, network_builder, config)
    actor =  Actor(
        action_factory,
        observation_factory,
        epsilon_kwrgs=epsilon_kwrgs,
        network_model_factory=network_factory,
        batch_size= config.batch_size,
        trace_length= config.trace_length,
        gamma = config.gamma,
        categorisation_mode=config.categorisation_mode,
        decision_mode=config.decision_mode,
    )
    return game, actor, data_generator

def simple_learn(data_path):
    config = TrainerConfig()
    game, actor, data_generator = prepare_game(data_path, config, network_type=config.network_type)
    logger = Logger()
    mlf_logger = MLFlowLogger()
    mlf_logger.log_config(config)
    for iterate_maps in range(config.maps_iterations):
        map_number = np.random.randint(0, len(data_generator))
        game.env.set_map(*data_generator[map_number])
        iterations = []
        for iterate_tries in range(config.trials):
            game.start()
            trial_run_history = []
            for model_run_iteration in range(game.max_step_number):
                current_state = game.get_state()
                model_action = actor.create_action(current_state.obs)
                game_action = actor.action_factory.model_action_to_game(model_action)
                new_state = game.step(game_action)
                trial_run_history.append((current_state, game_action, new_state))
                if new_state.done:
                    break
            actor.memory.add(trial_run_history)
            iterations.append(trial_run_history.copy())
            if actor.enough_samples_to_learn():
                h = actor.replay()
                logger.add_train_history(h)
                mlf_logger.log_history(h)
                actor.target_train()
        logger.game_records(dict(map=map_number, data=iterations))
        mlf_logger.log_game(map_number, iterations)
        if actor.enough_samples_to_learn() and iterate_maps%4==0:
            logger.plot()
        actor.dump_models(Path('assets/model_dumps'))
        logger.dump_log()


if __name__ == "__main__":
    data_path = 'assets/dump'
    #data_path = '/home/mwm/repositories/content/dump'  # home cluster
    #ftest_draw_random_cursor(data_path)
    simple_learn(data_path)
