from game.game import Environment2D, Game2D
from pathlib import Path
from actors.observations import Observation2DFactory
import data
import numpy as np
from models import Actor, ActorFactory
from actors.actions import Action2DFactory
from networks import NetworkFactory
#from viz import  Visualisation
from common_configs import TrainerConfig, ClassicConfConfig
from logger import Logger, MLFlowLogger


def prepare_game(data_path, config: TrainerConfig, classic_config, network_type='empty'):
    data_generator = data.LartpcData.from_path(data_path)
    result_dimensions = 3
    env = Environment2D(result_dimensions=result_dimensions)
    env.set_map(*data_generator[3])
    game = Game2D(env, max_step_number=config.max_step_number)
    action_factory = Action2DFactory(game.cursor.copy(), categories=result_dimensions)
    observation_factory = Observation2DFactory(game.cursor.copy(), categories=result_dimensions)
    actor_factory = ActorFactory(
        action_factory, observation_factory, config, classic_config
    )
    actor = actor_factory.produce_actor(network_type)
    return game, actor, data_generator

def simple_learn(data_path):
    config = TrainerConfig()
    classic_config = ClassicConfConfig()
    game, actor, data_generator = prepare_game(data_path, config, network_type=config.network_type, classic_config=classic_config)
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
        mlf_logger.log_model(actor)


if __name__ == "__main__":
    data_path = 'assets/dump'
    #data_path = '/home/mwm/repositories/content/dump'  # home cluster
    #ftest_draw_random_cursor(data_path)
    simple_learn(data_path)
