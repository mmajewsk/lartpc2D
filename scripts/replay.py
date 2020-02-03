from pathlib import  Path
import numpy as np
from lartpc_game.viz import MixedModelVisualisation
from scripts.train import prepare_game
from common_configs import TrainerConfig

def simple_replay(data_path):
    config = TrainerConfig()
    game, actor, data_generator = prepare_game(data_path, config, network_type='empty')
    vis = MixedModelVisualisation(game)
    actor.load_models(Path('assets/model_dumps'))
    game.max_step_number =12
    for iterate_maps in range(50):
        map_number = np.random.randint(0, len(data_generator))
        game.env.set_map(*data_generator[map_number])
        for iterate_tries in range(10):
            game.start()
            for model_run_iteration in range(game.max_step_number):
                current_observation = game.get_observation()
                model_action = actor.create_action(current_observation, use_epsilon=False)
                game_action = actor.action_factory.model_action_to_game(model_action)
                state = game.step(game_action)
                vis.obs_action(current_observation, game_action)
                vis.update(0)
                if state.done:
                    break

if __name__ == "__main__":
    data_path = 'assets/dump'
    #data_path = '/home/mwm/repositories/lartpc/lartpc_notebooks/Blog/content/dump'
    simple_replay(data_path)