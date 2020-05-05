from pathlib import  Path
import numpy as np
from lartpc_game.viz import MixedModelVisualisation
from scripts.train import prepare_game
from common_configs import ReplayConfig, ClassicConfConfig
from reinforcement_learning.networks import CombinedNetwork

def simple_replay(data_path):
    config = ReplayConfig()
    game, actor, data_generator = prepare_game(data_path, config, classic_config=ClassicConfConfig(), network_type='empty')
    vis = MixedModelVisualisation(game)
    runpath = Path("assets/model_dumps")
    target = runpath/"target_model.h5"
    model = runpath/"model.h5"
    net = CombinedNetwork()
    net.load(model)
    #net.compiled()
    actor.model = net
    game.max_step_number =12
    for iterate_maps in range(50):
        map_number = np.random.randint(0, len(data_generator))
        game.env.set_map(*data_generator[map_number])
        for iterate_tries in range(10):
            game.start()
            for model_run_iteration in range(game.max_step_number):
                current_observation = game.get_observation()
                model_action = actor.create_action(current_observation, use_epsilon=False)
                game_action = model_action.to_game_aciton(actor.action_factory)
                state = game.step(game_action)
                vis.obs_action(current_observation, game_action)
                vis.update(0)
                if state.done:
                    break

if __name__ == "__main__":
    data_path = 'assets/dump'
    #data_path = '/home/mwm/repositories/lartpc/lartpc_notebooks/Blog/content/dump'
    simple_replay(data_path)