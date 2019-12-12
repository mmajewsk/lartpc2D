from common_config import LConfig
import data
from envs.game import Environment2D,Game2D
from actors.actions import Action2DFactory
from actors.observations import Observation2DFactory
from viz import Visualisation
from actors.models import BotActor
import numpy as np


def bot_replay(data_path):
    config = LConfig()
    config.max_step_number = 20
    data_generator = data.LartpcData(data_path)
    env = Environment2D()
    env.set_map(*data_generator[3])
    game = Game2D(env, max_step_number=config.max_step_number)
    vis = Visualisation(game)
    #vis.update()
    action_factory = Action2DFactory(game.cursor.copy())
    observation_factory = Observation2DFactory(game.cursor.copy())
    actor = BotActor(
        action_factory,
        observation_factory,
    )
    for iterate_maps in range(30):
        map_number = np.random.randint(0, len(data_generator))
        game.env.set_map(*data_generator[map_number])
        for iterate_tries in range(10):
            game.start()
            for model_run_iteration in range(game.max_step_number):
                current_state = game.get_observation()
                model_action = actor.create_action(current_state)
                game_action = actor.action_factory.model_action_to_game(model_action)
                new_state, reward, done, info = game.step(game_action)
                vis.update(0)
                if done:
                    break



if __name__ == "__main__":
    data_path = '/home/mwm/repositories/lartpc/lartpc2D-rl/dump'
    data_path = '/home/mwm/repositories/lartpc/lartpc_notebooks/Blog/content/dump'
    bot_replay(data_path)