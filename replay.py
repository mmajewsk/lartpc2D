from envs.game import Environment2D, Game2D
from pathlib import  Path
from actors.observations import Observation2DFactory
import data
import pickle
import numpy as np
from actors.models import Actor
from actors.actions import Action2DFactory
from envs.dims import neighborhood2d
from viz import  Visualisation
import time

def simple_replay(data_path):
    data_generator = data.LartpcData(data_path)
    env = Environment2D()
    env.set_map(*data_generator[3])
    game = Game2D(env, max_step_number=8)
    vis = Visualisation(game)
    action_factory = Action2DFactory(game.cursor.copy())
    observation_factory = Observation2DFactory(game.cursor.copy())
    epsilon_kwrgs = dict(value=1.0, decay=0.996, min=0.05)
    dummy_factory = lambda : None
    actor =  Actor(
        action_factory,
        observation_factory,
        epsilon_kwrgs=epsilon_kwrgs,
        network_model_factory=dummy_factory,
        batch_size=128,
        trace_length=1,
        gamma=0.8
    )
    actor.load_models(Path('./model_dumps'))
    actor.batch_size = 32
    game.max_step_number =12
    for iterate_maps in range(50):
        map_number = np.random.randint(0, len(data_generator))
        game.env.set_map(*data_generator[map_number])
        for iterate_tries in range(10):
            game.start()
            for model_run_iteration in range(game.max_step_number):
                curent_state = game.get_observation()
                model_action = actor.create_action(curent_state, use_epsilon=False)
                game_action = actor.action_factory.model_action_to_game(model_action)
                new_state, reward, done, info = game.step(game_action)
                vis.update(0)
                if done:
                    break

if __name__ == "__main__":
    data_path = '/home/mwm/repositories/lartpc/lartpc2D-rl/assets/dump'
    #data_path = '/home/mwm/repositories/lartpc/lartpc_notebooks/Blog/content/dump'
    simple_replay(data_path)