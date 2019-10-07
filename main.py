from envs.game import Environment2D, Game2D
import data
import matplotlib.pyplot as plt
import numpy as np
from envs.dims import neighborhood2d
from viz import  Visualisation
import time

def ftest_draw_random_cursor(data_path):
    data_generator = data.LartpcData(data_path)
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

def create_model():
    return None

def simple_learn(data_path):
    data_generator = data.LartpcData(data_path)
    env = Environment2D()
    env.set_map(*data_generator[3])
    game = Game2D(env)
    vis = Visualisation(game)
    vis.update()
    model = create_model()
    trial_run_history = []
    for iterate_maps in range(100):
        game.env.set_map(*data_generator.random())
        for iterate_tries in range(30):
            game.reset()
            for model_run_iteration in range(8):
                curent_state = game.get_observation()
                action = model.create_action(curent_state)
                new_state, reward, done, info = game.step(action)
                trial_run_history.append((curent_state, action, reward, new_state, done))


if __name__ == "__main__":
    data_path = '/home/mwm/repositories/lartpc/lartpc2D-rl/dump'
    data_path = '/home/mwm/repositories/lartpc/lartpc_notebooks/Blog/content/dump'
    ftest_draw_random_cursor(data_path)
