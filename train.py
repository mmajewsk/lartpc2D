from envs.game import Environment2D, Game2D
from pathlib import  Path
from actors.observations import Observation2DFactory
import data
import matplotlib.pyplot as plt
import pickle
import numpy as np
from actors.models import Actor
from actors.actions import Action2DFactory
from envs.dims import neighborhood2d
from actors.networks import movement_network
#from viz import  Visualisation
import time
import datetime as dt
import git

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

import matplotlib.pyplot as plt
class Logger:
    def __init__(self):
        self.train_hist = []
        self.records = []
        repo = git.Repo(search_parent_directories=True)
        sha = repo.head.object.hexsha
        self.outputfilename = 'plots/{}_{}.png'.format(dt.datetime.now().strftime('%Y%m%d%H%M%S'), sha)

    def add_train_history(self, th):
        self.train_hist.append(th)

    def game_records(self,record):
        self.records.append(record)

    def plot(self):
        fig, axe = plt.subplots(3)
        axe[0].plot([h.history['loss'] for h in self.train_hist])
        axe[1].plot([h.history['acc'] for h in self.train_hist])
        axe[2].plot([h.history['mae'] for h in self.train_hist])
        fig.savefig(self.outputfilename)
        #plt.show()

    def dump_log(self):
        with open('loggerdump.pkl','wb') as f:
            pickle.dump(self, f)

def create_model_params(action_factory, observation_factory):
    input_parameters = dict(
        source_feature_size =observation_factory.cursor.region_source_input.basic_block_size, # size of input window
        result_feature_size = observation_factory.cursor.region_result_input.basic_block_size, # this is a space where we
                                   # log touched field (same cursor)
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

def simple_learn(data_path):
    data_generator = data.LartpcData(data_path)
    env = Environment2D()
    env.set_map(*data_generator[3])
    game = Game2D(env, max_step_number=4)
    #vis = Visualisation(game)
    #vis.update()
    action_factory = Action2DFactory(game.cursor.copy())
    observation_factory = Observation2DFactory(game.cursor.copy())
    epsilon_kwrgs = dict(value=1.0, decay=0.9987, min=0.5)
    model_params = create_model_params(action_factory, observation_factory)
    nm_factory = lambda : movement_network(**model_params)
    actor =  Actor(
        action_factory,
        observation_factory,
        epsilon_kwrgs=epsilon_kwrgs,
        network_model_factory=nm_factory
    )
    actor.batch_size = 32
    game.max_step_number = 8
    history = []
    p = Logger()
    for iterate_maps in range(4000):
        map_number = np.random.randint(0, len(data_generator))
        game.env.set_map(*data_generator[map_number])
        iterations = []
        for iterate_tries in range(8):
            game.start()
            trial_run_history = []
            for model_run_iteration in range(game.max_step_number):
                curent_state = game.get_observation()
                model_action = actor.create_action(curent_state)
                game_action = actor.action_factory.model_action_to_game(model_action)
                new_state, reward, done, info = game.step(game_action)
                trial_run_history.append((curent_state, game_action, reward, new_state, done))
                if done:
                    break
            actor.memory.add(trial_run_history)
            iterations.append(trial_run_history.copy())
            if actor.enough_samples_to_learn():
                h = actor.replay()
                p.add_train_history(h)
                actor.target_train()
        p.game_records(dict(map=map_number, data=iterations))
        if actor.enough_samples_to_learn() and iterate_maps%4==0:
            p.plot()
        actor.dump_models(Path('./model_dumps'))
        p.dump_log()


if __name__ == "__main__":
    data_path = '/home/mwm/repositories/lartpc/lartpc2D-rl/dump'
    data_path = '/home/mwm/repositories/lartpc/lartpc_notebooks/Blog/content/dump'
    #ftest_draw_random_cursor(data_path)
    simple_learn(data_path)
