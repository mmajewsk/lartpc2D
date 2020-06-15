import pickle
import tempfile
import tensorflow as tf
import mlflow
import git
import datetime as dt
from matplotlib import pyplot as plt
import os
import binascii
from common_configs import ClassicConfConfig, TrainerConfig
import dataclasses
from pathlib import Path
import mlflow.keras

class Logger:
    def __init__(self):
        self.train_hist = []
        self.records = []
        try:
            repo = git.Repo(search_parent_directories=True)
            sha = repo.head.object.hexsha
        except git.exc.InvalidGitRepositoryError:
            sha = 'nosha'+str(binascii.hexlify(os.urandom(16)))
        self.outputfilename = 'assets/plots/{}_{}.png'.format(dt.datetime.now().strftime('%Y%m%d%H%M%S'), sha)

    def add_train_history(self, th):
        self.train_hist.append(th.history)

    def game_records(self,record):
        self.records.append(record)

    def plot(self):
        fig, axe = plt.subplots(3)
        axe[0].plot([h['output_movement_loss'] for h in self.train_hist])
        axe[1].plot([h['output_movement_acc'] for h in self.train_hist])
        axe[2].plot([h['output_movement_mae'] for h in self.train_hist])
        fig.savefig(self.outputfilename)
        #plt.show()

    def dump_log(self):
        with open('assets/loggerdump.pkl','wb') as f:
            pickle.dump(self, f)

class MLFlowLogger:
    def __init__(self, trainer_config):
        self.experiment = trainer_config.agent_type

    def start(self):
        #mlflow.set_tracking_uri('file:///home/mwm/repositories/lartpc_remote_pycharm')
        mlflow.set_experiment(self.experiment)
        mlflow.start_run()

    def log_config(self, config: TrainerConfig):
        mlflow.log_params(dataclasses.asdict(config))

    def log_history(self, hist: tf.keras.callbacks.History):
        h = hist.history.copy()
        new_h = {}
        for k,v in h.items():
            assert len(v) == 1
            new_h[k] = v[0]
        mlflow.log_metrics(new_h)

    def log_game(self, map, it):
        data = {'map': map, 'iterations': it}
        dirpath  = Path(tempfile.mkdtemp())
        pkl_path = dirpath/'game_log.pkl'
        with open(pkl_path,'wb') as f:
            pickle.dump(data, f)
        mlflow.log_artifact(pkl_path)
        os.remove(pkl_path)

    def log_model(self, actor):
        actor.log_mlflow(mlflow)

    def stop(self):
        mlflow.end_run()

