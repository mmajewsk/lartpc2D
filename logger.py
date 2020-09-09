import pickle
import tempfile
# import tensorflow as tf
import mlflow
import git
import datetime as dt
from matplotlib import pyplot as plt
import os
import binascii
from common_configs import ClassicConfConfig, TrainerConfig
import pytorch_lightning as pl
import dataclasses
from pathlib import Path
# import mlflow.keras
import neptune
import os
import torch

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
        self.train_hist.append(th)

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
    package = mlflow
    def __init__(self, params):
        exp_host = os.environ.get('EXPERIMENT_HOST')
        self.host = os.uname()[1] if exp_host is None else exp_host
        self.experiment = "{}@{}".format(params['agent_type'],self.host)

    def start(self):
        #mlflow.set_tracking_uri('file:///home/mwm/repositories/lartpc_remote_pycharm')
        self.package.set_experiment(self.experiment)
        self.package.start_run()

    def log_config(self, config: TrainerConfig):
        self.package.log_params(dataclasses.asdict(config))

    def log_history(self, hist):
        pass

    def log_game(self, map, it):
        data = {'map': map, 'iterations': it}
        dirpath  = Path(tempfile.mkdtemp())
        pkl_path = dirpath/'game_log.pkl'
        with open(pkl_path,'wb') as f:
            pickle.dump(data, f)
        self.package.log_artifact(pkl_path)
        os.remove(pkl_path)

    def log_model(self, actor):
        pass

    def stop(self):
        self.package.end_run()

class NeptuneLogger(MLFlowLogger):
    package = neptune

    def __init__(self, params):
        MLFlowLogger.__init__(self,params)
        self.params = params
        self.package.init(
            api_token=os.environ['NEPTUNE_API_TOKEN'],
            project_qualified_name='mmajewsk/lartpc'
        )


    def start(self):
        self.package.create_experiment(
            self.experiment,
            params=self.params
        )
        self.package.append_tag(self.host)

    def log_metrics(self, name, values):
        for v in values:
            self.package.log_metric(name,v)

    def log_history(self, hist):
        # @TODO this could be more efficient
        for k,v in hist.items():
            self.package.log_metric(k,v)

    def log_model(self, model):
        with tempfile.NamedTemporaryFile() as f:
            torch.save(model, f.name)
            self.package.log_artifact(f.name)


# class MLFlowLoggerTF:
#     def log_history(self, hist: tf.keras.callbacks.History):
#         h = hist.history.copy()
#         new_h = {}
#         for k,v in h.items():
#             assert len(v) == 1
#             new_h[k] = v[0]
#         mlflow.log_metrics(new_h)

#     def log_model(self, actor):
#         actor.log_mlflow(mlflow)


class MLFlowLoggerTorch(MLFlowLogger):
    def log_history(self, hist):
        self.package.log_metrics(hist)

    def log_model(self, actor):
        self.package.pytorch.log_model(actor.policy, "policy")
        self.package.pytorch.log_model(actor.target, "target")


def get_neptune_logger( config, classic_conf, device_str):
    exp_host = os.environ.get('EXPERIMENT_HOST')
    host = os.uname()[1] if exp_host is None else exp_host
    experiment = "{}@{}".format(config.agent_type,host)
    return NPLogger(
        api_key=os.environ['NEPTUNE_API_TOKEN'],
        project_name='mmajewsk/lartpc',
        experiment_name = experiment,
        params={**dataclasses.asdict(config), **dataclasses.asdict(classic_conf)},
        tags=[device_str, host]
    )

class NPLogger(pl.loggers.neptune.NeptuneLogger):
    def log_model(self, model):
        with tempfile.NamedTemporaryFile() as f:
            torch.save(model, f.name)
            self.log_artifact(f.name)

    def log_metric_arr(self, name, arr):
        for v in arr:
            self.log_metric(name, v)
