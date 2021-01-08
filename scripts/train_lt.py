from typing import Tuple, List
import dataclasses
import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader
from torch.utils.data.dataset import IterableDataset

import pytorch_lightning as pl
from reinforcement_learning.torch_agents import GeneralAgent, ModelOutputToAction, \
    StateToObservables, ToTorchTensorTuple, ToNumpy, ToFlat1D, ToDevice

import lartpc_game
import lartpc_game.data
from lartpc_game.game.game_ai import Lartpc2D
from lartpc_game.agents.observables import Action2Dai, State2Dai, Observation2Dai
from collections import OrderedDict

from common_configs import TrainerConfig, ClassicConfConfig
from scripts.train_torch import create_model_params
from reinforcement_learning.torch_networks import MovementTorch, CombinedNetworkTorch
from scripts.classic_conv_lt import CatLt
from reinforcement_learning.game_agents import SquashedTraceBuffer

import copy
from tqdm import trange
# inspired by https://github.com/PyTorchLightning/pytorch-lightning/blob/master/pl_examples/domain_templates/reinforce_learn_Qnet.py

class RLDataset(IterableDataset):
    """
    Iterable Dataset containing the ExperienceBuffer
    which will be updated with new experiences during training

    Args:
        buffer: replay buffer
        sample_size: number of experiences to sample at a time
    """


    def __init__(self, buffer, sample_size: int = 200) -> None:
        self.buffer = buffer
        self.sample_size = sample_size

    def __iter__(self) -> Tuple:
        sample = self.buffer.sample(self.sample_size, trace_length=1)
        current_state, game_action, new_state = sample.T
        # @TODO can we batch this ?
        for i in range(self.sample_size):
            cur_state = dataclasses.astuple(current_state[i].obs)
            mov_vec = game_action[i].movement_vector
            # import ipdb; ipdb.set_trace()
            new_state_ = new_state[i].reward, new_state[i].done, dataclasses.astuple(new_state[i].obs)
            yield cur_state, mov_vec, new_state_

class AgentPL(GeneralAgent):
    def __init__(self, env, data_generator):
        GeneralAgent.__init__(self, env)
        self.data_generator = data_generator

    @staticmethod
    @torch.no_grad()
    def model_action(net, model_obs):
        net.eval()
        src, canv = model_obs
        result  = net(src,canv)
        net.train()
        return result

    def create_random_action(self):
        movement_random = np.random.random(self.action_settings.movement_size).astype(np.float32)
        put_random = np.random.random(self.action_settings.put_shape).astype(np.float32)
        movement_random, put_random= ToTorchTensorTuple()((movement_random, put_random))
        return movement_random, put_random


    def create_action(self, net, observation, use_epsilon=True) -> Action2Dai:
        if use_epsilon:
            if self.epsilon.condition():
                return self.create_random_action()
        # __import__("pdb").set_trace()

        return AgentPL.model_action(net, observation)


    def do_map(self, net,  trials, max_step_number, device=None):
        # one call will populate the trial_records with:
        # (trials * max_step_number, 3) data where 3 stand
        # for state, action, new_state
        trials_record = []
        map_number = np.random.randint(1000, len(self.data_generator))
        self.env.detector.set_maps(*self.data_generator[map_number])
        for trial in range(trials):
            trial = []
            self.env.start()
            for i in range(max_step_number):
                current_state = self.env.get_state()
                model_state = StateToObservables()(current_state.obs)
                model_state = ToFlat1D()(model_state)
                model_state_tensor = ToTorchTensorTuple()(model_state)
                model_state_tensor = ToDevice()(model_state_tensor, device)
                model_action = self.create_action(net, model_state_tensor)
                # @TODO loose to numpy
                model_action = ToNumpy()(model_action)
                game_action = ModelOutputToAction()(model_action, self.action_settings)
                new_state = self.env.step(game_action)
                trial.append((current_state, game_action, new_state))
                if new_state.done:
                    break
            trials_record.append(trial)
        return trials_record



class DQNLightning(pl.LightningModule):

    def __init__(self, data_path):
        super().__init__()
        self.config = TrainerConfig()
        self.max_step_number = self.config.max_step_number
        self.trials_number = self.config.trials
        classic_config = ClassicConfConfig()
        self.data_generator = lartpc_game.data.LartpcData.from_path(data_path)
        self.result_dimensions = 3
        self.batch_size = self.config.batch_size
        # @TODO env.set_maps(*data_generator[3])
        self.env = Lartpc2D(self.result_dimensions , max_step_number=self.config.max_step_number)
        model_settings = create_model_params(self.env)
        self.policy, self.target = DQNLightning.make_nets(model_settings, classic_config, self.result_dimensions)
        self.buffer = SquashedTraceBuffer(buffer_size=4000)
        self.agent = AgentPL(self.env, self.data_generator)
        self.sync_rate = 5
        self.populate()

    @staticmethod
    def make_nets(model_settings, classic_config, result_dimensions):
        mov_net = MovementTorch(
            source_in_size=model_settings['input_parameters']['source_feature_size'],
            result_in_size=model_settings['input_parameters']['result_feature_size'],
            moves_out_size=model_settings['output_parameters']['possible_moves'],
            dense_size = model_settings['other_params']['dense_size'],
            dropout_rate = model_settings['other_params']['dropout_rate']
        )
        cat_net = CatLt(
            dense_size=classic_config.dense_size,
            dropout_rate=classic_config.dropout_rate,
            result_feature_size=result_dimensions
        )
        policy = CombinedNetworkTorch(mov_net, cat_net)
        target = CombinedNetworkTorch(copy.deepcopy(mov_net), copy.deepcopy(cat_net))
        return policy, target

    @staticmethod
    def create_model_params(env: Lartpc2D):
        action_settings = env.action_settings
        observation_settings = env.observation_settings
        input_parameters = dict(
            source_feature_size =observation_settings.cursor.region_source_input.basic_block_size, # size of input window
            result_feature_size = np.prod(observation_settings.result_shape), # this is a space where we
        )
        output_parameters= dict(
            possible_moves = action_settings.movement_size, #where it can move
        )
        other_params = dict(dense_size=32, dropout_rate=0.2)
        model_params = dict(
            input_parameters=input_parameters,
            output_parameters = output_parameters,
            other_params = other_params,
        )
        return model_params

    def populate(self, steps: int = 2000) -> None:
        calls = steps // (self.config.trials*self.config.max_step_number)
        for i in trange(calls, desc='populate'):
            trials = self.agent.do_map(self.policy, self.trials_number, self.max_step_number)
            [self.buffer.add(trial) for trial in trials]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self.policy(x)
        return output

    def batch_train(self, batch) -> torch.Tensor:
        cur_state_obs, mov_vec, new_state = batch
        cur_state_src, cur_state_canv, cur_state_target = cur_state_obs
        new_state_reward, new_state_done, new_state_obs = new_state
        new_state_src, new_state_canv, _ = new_state_obs
        pol_mov, pol_cat = self.policy(cur_state_src.flatten(1), cur_state_canv.flatten(1))
        self.target.eval()
        with torch.no_grad():
            tar_mov, tar_cat  = self.target(new_state_src.flatten(1), new_state_canv.flatten(1))
            future_val = tar_mov.detach().max(1)[0]
        future_val[new_state_done] = 0.0
        target_val = new_state_reward + future_val * self.agent.gamma

        cat_labels_ = pl.metrics.functional.to_categorical(cur_state_target)
        mov_val = pol_mov.max(1).values.unsqueeze(1)
        mov_loss = F.mse_loss(pol_mov, target_val)
        cat_loss = F.cross_entropy(pol_cat, cat_labels_)
        metrics = self.make_metrics((pol_mov, po_cat), (target_val, cat_labels_))
        cat_mse, cat_acc = metrics
        # @TODO remove detach?
        hist = {
            'mov_loss': mov_loss.detach().to(device),
            'cat_loss': cat_loss.detach().to(device),
            'cat_mse': cat_mse.detach().to(device),
            'cat_acc': cat_acc.detach().to(device),
        }
        return (mov_loss+cat_loss), hist


    def training_step(self, batch, nb_batch) -> OrderedDict:
        # @TODO check if that below works
        device = self.get_device(batch)
        # @TODO keep eye on production vs consuption of data
        # do_map produces trials * steps experiences
        # batch_train consumes batch_size of experiences
        trials = self.agent.do_map(self.policy, self.trials_number, self.max_step_number, device)
        rewards = []
        for trial in trials:
            self.buffer.add(trial)
            rewards += [new_state.reward for _,_, new_state in trial]
        # import ipdb; ipdb.set_trace()
        loss, hist = self.batch_train(batch)
        if self.global_step % self.sync_rate == 0:
            self.target_train()

        log = {
               'reward': torch.tensor(rewards).to(device),
               'steps': torch.tensor(self.global_step).to(device)}
        log.update(hist)
        # return OrderedDict({'loss': loss, 'log': log, 'progress_bar': log})
        return OrderedDict({'loss': loss, 'log': log})


    def target_train(self):
        for target_param, local_param in zip(self.target.parameters(),
                                           self.policy.parameters()):
            new_val = self.tau*local_param.data + (1-self.tau)*target_param.data
            target_param.data.copy_(new_val)


    def configure_optimizers(self) -> List[Optimizer]:
        optimizer = optim.Adam(self.parameters(), lr=0.00001)
        return [optimizer]

    def __dataloader(self) -> DataLoader:
        dataset = RLDataset(self.buffer, self.batch_size)
        dataloader = DataLoader(
            dataset=dataset,
            batch_size=self.batch_size,
            sampler=None,
        )
        return dataloader

    def train_dataloader(self) -> DataLoader:
        return self.__dataloader()

    def get_device(self, batch) -> str:
        return batch[0].device.index if self.on_gpu else 'cpu'

if __name__ == '__main__':
    model = DQNLightning('../assets/dump')

    trainer = pl.Trainer(
        # gpus=1,
        distributed_backend='dp',
        early_stop_callback=False,
        val_check_interval=100
    )

    trainer.fit(model)
