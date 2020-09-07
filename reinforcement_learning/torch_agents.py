from reinforcement_learning.common import RLAgent
from lartpc_game.game.game_ai import Lartpc2D
from lartpc_game.agents.agents import SquashedTraceBuffer, ExperienceBuffer
from lartpc_game.agents.observables import Action2Dai, State2Dai, Observation2Dai
from common_configs import TrainerConfig

import numpy as np
from reinforcement_learning.misc import Epsilon

import torch

class ToNumpy:
    def __call__(self, output):
        # @TODO it might speed things up if you onlu use torch tensor
        return tuple( i.numpy() for i in output)

class ModelOutputToAction:
    def __call__(self, output, action_settings):
        mov, put = output
        simple_index = self._get_simple_index(mov)
        unflat_data = put.reshape(action_settings.put_shape)
        new_mov = action_settings.possible_movement[simple_index]
        new_mov = new_mov[np.newaxis,:]
        return Action2Dai(new_mov, unflat_data)

    def _get_simple_index(self, mov):
        return np.argmax(mov)

class StateToObservables:
    def __call__(self, obs:Observation2Dai):
        ob_src = obs.source
        ob_res = obs.result
        return ob_src, ob_res

class ToFlat1D:
    def __call__(self, inp):
        return tuple(i.flatten()[np.newaxis,:] for i in inp)

class ToTorchTensorTuple:
    def __call__(self, inp):
        return tuple(torch.from_numpy(i) for i in inp)

class ToDevice:
    def __call__(self, inp, device):
        return tuple(i.to(device) for i in inp)

class GeneralAgent(RLAgent):
    def __init__(
            self,
            env: Lartpc2D,
    ):
        config = TrainerConfig()
        batch_size= config.batch_size
        trace_length= config.trace_length
        gamma = config.gamma
        RLAgent.__init__(
            self,
            env,
            batch_size,
            trace_length,
            memory = SquashedTraceBuffer(buffer_size=4000),
        )
        epsilon_kwrgs = dict(
            value=config.epsilon_initial_value,
            decay=config.epsilon_decay,
            min=config.epsilon_min
        )
        self.epsilon = Epsilon(**epsilon_kwrgs)
        self.gamma = gamma
        self.tau = .225

    def create_random_action(self):
        movement_random = np.random.random(self.action_settings.movement_size).astype(np.float32)
        put_random = np.random.random(self.action_settings.put_shape).astype(np.float32)
        movement_random, put_random= ToTorchTensorTuple()((movement_random, put_random))
        return movement_random, put_random

    def target_train(self, target, policy):
        for target_param, local_param in zip(target.parameters(),
                                           policy.parameters()):
            new_val = self.tau*local_param.data + (1-self.tau)*target_param.data
            target_param.data.copy_(new_val)


class TorchAgent(GeneralAgent):


    def create_action(self, observation, use_epsilon=True) -> Action2Dai:
        if use_epsilon:
            if self.epsilon.condition():
                return self.create_random_action()
        return self.model_action(observation)

    def enough_samples_to_learn(self):
        return len(self.memory.buffer) >self.batch_size

    def model_action(self, model_obs):
        self.policy.eval()
        src, canv = model_obs
        with torch.no_grad():
            return self.policy(src,canv)
        self.policy.train()


    def set_models(self, target, policy):
        self.target = target
        self.policy = policy

    def target_train(self):
        GeneralAgent.target_train(self, self.target, self.policy)


    def iterate_samples_nicely(self, samples):
        assert len(samples) ==self.batch_size

        for sample in samples:
            assert len(sample) == 3
            assert isinstance(sample[0], State2Dai)
            assert isinstance(sample[1], Action2Dai)
            assert isinstance(sample[2], State2Dai)
            for i, data in enumerate(sample):

                # print(i)
                # data.type_check()
                pass

            yield sample

    def add_future_to_samples(self, samples):
        obs_to_input = lambda x: (x.source,  x.result)
        batched_samples = [[], []]
        batched_targets = [[],[]]
        # @TODO make this batchable
        for current_state, action, new_state in self.iterate_samples_nicely(samples):

            observation = current_state.obs
            # observation = self.observation_settings.game_to_model_observation()
            # observation_to_predict = self.observation_settings.to_network_input(observation)
            src, canv = StateToObservables()(current_state.obs)
            src, canv = ToFlat1D()((src, canv))
            src, canv = ToTorchTensorTuple()((src,canv))

            # print(movement_target, category_target)
            if new_state.done:
                best_movement = np.argmax(action.movement_vector)
                target_val = torch.tensor(new_state.reward)
            else:
                new_obs = StateToObservables()(new_state.obs)
                new_obs = ToFlat1D()(new_obs)
                new_obs = ToTorchTensorTuple()(new_obs)
                self.target.eval()
                self.policy.eval()
                with torch.no_grad():
                    pol_mov, pol_cat = self.policy(*new_obs)
                    tar_mov, tar_cat  = self.target(*new_obs)
                    future_pol = pol_mov.detach()
                    future_val = tar_mov.detach()
                self.target.train()
                self.policy.train()
                best_policy = torch.argmax(future_pol)
                Q_val = tar_mov[0,best_policy]
                target_val = new_state.reward + Q_val * self.gamma
                # print(Q_val, best_policy, target_val)
            batched_targets[0].append(target_val)
            batched_targets[1].append(torch.from_numpy(current_state.obs.target))
            batched_samples[0].append(src)
            batched_samples[1].append(canv)
        y_mov = torch.tensor(batched_targets[0]).unsqueeze(1)
        y_cat = torch.cat(batched_targets[1])
        a,b = map(torch.cat, batched_samples)
        # print("="*9)
        # print(a.size())
        # print(b.size())
        # print(y_mov.size())
        # print(y_cat.size())
        return [a,b],[y_mov, y_cat]

    def train_agent(self):
        samples = self.memory.sample(self.batch_size, self.trace_length)
        batched_samples, batched_targets = self.add_future_to_samples(samples)
        return self.policy.optimise(batched_samples, batched_targets)
