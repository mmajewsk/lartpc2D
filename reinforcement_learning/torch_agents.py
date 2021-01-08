import collections
import random
import numpy as np
import torch
import torchvision

from lartpc_game.game.game_ai import Lartpc2D
from lartpc_game.agents.observables import Action2Dai, State2Dai, Observation2Dai


from reinforcement_learning.misc import Epsilon
from reinforcement_learning.game_agents import BaseMemoryBuffer
from reinforcement_learning.common import RLAgent
from common_configs import TrainerConfig

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

class ToFlat1DTorch:
    def __call__(self, inp):
        return tuple(i.flatten(start_dim=1) for i in inp)

class ToTorchTensorTuple:
    def __call__(self, inp):
        return tuple(torch.from_numpy(i) for i in inp)

class ToDevice:
    def __init__(self, device):
        self.device = device

    def __call__(self, inp):
        return tuple(i.to(self.device) for i in inp)

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
            memory = NoTraceBuffer(buffer_size=4000),
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
        movement_random = np.random.uniform(-100, 100, self.action_settings.movement_size).astype(np.float32)[np.newaxis,:]
        put_random = np.random.random(self.action_settings.put_shape).astype(np.float32)
        movement_random, put_random= ToTorchTensorTuple()((movement_random, put_random))
        return movement_random, put_random

    def target_train(self, target, policy):
        for target_param, local_param in zip(target.parameters(),
                                           policy.parameters()):
            new_val = self.tau*local_param.data + (1-self.tau)*target_param.data
            target_param.data.copy_(new_val)

class NoTraceBuffer(BaseMemoryBuffer):
    def __init__(self, buffer_size=1000):
        self.buffer = collections.deque(maxlen=buffer_size)

    def add(self, cur: State2Dai, act: Action2Dai, new: State2Dai):
        self.buffer.append(
            (
                act[0],
                cur.obs.source,
                cur.obs.result,
                cur.obs.target,
                new.obs.source,
                new.obs.result,
                new.done,
                new.reward
            )
        )       

    def sample(self, batch_size:int):
        sampled_episodes = random.sample(self.buffer,batch_size)
        return sampled_episodes


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
            result = self.policy(src,canv)
        self.policy.train()
        return result


    def set_models(self, target, policy):
        self.target = target
        self.policy = policy

    def target_train(self):
        GeneralAgent.target_train(self, self.target, self.policy)


    def train_agent(self, device):
        samples = self.memory.sample(self.batch_size)
        npsamples = list(map(list,zip(*samples)))
        transform_to_input = torchvision.transforms.Compose([
            ToFlat1DTorch(),
            ToDevice(device),
        ])

        # import pdb; pdb.set_trace()
        act_movement_vector = np.stack(npsamples[0]).astype(np.float32)
        cur_obs_source = np.stack(npsamples[1]).astype(np.float32)
        cur_obs_canvas = np.stack(npsamples[2]).astype(np.float32)
        cur_obs_target = np.stack(npsamples[3]).astype(np.int32)
        new_obs_source = np.stack(npsamples[4]).astype(np.float32)
        new_obs_canvas = np.stack(npsamples[5]).astype(np.float32)
        new_done = np.stack(npsamples[6])[:,np.newaxis]
        new_reward = np.stack(npsamples[7]).astype(np.float32)[:,np.newaxis]
        # import pdb; pdb.set_trace()

        sample_t = (act_movement_vector,
        cur_obs_source,
        cur_obs_canvas,
        cur_obs_target,
        new_obs_source,
        new_obs_canvas,
        new_done,
        new_reward)

        (act_movement_vector,
        cur_obs_source,
        cur_obs_canvas,
        cur_obs_target,
        new_obs_source,
        new_obs_canvas,
        new_done,
        new_reward) = ToTorchTensorTuple()(sample_t)

        # print(act_movement_vector.size() )
        # print(cur_obs_source.size() )
        # print(cur_obs_canvas.size() )
        # print(cur_obs_target.size() )
        # print(new_obs_source.size() )
        # print(new_obs_canvas.size() )
        # print(new_done.size() )
        # print(new_reward.size() )

        # print(movement_target, category_target)
        # if new_state.done:
        #     best_movement = torch.argmax(action_movement_vector)
        #     target_val = torch.tensor(new_state.reward)
        # else:

        state_input = transform_to_input((cur_obs_source, cur_obs_canvas))
        new_state_input =  transform_to_input((new_obs_source, new_obs_canvas))
        pol_mov, pol_cat = self.policy(*state_input)
        self.target.eval()
        with torch.no_grad():
            tar_mov, tar_cat  = self.target(*new_state_input)
            Q_val = tar_mov.max(1)[0].unsqueeze(1)
            Q_val[new_done.squeeze()] = 0.0
            Q_val = Q_val.detach()
        target_val = new_reward+ Q_val * self.gamma
        # batched_targets[0].append(target_val)
        # batched_targets[1].append(torch.from_numpy(current_state.obs.target))
        # batched_samples[0].append(src)
        # batched_samples[1].append(canv)
        # y_mov = torch.tensor(batched_targets[0]).unsqueeze(1)
        # y_cat = torch.cat(batched_targets[1])
        # a,b = map(torch.cat, batched_samples)
        # a,b, y_mov, y_cat = ToDevice(device)((a,b, y_mov, y_cat))
        # print("="*9)
        # print(a.size())
        # print(b.size())
        # print(y_mov.size())
        # print(y_cat.size())
        net_output, batched_targets = [pol_mov, pol_cat],[target_val, cur_obs_target]
        return self.policy.optimise(net_output, batched_targets)
