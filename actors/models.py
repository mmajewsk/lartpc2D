import numpy as np
from pathlib import  Path
import random
from actors.actions import Action2DFactory, GameAction2D, ModelAction2D
from actors.networks import movement_network, load_model
from actors.observations import ModelObservation2D, GameObservation2D, Observation2DFactory


class BaseMemoryBuffer:
    def add(self, experience):
        """

        :param experience:
        looks like this:
        experience = [
            (cur_state, action, reward, new_state, done),
            (cur_state, action, reward, new_state, done),
            ...
        ]
        :return:
        """
        pass

    def sample(self, batch_size: int, trace_length:int):
        pass

class ExperienceBuffer(BaseMemoryBuffer):
    def __init__(self, buffer_size = 1000):
        self.buffer = []
        self.buffer_size = buffer_size

    def add(self,experience):
        # if the buffer overflows over the size,
        # delete old from the beginning
        buffer_overflow = (len(self.buffer) + 1 >= self.buffer_size)
        if buffer_overflow:
            old_to_overwrite = (1+len(self.buffer))-self.buffer_size
            self.buffer[0:old_to_overwrite] = []
        self.buffer.append(experience)

    def sample(self,batch_size: int, trace_length: int) -> np.ndarray:
        """
        samples buffer wise:
        self.buffer = [ experience, experience, experience, ...]
                           True,       False,      True
        and picks list of size of batch_size

        then picks the episode trace

        like this, if trace_length == 3
        experience = [
            step1, # False
            step2, # False
            step3, # True
            step4, # True
            step5, # True
            step6  # False
            ...,   # False
            stepn  # False
        ]

        so in the end
        result : np.ndarray = [
            [step3, step4, step5],
            ....,
            [stepX, stepX+1, stepX+2]
        ]
        ande len(result) == batch_size
        """
        sampled_episodes = random.sample(self.buffer,batch_size)
        sampledTraces = []
        for episode in sampled_episodes:
            point = np.random.randint(0,len(episode)+1-trace_length)
            sampledTraces.append(episode[point:point+trace_length])
        sampledTraces = np.array(sampledTraces)
        return sampledTraces

class SquashedTraceBuffer(ExperienceBuffer):
    def sample(self,batch_size: int, trace_length: int) -> np.ndarray:
        samples = ExperienceBuffer.sample(self, batch_size, trace_length)
        return samples.reshape([samples.shape[0]*samples.shape[1], samples.shape[2]])

class BaseActor:
    def __init__(self):
        self.memory =BaseMemoryBuffer()

    def create_action(self, state):
        pass

class Epsilon:
    def __init__(self, value=1.0, decay=0.995, min=0.01):
        # epsilon is a decaying value used for exploration
        # the lower it is, the lower the chance of
        # actor generating random action
        self.value = value
        self.decay = decay
        self.min = min

    def condition(self):
        self.value *= self.decay
        self.value = max(self.min, self.value)
        return np.random.random() < self.value


class Actor(BaseActor):
    def __init__(
            self,
            action_factory: Action2DFactory,
            observation_factory: Observation2DFactory,
            epsilon_kwrgs: dict,
            network_model_factory,
        ):
        BaseActor.__init__(self,)
        self.action_factory = action_factory
        self.observation_factory = observation_factory
        self.memory = SquashedTraceBuffer()
        self.batch_size = 16
        self.trace_length = 1
        self.epsilon = Epsilon(**epsilon_kwrgs)
        self.model = network_model_factory()
        self.target_model = network_model_factory()
        self.gamma = 0.5
        self.tau = .225

    def load_models(self, path: Path):
        self.model = load_model(path/'model.h5')
        self.target_model = load_model(path/'target_model.h5')


    def create_action(self, state: GameObservation2D, use_epsilon=True) -> ModelAction2D:
        if use_epsilon:
            if self.epsilon.condition():
                return self.action_factory.create_random_action()
        model_obs = self.observation_factory.game_to_model_observation(state)
        return self.model_action(model_obs)

    def model_action(self, state: ModelObservation2D):
        model_response = self.model.predict(self.observation_factory.to_network_input(state))
        action = self.action_factory.movement_only(model_response[0])
        return action

    def enough_samples_to_learn(self):
        return len(self.memory.buffer) >self.batch_size

    def target_train(self):
        weights = self.model.get_weights()
        target_weights = self.target_model.get_weights()
        for i in range(len(target_weights)):
            target_weights[i] = weights[i] * self.tau + target_weights[i] * (1 - self.tau)
        self.target_model.set_weights(target_weights)

    def add_future_to_samples(self, samples):
        batched_samples = [[], []]
        batched_targets = []
        # effectively len(samples) == self.batches
        assert len(samples) ==self.batch_size
        for sample in samples:
            state, action, reward, new_state, done = sample
            state = self.observation_factory.game_to_model_observation(state)
            target_arr = self.target_model.predict(self.observation_factory.to_network_input(state))
            target_arr = target_arr.squeeze()
            if done:
                target_arr[action.movement_number] = reward
            else:
                future = self.target_model.predict(self.observation_factory.to_network_input(state)).squeeze()
                Q_future = max(future)
                target_arr[action.movement_number] = reward + Q_future * self.gamma

            target = np.expand_dims(np.expand_dims(target_arr, axis=0), axis=0)
            batched_targets.append(target)
            batched_samples[0].append(state.source_data)
            batched_samples[1].append(state.result_data)
        y = np.array(batched_targets)
        a,b = map(np.array, batched_samples)
        y = y.squeeze(1)
        return [a,b],y

    def replay(self):
        """
        Based on:
        https://towardsdatascience.com/reinforcement-learning-w-keras-openai-dqns-1eed3a5338c

        Also, the dimensionalities for fit;

        [ 
            [batch_size, trace_length, feature_size1],
            [batch_size, trace_length, feature_size2],
            ...,
            [batch_size, trace_length, feature_sizeX]
        ]

        :return:
        """
        samples = self.memory.sample(self.batch_size, self.trace_length)
        batched_samples, batched_targets = self.add_future_to_samples(samples)
        #assert batched_samples.shape[:-1] == (2, self.batch_size)
        return self.model.fit(batched_samples, batched_targets, epochs=1)

    def dump_models(self, path: Path):
        self.model.save(path/'model.h5')
        self.target_model.save(path/'target_model.h5')
