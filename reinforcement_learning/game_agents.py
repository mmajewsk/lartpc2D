import numpy as np
import random


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


class BaseMemoryAgent:
    def __init__(self):
        self.memory = BaseMemoryBuffer()

    def create_action(self, state):
        pass


class NonRepeatingSimpleBuffer(BaseMemoryBuffer):
    def __init__(self, buffer_size = 32):
        self.buffer = []
        self.buffer_size = buffer_size

    def add(self, experience : list):
        if len(experience) != self.buffer_size:
            raise ValueError("Experience must fit buffer! \n Got experience of length: {}, expected: {}.".format(len(experience), self.buffer_size))
        self.buffer = experience

    def sample(self, batch_size: int, trace_length:int):
        return self.buffer

class ExperienceBuffer(BaseMemoryBuffer):
    def __init__(self, buffer_size = 1000):
        self.buffer = []
        self.buffer_size = buffer_size

    def add(self, experience : list):
        """

        :param experience: In case of DDQN it was tst of tuples of [(obs, actions, obs),...,(..)]
        :return:
        """
        # if the buffer overflows over the siz,
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
        # just because in our usecase trace length is 1 he samples are for (batch_size, trace_length, 3)
        # where 3  stands for [GameVisibleState2D, EnvAction2D, GameVisibleState2D]
        return samples.reshape([samples.shape[0]*samples.shape[1], samples.shape[2]])

