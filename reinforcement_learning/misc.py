import numpy as np
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