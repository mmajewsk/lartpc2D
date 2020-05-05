from abc import ABC, abstractmethod

import tensorflow as tf
from keras.engine.saving import load_model


class BaseNetwork(ABC):
    def __init__(self, *args, **kwargs):
        self.model = None

    def load(self, path):
        print("Loading from {}".format(path))
        self.model = load_model(path, custom_objects={'tf': tf})

    def set_model(self, model):
        self.model = model

    def set_output(self, output):
        self.output = output

    def save(self, path, name):
        self.model.save(path/name)

    @abstractmethod
    def build(self, *args, **kwargs):
        pass

    @abstractmethod
    def compiled(self, *args, **kwargs):
        pass