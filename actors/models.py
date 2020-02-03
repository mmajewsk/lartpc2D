import numpy as np
from pathlib import  Path
from rl_environments.actors.actions import Action2DFactory, GameAction2D, ModelAction2D
from actors.networks import load_model
from rl_environments.actors.base_models import BaseActor, BaseMemoryActor, SquashedTraceBuffer
from rl_environments.actors.states import GameVisibleState2D
from rl_environments.actors.observations import GameObservation2D, Observation2DFactory


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


class Actor(BaseMemoryActor, BaseActor):
    def __init__(
            self,
            action_factory: Action2DFactory,
            observation_factory: Observation2DFactory,
            epsilon_kwrgs: dict,
            network_model_factory,
            batch_size: int,
            trace_length: int,
            gamma: float,
            decision_mode: str = 'network',
            categorisation_mode: str = 'network',
    ):
        BaseMemoryActor.__init__(self, )
        BaseActor.__init__(self, action_factory, observation_factory)
        self.memory = SquashedTraceBuffer(buffer_size=4000)
        self.batch_size = batch_size
        self.trace_length = trace_length
        self.epsilon = Epsilon(**epsilon_kwrgs)
        self.model = network_model_factory()
        self.target_model = network_model_factory()
        self.gamma = gamma
        self.tau = .225
        assert decision_mode in ['network', 'random']
        assert categorisation_mode in ['network', 'random']
        self.decision_mode = decision_mode
        self.categorisation_mode = categorisation_mode
        self._choose_action_creation()

    def load_models(self, path: Path):
        import tensorflow as tf
        self.model = load_model(path/'model.h5', custom_objects={'tf':tf})
        self.target_model = load_model(path/'target_model.h5', custom_objects={'tf':tf})


    def create_action(self, state: GameObservation2D, use_epsilon=True) -> ModelAction2D:
        if use_epsilon:
            if self.epsilon.condition():
                return self.action_factory.create_random_action()
        model_obs = self.observation_factory.game_to_model_observation(state)
        return self.model_action(model_obs)

    def _choose_action_creation(self):
        if self.categorisation_mode is 'random' and self.decision_mode is 'random':
            self.model_action = lambda x: self.action_factory.create_random_action()
        else:
            if self.categorisation_mode is 'network' and self.decision_mode is 'network':
                post_action = lambda x: x
            elif self.categorisation_mode is 'random':
                post_action = self.action_factory.randomise_category
            elif self.decision_mode is 'random':
                post_action = self.action_factory.randomise_movement
            def model_response(state) -> ModelAction2D:
                observation_to_predict = self.observation_factory.to_network_input(state)
                response = self.model.predict(observation_to_predict)
                action = ModelAction2D(*response)
                return post_action(action)
            self.model_action = model_response

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
        batched_targets = [[],[]]
        # effectively len(samples) == self.batches
        assert len(samples) ==self.batch_size
        for sample in samples:
            assert len(sample) == 3
            assert isinstance(sample[0], GameVisibleState2D)
            assert isinstance(sample[1], GameAction2D)
            assert isinstance(sample[2], GameVisibleState2D)
            current_state, action, new_state = sample
            #state, action, reward, new_state, done = sample
            observation = self.observation_factory.game_to_model_observation(current_state.obs)
            observation_to_predict = self.observation_factory.to_network_input(observation)
            movement_target, category_target = self.target_model.predict(observation_to_predict)
            movement_target = movement_target.squeeze()
            if new_state.done:
                movement_target[action.movement_number] = new_state.reward
            else:
                future_movement, predicted_classes = self.target_model.predict(self.observation_factory.to_network_input(observation))
                Q_future = max(future_movement.squeeze())
                movement_target[action.movement_number] = new_state.reward + Q_future * self.gamma
            movement_target_with_future = np.expand_dims(np.expand_dims(movement_target, axis=0), axis=0)
            current_model_state = self.state_factory.game_to_model_visible_state(current_state)
            batched_targets[0].append(movement_target_with_future)
            batched_targets[1].append(current_model_state.target)
            batched_samples[0].append(observation.source_data)
            batched_samples[1].append(observation.result_data)
        y_mov = np.array(batched_targets[0])
        y_cat = np.array(batched_targets[1])
        a,b = map(np.array, batched_samples)
        y_mov = y_mov.squeeze(1)
        y_cat = y_cat.squeeze(1)
        return [a,b],[y_mov, y_cat]

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
        #print(len(batched_samples), batched_samples[0].shape)
        return self.model.fit(batched_samples, batched_targets, epochs=1)

    def dump_models(self, path: Path):
        self.model.save(path/'model.h5')
        self.target_model.save(path/'target_model.h5')
