import numpy as np
from pathlib import  Path
from agents.actions import Action2DSettings, EnvAction2D, QAction2D, PolicyAction
from agents.base_agents import BaseAgent, BaseMemoryAgent, SquashedTraceBuffer, ExperienceBuffer
from agents.base_agents import NoRepeatExperienceBuffer
from agents.states import GameVisibleState2D
from agents.observations import EnvObservation2D, Observation2DSettings
from game.game import Lartpc2D
from reinforcement_learning.misc import Epsilon
from reinforcement_learning.networks import NetworkFactory
from reinforcement_learning.common import RLAgent


class DDQNAgent(RLAgent):
    def __init__(
            self,
            epsilon_kwrgs: dict,
            env: Lartpc2D,
            batch_size: int,
            trace_length: int,
            gamma: float,
            decision_mode: str = 'network',
            categorisation_mode: str = 'network',
    ):
        RLAgent.__init__(
            self,
            env,
            batch_size,
            trace_length,
            memory = SquashedTraceBuffer(buffer_size=4000),
        )
        self.epsilon = Epsilon(**epsilon_kwrgs)
        self.gamma = gamma
        self.tau = .225
        assert decision_mode in ['network', 'random']
        assert categorisation_mode in ['network', 'random']
        self.decision_mode = decision_mode
        self.categorisation_mode = categorisation_mode
        self._choose_action_creation()

    def set_models(self, target, policy):
        self.target = target
        self.polivy = policy

    def create_action(self, state: EnvObservation2D, use_epsilon=True) -> QAction2D:
        if use_epsilon:
            if self.epsilon.condition():
                return QAction2D.create_random_action(self.action_settings)
        model_obs = self.observation_settings.game_to_model_observation(state)
        return self.model_action(model_obs)

    def _choose_action_creation(self):
        if self.categorisation_mode is 'random' and self.decision_mode is 'random':
            self.model_action = lambda x: QAction2D.create_random_action(self.action_settings)
        else:
            if self.categorisation_mode is 'network' and self.decision_mode is 'network':
                post_action = lambda x: x
            elif self.categorisation_mode is 'random':
                post_action = lambda x: QAction2D.randomise_category(x, self.action_settings)
            elif self.decision_mode is 'random':
                post_action = lambda x: QAction2D.randomise_movement(x, self.action_settings)
            def model_response(state) -> QAction2D:
                observation_to_predict = self.observation_settings.to_network_input(state)
                response = self.model.model.predict(observation_to_predict)
                action = QAction2D(*response)
                return post_action(action)
            self.model_action = model_response

    def enough_samples_to_learn(self):
        return len(self.memory.buffer) >self.batch_size

    def target_train(self):
        weights = self.model.model.get_weights()
        target_weights = self.target_model.model.get_weights()
        for i in range(len(target_weights)):
            target_weights[i] = weights[i] * self.tau + target_weights[i] * (1 - self.tau)
        self.target_model.model.set_weights(target_weights)

    def add_future_to_samples(self, samples):
        batched_samples = [[], []]
        batched_targets = [[],[]]
        # effectively len(samples) == self.batches
        assert len(samples) ==self.batch_size
        for sample in samples:
            assert len(sample) == 3
            assert isinstance(sample[0], GameVisibleState2D)
            assert isinstance(sample[1], EnvAction2D)
            assert isinstance(sample[2], GameVisibleState2D)
            current_state, action, new_state = sample
            #state, action, reward, new_state, done = sample
            observation = self.observation_settings.game_to_model_observation(current_state.obs)
            observation_to_predict = self.observation_settings.to_network_input(observation)
            movement_target, category_target = self.target_model.model.predict(observation_to_predict)
            movement_target = movement_target.squeeze()
            if new_state.done:
                movement_target[action.movement_number] = new_state.reward
            else:
                future_movement, predicted_classes = self.target_model.model.predict(self.observation_settings.to_network_input(observation))
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

    def train_agent(self):
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
        return self.model.model.fit(batched_samples, batched_targets, epochs=1)

    def log_mlflow(self, package):
        package.keras.log_model(self.model.model, "models")
        package.keras.log_model(self.target_model.model, "target_models")

    def dump_models(self, path: Path):
        self.model.save(path, 'model.h5')
        self.target_model.save(path, 'target_model.h5')

class A2CAgent(RLAgent):
    def __init__(
            self,
            action_settings: Action2DSettings,
            observation_settings: Observation2DSettings,
            batch_size: int,
            trace_length: int,
            gamma: float = 0.99,
            actor_lr = 0.0001,
            critic_lr = 0.0001,
            model_actor=None,
            model_critic=None,
    ):
        RLAgent.__init__(
            self,
            action_settings,
            observation_settings,
            batch_size,
            trace_length,
            # below is so that there would be no repetition in samples
            memory=NoRepeatExperienceBuffer(buffer_size=batch_size)
        )
        self.gamma = gamma
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.model_actor = model_actor
        self.model_critic = model_critic
        self._dump_message = False


    def discount_rewards(self, rewards):
        discounted_rewards = np.zeros_like(rewards)
        running_add = 0
        for t in reversed(range(0, len(rewards))):
            if rewards[t] != 0:
                running_add = 0
            running_add = running_add * self.gamma + rewards[t]
            discounted_rewards[t] = running_add
        return discounted_rewards

    def train_agent(self):
        #assert False
        samples = self.memory.sample(self.batch_size, self.trace_length)
        assert len(samples) == self.batch_size
        # the multiplication here is because trace is only important when calculating discounts
        states_batch_1 = []
        states_batch_2 = []
        adv_batch = []
        disc_batch = []
        targets_batch = []
        _to_input = lambda x: self.observation_settings.to_network_input(self.observation_settings.game_to_model_observation(x))
        for sample_no, sample in enumerate(samples):
            # @TODO check if this is correct (if steps are in correct order)
            rewards = [ episode[2].reward for episode in sample]
            discounted_rewards = self.discount_rewards(rewards)
            discounted_rewards -= np.mean(discounted_rewards)
            if np.std(discounted_rewards):
                discounted_rewards /= np.std(discounted_rewards)
            else:
                # this could be covered differently
                continue
            disc_batch += discounted_rewards.tolist()
            states_1 = np.concatenate([_to_input(step[0].obs)[0] for step in sample])
            states_2 = np.concatenate([_to_input(step[0].obs)[1] for step in sample])
            actions = [step[1] for step in sample]
            values = self.model_critic.model.predict([states_1,states_2])
            #advantages = np.zeros((self.trace_length, self.action_settings.movement_size))
            for i in range(self.trace_length):
                policy_action = PolicyAction.from_game(actions[i], self.action_settings)
                # it coulde be advantage[i][self.actions[i]] = discounted_rewards[i]-values[i], but because policy is already an array,
                # with only one element equal to zero, we can do it like this:
                advantage = policy_action.policy * (discounted_rewards[i]-values[i])
                adv_batch.append(advantage)
            #advantages_batch[sample_no*self.trace_length:sample_no*self.trace_length+self.trace_length] = advantages
            #discounted_rewards_batch[sample_no * self.trace_length:sample_no * self.trace_length + self.trace_length] = discounted_rewards
            targets_batch += [self.state_factory.game_to_model_visible_state(step[0]).target for step in sample]
            states_batch_1.append(states_1)
            states_batch_2.append(states_2)
        states_batch_1 = np.concatenate(states_batch_1)
        states_batch_2 = np.concatenate(states_batch_2)
        states_batch = [states_batch_1, states_batch_2]
        targets_batch = np.concatenate(targets_batch)
        adv_batch = np.concatenate(adv_batch)[:,np.newaxis,:]
        disc_batch = np.array(disc_batch)[:,np.newaxis,np.newaxis]
        actor_loss = self.model_actor.model.fit(states_batch, [adv_batch, targets_batch], epochs=1)
        critic_loss = self.model_critic.model.fit(states_batch,disc_batch, epochs=1)
        return actor_loss, critic_loss

    def enough_samples_to_learn(self):
        if len(self.memory.buffer)>=self.batch_size:
            self.memory.trim_to_trace(self.trace_length)
        return len(self.memory.buffer) >= self.batch_size

    def create_action(self, state: EnvObservation2D) -> PolicyAction:
        model_obs = self.observation_settings.game_to_model_observation(state)
        observation_to_predict = self.observation_settings.to_network_input(model_obs)
        response = self.model_actor.model.predict(observation_to_predict)
        action = PolicyAction(*response)
        return action

    def target_train(self):
        pass

    def log_mlflow(self, package):
        package.keras.log_model(self.model_critic.model, "critic")
        package.keras.log_model(self.model_actor.model, "actor")

    def dump_models(self, path):
        if not self._dump_message:
            print("dumping models not implemented")
            self._dump_message = True
        pass

def create_model_params(action_settings, observation_settings):
    input_parameters = dict(
        source_feature_size =observation_factory.cursor.region_source_input.basic_block_size, # size of input window
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

class AgentFactory:
    def __init__(self, action_settings, observation_settings, config, classic_config):
        self.action_settings = action_settings
        self.observation_settings = observation_settings
        self.classic_config = classic_config


    def get_network_builder(self, model_params, config ):
        network_builder = NetworkFactory(
            **model_params,
            action_factory=self.action_settings,
            observation_factory=self.observation_settings,
            config=config,
            classic_config=self.classic_config
        )
        return network_builder

    def produce_ddqn(self, network_type, config) -> DDQNAgent:
        model_params = create_model_params(self.action_settings, self.observation_settings)
        network_builder = self.get_network_builder(model_params, config)
        model = network_builder.create_network(network_type)
        target_model = network_builder.create_network(network_type)

        return  DDQNAgent(
            self.action_settings,
            self.observation_settings,
            epsilon_kwrgs=epsilon_kwrgs,
            model=model,
            target_model=target_model,
            batch_size= config.batch_size,
            trace_length= config.trace_length,
            gamma = config.gamma,
            categorisation_mode=config.categorisation_mode,
            decision_mode=config.decision_mode,
        )


    def produce_a2c(self, config):
        model_params = create_model_params(self.action_settings, self.observation_settings)
        network_builder = self.get_network_builder(model_params, config)
        actor, critic = network_builder.create_network('actor_critic')
        return A2CAgent(
            self.action_settings,
            self.observation_settings,
            batch_size=config.batch_size,
            trace_length=config.trace_length,
            gamma=config.gamma,
            actor_lr=config.actor_lr,
            critic_lr=config.critic_lr,
            model_actor=actor,
            model_critic=critic
        )

    def get_agent(self, agent_type, network_type, config):
        if agent_type == 'a2c':
            return self.produce_a2c(config)
        elif agent_type == 'ddqn':
            return self.produce_ddqn(network_type, config)
