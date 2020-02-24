import tensorflow as tf
from dataclasses import asdict
from keras import Input, Model
from keras.models import Model, load_model
from keras.layers import Input, Concatenate, Dense, Activation
from keras.layers import Dropout, BatchNormalization, Lambda, Layer, Flatten
from keras.optimizers import Adam, SGD
from keras.models import load_model
from common_configs import ClassicConfConfig, TrainerConfig
from abc import ABC, abstractmethod


def create_movement_inputs(source_feature_size, result_feature_size):
    source_data_input = Input((None,source_feature_size), name='source_input')
    result_data_input = Input((None, result_feature_size), name='result_input')
    return source_data_input, result_data_input

def create_movement_output(previous_layer, possible_moves):
    output = Dense(possible_moves,name='output_movement' )(previous_layer)
    #output = Activation('softmax', name='output_movement')(output)
    return output


def fun(x):
    x = tf.cast(x > 0.0, tf.int32)
    x = tf.cast(x, tf.float32)
    return x

#def fun2(x):
#    comparison = tf.not_equal(x, tf.constant(0.0, dtype=tf.float32))
#    conditional_assignment_op = x.assign(tf.where(comparison, tf._like(x), a))
# only movement network
def fun_cat_produce(categories, input_result_size):
    def fun(x):
        original = tf.shape(x)
        x = tf.cast(x > 0.0, tf.bool)
        x = tf.reshape(x, [-1, categories])
        x = tf.reduce_any(x, 1)
        x = tf.cast(x, tf.float32)
        x = tf.reshape(x, [original[0], original[1], input_result_size // categories])
        # x = tf.reshape(x, [original[0],original[1], 25])
        return x

    return fun


def produce_shaper(categories):
    def fun_cat_output_shape(xsh_in):
        d1, d2, d3 = xsh_in
        return d1, d2, d3 //categories

    return fun_cat_output_shape

# MovementNetwork(other_params['dense_size'], other_params['dropout_rate'],
#
#fun_cat = fun_cat_produce(self.observation_factory.categories, self.input_parameters['result_feature_size'])
#result_clip = Lambda(fun_cat, output_shape=produce_shaper(self.observation_factory.categories))(result_in)

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

class MovementNetwork(BaseNetwork):
    def __init__(self,
                 dense_size=None,
                 dropout_rate=None,
                 categories=None,
                 result_feature_size=None,
                 possible_moves=None,
                 source_feature_size=None,
                 *args,
                 **kwargs):
        BaseNetwork.__init__(self, *args, **kwargs)
        self.dense_size = dense_size
        self.dropout_rate = dropout_rate
        self.categories = categories
        self.result_feature_size = result_feature_size
        self.possible_moves = possible_moves # self.output_parameters
        self.source_feature_size = source_feature_size #self.input_parameters
        self.result_feature_size = result_feature_size

    def build(self, source_in, result_in):
        dense_size = self.dense_size
        dropout_rate = self.dropout_rate
        #source_clip =  tf.keras.layers.Lambda(lambda x : tf.cast(x>0.0,tf.float32))(source_in)
        source_clip =  Lambda(fun)(source_in)
        fun_cat = fun_cat_produce(self.categories, self.result_feature_size)
        result_clip =  Lambda(fun_cat, output_shape=produce_shaper(self.categories))(result_in)
        all_input = Concatenate(axis=2)([source_clip, result_clip])
        l = Dense(dense_size)(all_input)
        l = Activation("relu")(l)
        l = Dropout(rate=dropout_rate)(l)
        l = Dense(dense_size)(l)
        l = Activation("relu")(l)
        l = Dropout(rate=dropout_rate)(l)
        l = Dense(dense_size)(l)
        l = Activation("relu")(l)
        l = Dropout(rate=dropout_rate)(l)
        output = create_movement_output(l, self.possible_moves)
        return output

    def compiled(self) -> Model:
        source_in, result_in = create_movement_inputs(self.source_feature_size, self.result_feature_size)
        self.output = self.build(source_in, result_in)
        model = Model(
            inputs=[source_in, result_in],
            outputs = [self.output],
        )
        compile_kwrgs = {}
        compile_kwrgs['loss'] = {
            'output_movement': 'mse'
        }
        compile_kwrgs['metrics'] = ['mae', 'acc']
        adam = Adam(lr=0.00001)
        model.compile(optimizer=adam, **compile_kwrgs)
        self.model = model



class CategorisationNetwork(BaseNetwork):
    def __init__(
            self,
            source_feature_size=None,
            result_feature_size=None,
            dense_size=None,
            dropout_rate=None,
            *args,
            **kwargs
        ):
        BaseNetwork.__init__(self, *args, **kwargs)
        self.source_feature_size = source_feature_size #network_config.source_feature_size
        self.result_feature_size = result_feature_size
        self.dense_size = dense_size
        self.dropout_rate = dropout_rate

    def create_input(self):
        source_input = Input((None, self.source_feature_size), name='source_input')
        return source_input

    def build(self, source_in):
        dense_size = self.dense_size
        dropout_rate = self.dropout_rate
        l = Dense(dense_size)(source_in)
        l = BatchNormalization()(l)
        l = Activation("relu")(l)
        l = Dropout(rate=dropout_rate)(l)
        l = Dense(dense_size)(l)
        l = BatchNormalization()(l)
        l = Activation("relu")(l)
        l = Dropout(rate=dropout_rate)(l)
        l = Dense(dense_size)(l)
        l = BatchNormalization()(l)
        l = Activation("relu")(l)
        l = Dropout(rate=dropout_rate)(l)
        l = Dense(dense_size)(l)
        l = BatchNormalization()(l)
        l = Activation("relu")(l)
        l = Dropout(rate=dropout_rate)(l)
        l = Dense(self.result_feature_size)(l)
        output = Activation("sigmoid")(l)
        return output

    def compiled(self):
        source_input = self.create_input()
        self.output = self.build(source_input)
        model = Model(inputs=[source_input], outputs=[self.output])
        adam = Adam(lr=0.00001)
        sgd = SGD(lr=0.00001, decay=1e-6, momentum=0.9, nesterov=True)
        model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['mse', 'acc'])
        self.model = model


class RandomCategoryNetwork:
    def build(self,):
        return tf.random.uniform([1, 1, 3])

class CombinedNetwork(BaseNetwork):
    def __init__(self, net_a: MovementNetwork, net_b: CategorisationNetwork, a_trainable=True, b_trainable=True, *args, **kwargs):
        BaseNetwork.__init__(self, *args, **kwargs)
        self.net_a = net_a
        self.net_b = net_b
        self.a_trainable = a_trainable
        self.b_trainable = b_trainable
        self.output_a, self.output_b = None, None

    def build(self, source_in, result_in):
        self.layer_a = self.net_a.model([source_in, result_in])
        self.layer_b = self.net_b.model([source_in])
        if self.a_trainable is False:
            self.layer_a.trainable = self.a_trainable
        if self.b_trainable is False:
            self.layer_b.trainable = self.b_trainable
        return self.layer_a, self.layer_b

    def compiled(self, inputs=None, outputs=None):
        source_in, result_in = create_movement_inputs(self.net_a.source_feature_size, self.net_a.result_feature_size)
        self.output_a, self.output_b = self.build(source_in, result_in)
        #self.output_a.name = 'output_movement'
        #self.output_b.name = 'output_category'
        model = Model(
            inputs=[source_in, result_in],
            outputs=[self.output_a, self.output_b],
        )
        compile_kwrgs = {}
        compile_kwrgs['loss'] = {
            'output_movement': 'mse',
            'output_category': 'categorical_crossentropy'
        }
        compile_kwrgs['metrics'] = ['mse', 'mae', 'acc']
        adam = Adam(lr=0.00001)
        model.compile(optimizer=adam, **compile_kwrgs)
        self.model = model
        return self

    def load(self, path):
        print("not like this")
        net_a_path = path.parent/"mov_"+path.name
        net_b_path = path.parent/"cat_"+path.name
        print("Loading weights from: {} , {}".format(net_a_path, net_b_path))
        self.net_a.load(net_a_path)
        self.net_b.load(net_b_path)



class MovNormalCatRandom(CombinedNetwork):
    def __init__(self, mov: MovementNetwork=None, trainable=True):
        self.mov = mov
        self.cat = RandomCategoryNetwork().build()
        CombinedNetwork.__init__(self, self.mov, self.cat, a_trainable=trainable)

    def load(self, path):
        model = load_model(path, custom_objects={'tf':tf})
        self.output_a = model.get_layer('output_movement').output
        self.output_b = RandomCategoryNetwork().build()

    def compiled(self, inputs=None):
        source_in, result_in = create_movement_inputs(self.net_a.source_feature_size, self.net_a.result_feature_size)
        if self.output_a is None and self.output_b is None:
            self.output_a, self.output_b= self.build(source_in, result_in)
        model = Model(
            inputs=[source_in, result_in],
            outputs=[self.output_a, self.output_b],
        )
        compile_kwrgs = {}
        compile_kwrgs['loss'] = {
            'output_movement': 'mse'
        }
        compile_kwrgs['metrics'] = ['mae', 'acc']
        adam = Adam(lr=0.00001)
        model.compile(optimizer=adam, **compile_kwrgs)
        self.model = model
        return self

class CombinedExtraNetwork(CombinedNetwork):
    def __init__(self, *args, **kwargs):
        CombinedNetwork.__init__(self, *args, **kwargs)

    def build(self, source_in, result_in):
        output_mov, output_cat = CombinedNetwork.build(source_in=source_in, result_in=result_in)
        dropout_rate=0.2
        dense_size=64
        result_output_size = ClassicConfConfig.result_output
        l1 = Flatten()(output_mov)
        l2 = Flatten()(output_cat)
        l = Concatenate()([l1, l2])
        l = Dense(dense_size)(l)
        l = Activation("relu")(l)
        l = Dropout(rate=dropout_rate)(l)
        l = Dense(dense_size)(l)
        l = Activation("relu")(l)
        l = Dropout(rate=dropout_rate)(l)
        l = Dense(dense_size)(l)
        l = Activation("relu")(l)
        l = Dropout(rate=dropout_rate)(l)
        mov_extra = create_movement_output(l, self.other_params )
        o2 = Dense(result_output_size)(l)
        cat_extra = Activation("Sigmoid")(o2)
        return mov_extra, cat_extra

    def compiled(self):
        source_in, result_in = create_movement_inputs(self.net_a.source_feature_size, self.net_a.result_feature_size)
        extra_movement, extra_category = self.build(source_in, result_in)
        model = Model(
            inputs=[source_in, result_in],
            outputs=[extra_movement, extra_category]
        )
        self.model = model
        return self


class NetworkFactory:
    def __init__(self, input_parameters, output_parameters, other_params, action_factory, observation_factory, config: TrainerConfig, classic_config: ClassicConfConfig):
        self.input_parameters = input_parameters
        self.output_parameters = output_parameters
        self.other_params = other_params
        self.action_factory = action_factory
        self.observation_factory = observation_factory
        self.config = config
        self.classic_config = classic_config


    def create_network(self, network_type):
        if network_type=='empty':
            return None
        elif network_type=='movement':
            mov = MovementNetwork(categories=self.observation_factory.categories, **self.other_params, **self.output_parameters, **self.input_parameters )
            return MovNormalCatRandom(mov=mov)

        elif network_type=='read_conv':
            category_network = CategorisationNetwork(self.config.conv_model_path)
            mov = MovementNetwork(categories=self.observation_factory.categories, **self.other_params, **self.output_parameters, **self.input_parameters )
            return CombinedNetwork(net_a=mov, net_b=category_network, a_trainable=self.config.mov_trainable, b_trainable=self.config.conv_trainable)

        elif network_type=='read_both':
            cat = CategorisationNetwork(**asdict(self.classic_config), result_feature_size=self.classic_config.result_output, categories=self.observation_factory.categories)
            cat.compiled()
            modelload = load_model(self.config.conv_model_path, custom_objects={'tf':tf})
            cat.model.set_weights(modelload.get_weights())
            cat.model.name = 'output_category'
            mov = MovementNetwork(**self.output_parameters, **self.input_parameters, **self.other_params, categories=self.observation_factory.categories)
            modelload2 = load_model(self.config.movement_model_path, custom_objects={'tf':tf})
            mov.compiled()
            mov.model.set_weights(modelload2.get_weights())
            mov.model.name = 'output_movement'
            return CombinedNetwork(net_a=mov, net_b=cat, a_trainable=self.config.mov_trainable, b_trainable=self.config.conv_trainable).compiled()

        elif network_type=='extra_layers':
            assert False
            category_network = load_model(self.config.conv_model_path)
            movement_network = load_model(self.config.movement_model_path)
            nm_factory = lambda: self.extra_layers(
                movement_network,
                category_network,
                cat_trainable=self.config.conv_trainable,
                mov_trainable=self.config.mov_trainable
            )
        else:
            raise ValueError
