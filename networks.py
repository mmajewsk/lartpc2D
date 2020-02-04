import tensorflow as tf
from keras import Input, Model
from keras.models import Model, load_model
from keras.layers import Input, Concatenate, Dense, Activation
from keras.layers import Dropout, BatchNormalization, Lambda, Layer, Flatten
from keras.optimizers import Adam, SGD
from keras.models import load_model
from common_configs import ClassicConfConfig, TrainerConfig


def create_movement_inputs(source_feature_size, result_feature_size):
    source_data_input = Input((None,source_feature_size), name='source_input')
    result_data_input = Input((None, result_feature_size), name='result_input')
    return source_data_input, result_data_input

def create_movement_output(previous_layer, possible_moves):
    output = Dense(possible_moves,name='output_movement' )(previous_layer)
    #output = Activation('softmax', name='output_movement')(output)
    return output

def random_category(t):
    return tf.random.uniform([1,1,3])

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


class ParameterBasedNetworks:
    def __init__(self, input_parameters, output_parameters, other_params, action_factory, observation_factory):
        self.input_parameters = input_parameters
        self.output_parameters = output_parameters
        self.other_params = other_params
        self.action_factory = action_factory
        self.observation_factory = observation_factory

    def movement_network(self, source_in, result_in):
        dense_size = self.other_params['dense_size']
        dropout_rate = self.other_params['dropout_rate']
        #source_clip =  tf.keras.layers.Lambda(lambda x : tf.cast(x>0.0,tf.float32))(source_in)
        source_clip =  Lambda(fun)(source_in)
        fun_cat = fun_cat_produce(self.observation_factory.categories, self.input_parameters['result_feature_size'])
        result_clip =  Lambda(fun_cat, output_shape=produce_shaper(self.observation_factory.categories))(result_in)
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
        output = create_movement_output(l, **self.output_parameters)
        return output

    def movement_network_compiled(self) -> Model:
        source_in, result_in = create_movement_inputs(**self.input_parameters)
        output = self.movement_network(source_in, result_in)
        model = Model(
            inputs=[source_in, result_in],
            outputs = [output],
        )
        compile_kwrgs = {}
        compile_kwrgs['loss'] = {
            'output_movement': 'mse'
        }
        compile_kwrgs['metrics'] = ['mae', 'acc']
        adam = Adam(lr=0.00001)
        model.compile(optimizer=adam, **compile_kwrgs)
        return model

    def movement_network_random_category(self):
        source_in, result_in = create_movement_inputs(**self.input_parameters)
        output = self.movement_network(source_in, result_in)
        output_category = random_category(source_in)
        model = Model(
            inputs=[source_in, result_in],
            outputs=[output, output_category],
        )
        compile_kwrgs = {}
        compile_kwrgs['loss'] = {
            'output_movement': 'mse'
        }
        compile_kwrgs['metrics'] = ['mae', 'acc']
        adam = Adam(lr=0.00001)
        model.compile(optimizer=adam, **compile_kwrgs)
        return model


    def combine_movement_category(self, movement_network,  category_network, mov_trainable=True, cat_trainable=True):
        source_in, result_in = create_movement_inputs(**self.input_parameters)
        output_movement = movement_network(source_in, result_in)
        output_category = category_network(source_in)
        output_category.trainable = cat_trainable
        output_movement.trainable = mov_trainable
        model = Model(
            inputs=[source_in, result_in],
            outputs=[output_movement, output_category],
        )
        compile_kwrgs = {}
        compile_kwrgs['loss'] = {
            'output_movement': 'mse',
            'model_1': 'categorical_crossentropy'
        }
        compile_kwrgs['metrics'] = ['mse','mae', 'acc']
        adam = Adam(lr=0.00001)
        model.compile(optimizer=adam, **compile_kwrgs)
        return model

    def add_extra(self, output_mov: Layer, output_cat: Layer):
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




    def extra_layers(self, movement_network,  category_network, mov_trainable=True, cat_trainable=True):
        source_in, result_in = create_movement_inputs(**self.input_parameters)
        output_movement = movement_network(source_in, result_in)
        output_category = category_network(source_in)
        output_category.trainable = cat_trainable
        output_movement.trainable = mov_trainable
        extra_movement, extra_category = self.add_extra(output_movement, output_category)
        model = Model(
            inputs=[source_in, result_in],
            outputs=[extra_movement, extra_category]
        )

def categorisation_network(network_config : ClassicConfConfig):
    source_feature_size = network_config.source_feature_size
    result_output_size = network_config.result_output
    source_input = Input((None, source_feature_size), name='source_input')
    dense_size = network_config.dense_size
    dropout_rate = network_config.dropout_rate
    l = Dense(dense_size)(source_input)
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
    l = Dense(result_output_size)(l)
    output = Activation("sigmoid")(l)
    model = Model(inputs=[source_input], outputs=[output])
    adam = Adam(lr=0.00001)
    sgd = SGD(lr=0.00001, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['mse','acc'])
    return model

def create_network_factory(network_type, network_builder: ParameterBasedNetworks, config: TrainerConfig ):
    if network_type=='empty':
        nm_factory = lambda : None
    elif network_type=='movement':
        movement_network = network_builder.movement_network
        categorisation_network = Lambda(random_category, name='model_1')
        nm_factory = lambda : network_builder.combine_movement_category(movement_network, categorisation_network)
    elif network_type=='read_conv':
        category_network = load_model(config.conv_model_path)
        movement_network = network_builder.movement_network
        nm_factory =  lambda : network_builder.combine_movement_category(
            movement_network,
            category_network,
            cat_trainable = config.conv_trainable,
            mov_trainable = config.mov_trainable
        )
    elif network_type=='read_both':
        category_network = load_model(config.conv_model_path)
        movement_network = load_model(config.movement_model_path)
        nm_factory = lambda : network_builder.combine_movement_category(
            movement_network,
            category_network,
            cat_trainable=config.conv_trainable,
            mov_trainable=config.mov_trainable
        )
    elif network_type=='extra_layers':
        category_network = load_model(config.conv_model_path)
        movement_network = load_model(config.movement_model_path)
        nm_factory = lambda: network_builder.extra_layers(
            movement_network,
            category_network,
            cat_trainable=config.conv_trainable,
            mov_trainable=config.mov_trainable
        )
    else:
        raise ValueError
    return nm_factory