from keras import Input, Model
from keras.models import Model, load_model
from keras.layers import Input, Concatenate, Dense, Activation, Dropout, BatchNormalization
from keras.optimizers import Adam, SGD
from keras.models import load_model
from common_configs import ClassicConfConfig


def create_movement_inputs(source_feature_size, result_feature_size):
    source_data_input = Input((None,source_feature_size), name='source_input')
    result_data_input = Input((None, result_feature_size), name='result_input')
    return source_data_input, result_data_input

def create_movement_output(previous_layer, possible_moves):
    output = Dense(possible_moves,name='output_movement' )(previous_layer)
    #output = Activation('softmax', name='output_movement')(output)
    return output


# only movement network

class ParameterBasedNetworks:
    def __init__(self, input_parameters, output_parameters, other_params):
        self.input_parameters = input_parameters
        self.output_parameters = output_parameters
        self.other_params = other_params

    def movement_network(self, source_in, result_in):
        dense_size = self.other_params['dense_size']
        dropout_rate = self.other_params['dropout_rate']
        all_input = Concatenate()([source_in, result_in])
        # @TODO result in is currently binarised, need to do in network
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

    def movement_and_category(self, category_network: Model):
        source_in, result_in = create_movement_inputs(**self.input_parameters)
        output_movement = self.movement_network(source_in, result_in)
        output_category = category_network(source_in)
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