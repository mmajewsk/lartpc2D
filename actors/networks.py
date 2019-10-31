from keras.models import Model, load_model
from keras.layers import Input, Concatenate, Dense, Activation, Dropout
from keras.optimizers import Adam


def create_inputs(source_feature_size, result_feature_size):
    source_data_input = Input((None,source_feature_size), name='source_input')
    result_data_input = Input((None, result_feature_size), name='result_input')
    return source_data_input, result_data_input

def create_output(previous_layer,possible_moves):
    output = Dense(possible_moves,name='movement_output' )(previous_layer)
    #output = Activation('softmax', name='movement_output')(output)
    return output


# only movement network
def movement_network(input_parameters, output_parameters, other_params) -> Model:
    dense_size = other_params['dense_size']
    dropout_rate = other_params['dropout_rate']
    source_in, result_in = create_inputs(**input_parameters)
    all_input = Concatenate()([source_in, result_in])
    l = Dense(dense_size)(all_input)
    l = Activation("relu")(l)
    l = Dropout(rate=dropout_rate)(l)
    l = Dense(dense_size)(l)
    l = Activation("relu")(l)
    l = Dropout(rate=dropout_rate)(l)
    l = Dense(dense_size)(l)
    l = Activation("relu")(l)
    l = Dropout(rate=dropout_rate)(l)
    output = create_output(l, **output_parameters)


    model = Model(
        inputs=[source_in, result_in],
        outputs = [output],
    )
    compile_kwrgs = {}
    compile_kwrgs['loss'] = {
        'movement_output': 'mse'
    }
    compile_kwrgs['metrics'] = ['mae', 'acc']
    adam = Adam(lr=0.00001)
    model.compile(optimizer=adam, **compile_kwrgs)
    return model