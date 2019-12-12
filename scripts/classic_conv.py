from keras.models import Model
from keras.layers import Dense, Activation, Input, Dropout
from keras.optimizers import Adam
from datetime import datetime
import numpy as np
import data
from keras.callbacks import TensorBoard
from keras.utils import to_categorical
from envs.dims import neighborhood2d


def categorisation_network(input_parameters, output_parameters, other_params):
    source_feature_size = input_parameters['source_feature_size']
    result_output_size = output_parameters['result_output']
    source_input = Input((None, source_feature_size), name='source_input')
    dense_size = source_feature_size**2
    dropout_rate = 0.2
    l = Dense(dense_size)(source_input)
    l = Activation("relu")(l)
    l = Dropout(rate=dropout_rate)(l)
    l = Dense(dense_size)(l)
    l = Activation("relu")(l)
    l = Dropout(rate=dropout_rate)(l)
    l = Dense(dense_size)(l)
    l = Activation("relu")(l)
    l = Dropout(rate=dropout_rate)(l)
    l = Dense(dense_size)(l)
    l = Activation("relu")(l)
    l = Dropout(rate=dropout_rate)(l)
    l = Dense(result_output_size)(l)
    output = Activation("sigmoid")(l)
    model = Model(inputs=[source_input], outputs=[output])
    adam = Adam(lr=0.00001)
    model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['mse','acc'])
    return model

def map_to_conv_data(source, target):
    ind_x, ind_y = np.nonzero(target)
    window = neighborhood2d(3)
    X, Y = [], []
    size = target.shape[0]
    for x_c, y_c in zip(ind_x, ind_y):
        if x_c <= 1 or x_c >=size-2 or y_c <= 1 or y_c >=size-2 :
            continue
        coords = np.array([x_c, y_c])
        window_on_map = coords + window
        w_x, w_y = window_on_map.T
        trgt = to_categorical(target[w_x,w_y], num_classes=3)
        trgt = np.array(trgt)
        #trgt = trgt.flatten()
        src = source[w_x,w_y]
        X.append(src)
        Y.append(trgt)
    return X,Y

def conv_net_gdata_generator(data_generator: data.LartpcData):
    for map_number in range(len(data_generator)):
        #source, target = data_generator[map_number]
        source, target = data_generator.random()
        x,y = map_to_conv_data(source, target)
        yield x,y


def batch_generator(data_generator : data.LartpcData, batch_size=32):
    while True:
        X, Y = [], []
        weights = [1.,1.76,3.13]
        weights = np.array(weights)
        for x,y in conv_net_gdata_generator(data_generator):
            if len(X) <= batch_size:
                X += x
                Y += y
            else:
                batch_y = Y[:batch_size]
                batch_x = np.concatenate(X[:batch_size])
                batch_x, batch_y = np.expand_dims(batch_x,1), batch_y
                batch_y_multiplied =  weights * batch_y
                batch_y_multiplied = np.squeeze(batch_y_multiplied,1)
                batch_weights = np.sum(batch_y_multiplied, axis=2)
                batch_weights = np.sum(batch_weights, axis=1)
                batch_y = np.array(batch_y)
                batch_y = batch_y.reshape((batch_size,1,batch_y.shape[2]*batch_y.shape[3]))
                batch =  batch_x, batch_y, batch_weights
                X, Y = X[batch_size:], Y[batch_size:]
                yield batch

if __name__=="__main__":
    data_generator = data.LartpcData('../dump')
    input_params = {
        'source_feature_size': 9,
    }
    output_params = {
        'result_output': 9*3,
    }
    other_params = {

    }
    logdir = "logs/scalars/" + datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = TensorBoard(log_dir=logdir)
    model = categorisation_network(input_params, output_params, other_params)
    model.fit_generator(batch_generator(data_generator), steps_per_epoch=100, epochs=50, callbacks=[tensorboard_callback])


