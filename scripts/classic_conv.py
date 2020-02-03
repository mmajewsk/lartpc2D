from datetime import datetime
from tensorflow.keras.callbacks import ModelCheckpoint
import numpy as np
from lartpc_game import data
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.utils import to_categorical
from actors.networks import categorisation_network
from lartpc_game.game.dims import neighborhood2d
from common_configs import ClassicConfConfig


def add_neighbours(ind_x, ind_y, mix=True):
    add_window = neighborhood2d(3)
    a,b = add_window.T #spliting indeces into two axes
    repeated_x = np.repeat(np.array([a]), len(ind_x), axis=0)
    repeated_y = np.repeat(np.array([b]), len(ind_y), axis=0)
    ind_x_t = np.array([ind_x]).T
    ind_y_t = np.array([ind_y]).T
    xflat = (ind_x_t+repeated_x).flatten()
    yflat = (ind_y_t+repeated_y).flatten()
    points = np.array([xflat, yflat]).T
    unique_points = np.unique(points, axis=0)
    if mix:
        np.random.shuffle(unique_points)
    return unique_points.T


def map_to_conv_data(source, target, input_size=3, output_size=3, extended_neighbours=False):
    ind_x, ind_y = np.nonzero(target)
    if extended_neighbours:
        ind_x, ind_y = add_neighbours(ind_x, ind_y, mix=True)
    input_window = neighborhood2d(input_size)
    output_window = neighborhood2d(output_size)
    X, Y = [], []
    size = target.shape[0]
    for x_c, y_c in zip(ind_x, ind_y):
        if x_c <= 1 or x_c >=size-2 or y_c <= 1 or y_c >=size-2 :
            continue
        coords = np.array([x_c, y_c])
        input_window_on_map = coords + input_window
        output_window_on_map = coords + output_window
        i_w_x, i_w_y = input_window_on_map.T
        o_w_x, o_w_y = output_window_on_map.T
        target_on_map = target[o_w_x,o_w_y]
        test_target = target[i_w_x, i_w_y]
        trgt = to_categorical(target_on_map, num_classes=3)
        trgt = np.array(trgt)
        if len(trgt.shape)!=3:
            trgt = trgt[np.newaxis,:]
        #trgt = trgt.flatten()
        src = source[i_w_x, i_w_y]
        X.append(src)
        Y.append(trgt)
    return X,Y

def conv_net_gdata_generator(data_generator: data.LartpcData, network_config):
    for map_number in range(len(data_generator)):
        #source, target = data_generator[map_number]
        source, target = data_generator.random()
        x,y = map_to_conv_data(
            source,
            target,
            input_size=network_config.input_window_size,
            output_size=network_config.output_window_size,
            extended_neighbours=network_config.extended_neighbours
        )
        yield x,y


def batch_generator(data_generator : data.LartpcData, network_config: ClassicConfConfig):
    batch_size = network_config.batch_size
    while True:
        X, Y = [], []
        weights = [1.        , 3.98, 6.79]
        weights = np.array(weights)
        for x,y in conv_net_gdata_generator(data_generator, network_config):
            if len(X) <= batch_size:
                X += x
                Y += y
            else:
                batch_y = Y[:batch_size]
                batch_x = np.concatenate(X[:batch_size])
                batch_x, batch_y = np.expand_dims(batch_x,1), batch_y
                batch_y_multiplied =  weights * np.array(batch_y)
                batch_y_multiplied = np.squeeze(batch_y_multiplied,1)
                batch_weights = np.sum(batch_y_multiplied, axis=2)
                batch_weights = np.sum(batch_weights, axis=1)
                batch_y = np.array(batch_y)
                batch_y = batch_y.reshape((batch_size,1,batch_y.shape[2]*batch_y.shape[3]))
                batch =  batch_x, batch_y, batch_weights
                X, Y = X[batch_size:], Y[batch_size:]
                yield batch

if __name__=="__main__":
    #dump_filepath = '../dump' # local
    dump_filepath = '/home/mwm/repositories/content/dump'  # home cluster
    data_generator = data.LartpcData.from_path(dump_filepath)
    train_data = data_generator.get_range(0,800)
    validation_data = data_generator.get_range(800,1000)
    network_config = ClassicConfConfig()
    logdir = "logs/scalars/" + datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = TensorBoard(log_dir=logdir)
    model = categorisation_network(network_config=network_config)
    mc = ModelCheckpoint('../assets/model_dumps/categorisation/model{epoch:08d}.h5', period=30)
    model.fit_generator(
        batch_generator(train_data, network_config),
        steps_per_epoch=200,
        epochs=300,
        callbacks=[tensorboard_callback, mc],
        validation_data=batch_generator(validation_data, network_config),
        validation_steps=100,
    )


