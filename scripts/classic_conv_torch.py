#!/usr/bin/env python
# coding: utf-8

import sys
sys.path.append("..")
from datetime import datetime
from sklearn.preprocessing import OneHotEncoder
import numpy as np
from lartpc_game import data
from common_configs import ClassicConfConfig
from lartpc_game.game.dims import neighborhood2d
from lartpc_game.agents.tools import to_categorical_


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
        trgt = to_categorical_(target_on_map, num_classes=3)
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
                batch_x = np.stack(X[:batch_size])
                batch_x, batch_y = np.expand_dims(batch_x,1), batch_y
                batch_y_multiplied =  weights * np.array(batch_y)
                batch_y_multiplied = np.squeeze(batch_y_multiplied,1)
                batch_weights = np.sum(batch_y_multiplied, axis=2)
                batch_weights = np.sum(batch_weights, axis=1)
                batch_y = np.array(batch_y)
                batch =  batch_x, batch_y, batch_weights
                X, Y = X[batch_size:], Y[batch_size:]
                yield batch

import torch
import torch.nn as nn
import torch.nn.functional as F


class CategorisationNetTorch(nn.Module):

    def __init__(self, dense_size, dropout_rate, result_feature_size):
        super(CategorisationNetTorch, self).__init__()
        self.dense_size = dense_size
        self.dropout_rate = dropout_rate
        self.l1 = nn.Linear(25, dense_size)
        self.b1 = nn.BatchNorm1d(dense_size)
        self.d1 = nn.Dropout(p=self.dropout_rate)

        self.l2 = nn.Linear(dense_size, dense_size)
        self.b2 = nn.BatchNorm1d(dense_size)
        self.d2 = nn.Dropout(p=self.dropout_rate)

        self.l3 = nn.Linear(dense_size, dense_size)
        self.b3 = nn.BatchNorm1d(dense_size)
        self.d3 = nn.Dropout(p=self.dropout_rate)

        self.l4 = nn.Linear(dense_size, dense_size)
        self.b4 = nn.BatchNorm1d(dense_size)
        self.d4 = nn.Dropout(p=self.dropout_rate)

        self.l5 = nn.Linear(dense_size, out_features=result_feature_size)

    def forward(self, x):
        x = self.l1(x)
        x = self.b1(x)
        x = F.relu(x)
        x = self.d1(x)

        x = F.relu(self.b2(self.l2(x)))
        x = self.d2(x)

        x = F.relu(self.b3(self.l3(x)))
        x = self.d3(x)

        x = F.relu(self.b4(self.l4(x)))
        x = self.d4(x)

        x = torch.sigmoid(self.l5(x))
        return x

class LartpcDataset(torch.utils.data.Dataset, data.LartpcData):
    def __init__(self, source_list, target_list):
        torch.utils.data.Dataset.__init__(self)
        data.LartpcData.__init__(self, source_list, target_list)


import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm, trange
if __name__ == "__main__":
    epochs=300
    steps_per_epoch=200

    dump_filepath = '../assets/dump'  # home cluster
    dataset = LartpcDataset.from_path(dump_filepath)
    train_data = dataset.get_range(0,800)
    validation_data = dataset.get_range(800,1000)
    network_config = ClassicConfConfig()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Using fevice: {}".format(device))

    writer = SummaryWriter("logs/torch/"+ datetime.now().strftime("%Y%m%d-%H%M%S"))

    net = CategorisationNetTorch(network_config.dense_size, network_config.dropout_rate, 3)
    net = net.to(device)
    weights2 = torch.tensor([1.     , 3.98, 6.79], dtype=torch.float).to(device)
    criterion = nn.CrossEntropyLoss(weight =weights2)
    optimizer = optim.SGD(net.parameters(),lr=0.00001, weight_decay=0.999999, momentum=0.9, nesterov=True)

    for epoch in trange(epochs):
        running_loss = 0.0
        generator = batch_generator(train_data, network_config)
        optimizer.zero_grad()
        for i in trange(steps_per_epoch):
            data = next(generator)
            inputs_, labels_, _ = data
            inputs_ = np.squeeze(inputs_, axis=1)
            labels_ = np.squeeze(labels_, axis=1)
            labels_ = np.squeeze(labels_)
            labels_ = np.argmax(labels_, axis=1)
            inputs = torch.from_numpy(inputs_)
            labels = torch.from_numpy(labels_)

            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = net(inputs)

            loss = criterion(outputs, labels)
            running_loss += loss.item()
            loss.backward()
            optimizer.step()
        writer.add_scalar("Loss/train", running_loss/steps_per_epoch, epoch)
        print("Loss: {}".format(running_loss/steps_per_epoch))
        writer.flush()
    writer.close()

