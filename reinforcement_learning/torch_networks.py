import torch
import pytorch_lightning as pl
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class MovementTorch(nn.Module):
    def __init__(self, source_in_size, result_in_size, moves_out_size, dense_size, dropout_rate):
        nn.Module.__init__(self)
        self.source_in_size = source_in_size
        self.result_in_size = result_in_size
        self.moves_out_size = moves_out_size
        self.dense_size = dense_size
        # @TODO cover the binarisation of the input
        input_size = self.source_in_size + self.result_in_size
        self.l1 = nn.Linear(input_size, dense_size)
        self.d1 = nn.Dropout(p=dropout_rate)
        # self.l2 = nn.Linear(dense_size, dense_size)
        # self.d2 = nn.Dropout(p=dropout_rate)
        self.l3 = nn.Linear(dense_size, dense_size)
        self.d3 = nn.Dropout(p=dropout_rate)
        self.l4 = nn.Linear(dense_size, moves_out_size)


    def forward(self, source, canvas):
        # print("sc,", source, canvas)
        x = torch.cat((source, canvas),dim=1)
        x = self.l1(x)
        x = self.d1(x)
        x = F.relu(x)

        x = self.l2(x)
        x = self.d2(x)
        x = F.relu(x)

        x = self.l3(x)
        x = self.d3(x)
        x = F.relu(x)

        x = self.l4(x)
        x = torch.tanh(x)
        return x

    def make_metrics(self, outputs, labels):
        mse = pl.metrics.functional.mse(outputs, labels)
        return mse

class MovementBinarised(MovementTorch):

    def forward(self, source, canvas):
        source_bin = (source>0)*1.0
        canvas_bin = (canvas>0)*1.0
        return MovementTorch.forward(self, source_bin, canvas_bin)


class CombinedNetworkTorch(nn.Module):
    def __init__(self, movement, categorisation, mov_trainable=True, cat_trainable=True):
        nn.Module.__init__(self)
        self.mov = movement
        self.cat = categorisation
        self.mov_trainable=mov_trainable
        self.cat_trainable=cat_trainable


    def forward(self, src, canv):
        x1 = self.mov(src, canv)
        x2 = self.cat(src)
        return x1, x2

    def optimise(self, net_output, labels):
        mov_labels, cat_labels = labels
        mov_output, cat_output = net_output
        optimizer = optim.Adam(self.parameters(), lr=0.0001)
        cat_labels_ = pl.metrics.functional.to_categorical(cat_output)
        mov_val = mov_output.max(1).values.unsqueeze(1)
        mov_loss = F.mse_loss(mov_val, mov_labels)
        cat_loss = F.cross_entropy(cat_output, cat_labels_)
        total_loss = mov_loss + cat_loss
        total_loss.backward()
        optimizer.step()
        metrics = self.make_metrics((mov_output, cat_output),(mov_labels, cat_labels))
        cat_mse, cat_acc = metrics
        hist = {
            'mov_loss': mov_loss.item(),
            'cat_loss': cat_loss.item(),
            'cat_mse': cat_mse.item(),
            'cat_acc': cat_acc.item(),
        }
        return hist

    def make_metrics(self, outputs, labels):
        cat_met = self.cat.make_metrics(outputs[1], labels[1])
        return cat_met

class CombinedAddedNetwork(CombinedNetworkTorch):
    def __init__(self, *args, **kwargs):
        CombinedNetworkTorch.__init__(self, *args, **kwargs)
        self.combined_input_size = self.mov.moves_out_size + self.cat.out_features
        self.e_l1 = nn.Linear(self.combined_input_size, self.mov.dense_size)
        self.e_l2 = nn.Linear(self.mov.dense_size, self.mov.dense_size)
        self.e_l3 = nn.Linear(self.mov.dense_size, self.mov.moves_out_size)
        self.e_l4 = nn.Linear(self.mov.dense_size, self.cat.out_features)
        self.e_d1 = nn.Dropout(p=self.mov.dropout_rate)
        self.e_d2 = nn.Dropout(p=self.mov.dropout_rate)


    def forward(self, src, canv):
        x1, x2 = CombinedNetworkTorch.forward(src, canv)
        x = torch.cat((x1, x2),dim=1)

        x = self.e_l1(x)
        x = self.e_d1(x)
        x = F.relu(x)

        x = self.e_l2(x)
        x = self.e_d2(x)
        x_m = F.relu(x)

        x1 = self.e_l3(x_m)
        x1 = F.softmax(x1, dim=1)

        x2 = self.e_l4(x_m)
        x2 = F.sigmoid(x2)
        return x1, x2
