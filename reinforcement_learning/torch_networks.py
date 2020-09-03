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
        # @TODO cover the binarisation of the input
        input_size = self.source_in_size + self.result_in_size
        self.l1 = nn.Linear(input_size, dense_size)
        self.d1 = nn.Dropout(p=dropout_rate)
        self.l2 = nn.Linear(dense_size, dense_size)
        self.d2 = nn.Dropout(p=dropout_rate)
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
        x = F.softmax(x, dim=1)
        #@TODO check if thath shouldnt be linear
        return x

    def make_metrics(self, outputs, labels):
        mse = pl.metrics.functional.mse(outputs, labels)
        return mse


class CombinedNetworkTorch(nn.Module):
    def __init__(self, movement, categorisation):
        nn.Module.__init__(self)
        self.mov = movement
        self.cat = categorisation


    def forward(self, src, canv):
        x2 = self.cat(src)
        x1 = self.mov(src, canv)
        return x1, x2

    def optimise(self, input, labels):
        mov_labels, cat_labels = labels
        mov_output, cat_output = self(*input)
        optimizer = optim.Adam(self.parameters(), lr=0.00001)
        cat_labels_ = pl.metrics.functional.to_categorical(cat_output)
        mov_loss = F.mse_loss(mov_output, mov_labels)
        cat_loss = F.cross_entropy(cat_output, cat_labels_)
        (mov_loss+cat_loss).backward()
        optimizer.step()
        metrics = self.make_metrics((mov_output, cat_output),(mov_labels, cat_labels))
        mov_mse, (cat_mse, cat_acc) = metrics
        hist = {
            'mov_loss': mov_loss.item(),
            'cat_loss': cat_loss.item(),
            'mov_mse': mov_mse.item(),
            'cat_mse': cat_mse.item(),
            'cat_acc': cat_acc.item(),
        }
        return hist

    def make_metrics(self, outputs, labels):
        cat_met = self.cat.make_metrics(outputs[1], labels[1])
        mov_met = self.mov.make_metrics(outputs[0], labels[0])
        return mov_met, cat_met
