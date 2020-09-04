import sys
sys.path.append("..")
from pathlib import Path
from pytorch_lightning.loggers import NeptuneLogger, TensorBoardLogger
import pytorch_lightning as pl
from scripts.classic_conv_torch import batch_generator
from lartpc_game.data import LartpcData
from common_configs import ClassicConfConfig
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import dataclasses
import os


class CatLt(pl.LightningModule):

    def __init__(self, dense_size, dropout_rate, result_feature_size):
        super(CatLt, self).__init__()
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

    def configure_optimizers(self):
        optimizer = optim.SGD(self.parameters(),lr=0.00001, momentum=0.9, nesterov=True)
        gamma=1e-6
        time_decay = lambda x: 1/(1+gamma*x)
        my_lr_scheduler = optim.lr_scheduler.MultiplicativeLR(optimizer, lr_lambda=time_decay)
        return [optimizer], [my_lr_scheduler]

    def make_metrics(self, outputs, labels):
        oh_labels = pl.metrics.functional.to_onehot(labels, num_classes=3)
        oh_labels = oh_labels.type(torch.float)
        mse = pl.metrics.functional.mse(outputs, torch.squeeze(oh_labels))
        cat_outputs = pl.metrics.functional.to_categorical(outputs)
        acc = pl.metrics.functional.accuracy(cat_outputs, labels)
        return mse, acc

    def training_step(self, batch, batch_idx):
        inputs, labels, weights = batch
        outputs = self(inputs)
        loss = F.cross_entropy(outputs, labels, weights)
        result = pl.TrainResult(loss)
        mse, acc = self.make_metrics(outputs, labels)
        result.log('train_loss', loss)
        result.log('train_mse', mse)
        result.log('train_acc', acc)
        result.log('train_loss_epoch', loss, on_step=False, on_epoch=True, reduce_fx=torch.mean)
        result.log('train_mse_epoch', mse, on_step=False, on_epoch=True, reduce_fx=torch.mean)
        result.log('train_acc_epoch', acc, on_step=False, on_epoch=True, reduce_fx=torch.mean)
        return result

    def validation_step(self, batch, batch_idx):
        inputs, labels, weights = batch
        outputs = self(inputs)
        loss = F.cross_entropy(outputs, labels, weights)
        result = pl.EvalResult(checkpoint_on=loss)
        mse, acc = self.make_metrics(outputs, labels)
        result.log('val_loss', loss)
        result.log('val_mse', mse)
        result.log('val_acc', acc)
        result.log('val_loss_epoch', loss, on_step=False, on_epoch=True, reduce_fx=torch.mean)
        result.log('val_mse_epoch', mse, on_step=False, on_epoch=True, reduce_fx=torch.mean)
        result.log('val_acc_epoch', acc, on_step=False, on_epoch=True, reduce_fx=torch.mean)
        return result


class  LartpcGenData(torch.utils.data.Dataset):
    def __init__(self, gen, steps_limit, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Initialize an iterator over the dataset.
        self.gen = gen
        self.iteration = 0
        self.steps_limit = steps_limit

    def __len__(self):
        return self.steps_limit

    def __iter__(self):
        return self

    def __next__(self):
        if self.iteration < self.steps_limit:
            batch = next(self.gen)
            inputs_, labels_, _ = batch
            inputs_ = np.squeeze(inputs_, axis=1)
            labels_ = np.squeeze(labels_, axis=1)
            labels_ = np.squeeze(labels_)
            labels_ = np.argmax(labels_, axis=1)
            inputs = torch.from_numpy(inputs_)
            labels = torch.from_numpy(labels_)
            weights = torch.tensor([1.     , 3.98, 6.79], dtype=torch.float)
            return inputs, labels, weights
        else:
            return None

class SavingCallback(pl.Callback):
    def on_epoch_end(self, trainer, pl_module):
        tt_logger = trainer.logger[0]
        checkpoint_dir = (Path(tt_logger.experiment.log_dir) / "checkpoints")
        if pl_module.current_epoch%30 == 0:
            checkpoint_path = checkpoint_dir/"{}_checkpoint.ckpt".format(pl_module.current_epoch)
            trainer.save_checkpoint(str(checkpoint_path))

# class EpochsMeanCallback(pl.Callback):
#     def __init__(self, metric_name, *args, **kwargs):
#         pl.Callback.__init__(*args, **kwargs)
#         self.metric_name = metric_name
#         self.metric_list = []

#     def add(self, trainer, pl_module):
#         self.metric_list.append(self.trainer.callback_metrics[self.metric_name])

#     def reset(self):
#         self.metric_list = []

#     def apply(self, fun):
#         return fun(self.metric_list)

#     def on_epoch_end(self, trainer, pl_module):
#         trainer.logger.log()

# class EMCTrain(EpochsMeanCallback):
#     def on_train_end(self, trainer, pl_module):
#         self.add(trainer, pl_module)



if __name__ == "__main__":

    dump_filepath = '../assets/dump'  # home cluster
    network_config = ClassicConfConfig()
    dataset = LartpcData.from_path(dump_filepath)
    train_data = dataset.get_range(0,800)
    validation_data = dataset.get_range(800,1000)
    train_gen = batch_generator(train_data, network_config)
    val_gen = batch_generator(validation_data, network_config)
    steps_per_epoch=200
    val_steps=40
    td =  LartpcGenData(train_gen, steps_limit=steps_per_epoch)
    vd =  LartpcGenData(val_gen, steps_limit=val_steps)
    net = CatLt(network_config.dense_size, network_config.dropout_rate, 3)
    epochs=300
    gpus = int(torch.cuda.is_available())
    neptune_logger = NeptuneLogger(
        api_key=os.environ['NEPTUNE_API_TOKEN'],
        experiment_name="conv@{}".format(os.uname()[1]),
        project_name="mmajewsk/lartpc-conv",  # Optional,
        params=dataclasses.asdict(network_config),  # Optional,
    )
    callbacks = [SavingCallback()]
    default_logger = TensorBoardLogger(
        save_dir=os.getcwd(),
        version=1,
        name='lightning_logs'
    )
    trainer = pl.Trainer(
        gpus=gpus,
        min_epochs=4,
        max_epochs=epochs,
        callbacks=callbacks,
        logger = [default_logger, neptune_logger]
    )
    trainer.fit(net, td, vd)
