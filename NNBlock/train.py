"""
Compile and train functions.
"""

import os
import tensorflow as tf
import yaml
from datetime import datetime
from dataclasses import asdict
from typing import Union
from pydantic.dataclasses import dataclass

cb = tf.keras.callbacks

try:
    from nn import losses
    from nn.metrics import nlogE_MAE, nSigmaMAE
    from models import TwoTapesModel
except:
    from .nn import losses
    from .nn.metrics import nlogE_MAE, nSigmaMAE
    from .models import TwoTapesModel


# function to save verbose of training in txt file
class VerboseSave(cb.Callback):
    def __init__(self, path, freq: int = 1):
        self.freq = freq
        if self.freq == 0:
            self.freq = 1
        self.path = path
        file = open(self.path, 'w')
        file.close()

    def on_batch_end(self, batch, logs=None):
        if batch % self.freq == 0:
            file = open(self.path, 'a+')
            string = ""
            if logs is not None:
                for l in logs:
                    if str(logs[l]) == 'nan':
                        self.model.stop_training = True
                    string += f"{l}: {logs[l].__round__(4)} - "
                print(f"BATCH: {batch} \n{string}",
                      file=file)
            file.close()


# describe input of the compile_and_train()
@dataclass
class CompileAndTrainInput:
    # datasets_info: dict  # tuple[str, int, Any, Any]

    # train options
    num_of_epochs: int = 1
    cutting: float = 50.

    # compile params
    loss_E_name: str = 'MyLoss'
    loss_sigma_name: Union[str, None] = None  # if None, sigma loss = E loss
    lr: float = 0.0001
    lr_sigma: float = 0.0001
    apply_lr_decay: bool = False
    lr_decay: float = 0.05
    optimizer: Union[str, None] = None
    metrics: Union[list, None] = None
    weighted_metrics: Union[list, None] = None

    # callbacks
    checkpoint_cutting: float = 2.
    early_stop_params: dict = None
    verbose: bool = True
    verbose_cutting: float = 100.


def compile_and_train(model: TwoTapesModel, path_to_save: str, train_ds_with_info: tuple, test_ds_with_info: tuple,
                      input_args: CompileAndTrainInput):
    # set default early stopping parameters if necessary
    if input_args.early_stop_params is None:
        input_args.early_stop_params = {'monitor': 'val_loss_E', 'mode': 'min', 'patience': 5, 'min_delta': 1e-6}
    os.makedirs(path_to_save, exist_ok=True)

    # get datasets, batch size and total num and calc num of steps per epoch
    train_dataset, batch_size, total_num = train_ds_with_info
    test_dataset, _a, _b = test_ds_with_info
    steps_per_epoch = int((total_num // batch_size) // input_args.cutting)
    print("Steps per epoch:", steps_per_epoch)

    '''
    Compiling the model.
    '''
    # setting learning rate
    lr_initial = input_args.lr
    lr_sigma_initial = input_args.lr_sigma
    if input_args.apply_lr_decay:
        decay_rate = input_args.lr_decay ** (1 / input_args.num_of_epochs)
        input_args.lr = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=lr_initial,
                                                                       decay_steps=steps_per_epoch,
                                                                       decay_rate=decay_rate)
        input_args.lr_sigma = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=lr_sigma_initial,
                                                                             decay_steps=steps_per_epoch,
                                                                             decay_rate=decay_rate)

    # setting metrics
    if input_args.metrics is None:
        input_args.metrics = [nlogE_MAE(weighted=False), nSigmaMAE(weighted=False)]
    if input_args.weighted_metrics is None:
        input_args.weighted_metrics = [nlogE_MAE(weighted=True), nSigmaMAE(weighted=True)]

    # setting losses
    loss_E = getattr(losses, input_args.loss_E_name)()
    if input_args.loss_sigma_name is not None:
        loss_sigma = getattr(losses, input_args.loss_sigma_name)()
    else:
        loss_sigma = None  # if None, sigma loss = E loss

    # setting optimizer
    if input_args.optimizer is not None:
        optimizer = input_args.optimizer
    else:
        optimizer = tf.keras.optimizers.Adam

    # compile model
    model.compile(optimizer_E=optimizer(learning_rate=input_args.lr),
                  optimizer_sigma=optimizer(learning_rate=input_args.lr_sigma),
                  loss_E=loss_E, loss_sigma=loss_sigma,
                  metrics=input_args.metrics,
                  weighted_metrics=input_args.weighted_metrics)

    '''
    Logging and Callbacks creation
    '''
    # convert objects to str in order to make log
    input_args.E_loss = str(loss_E)
    input_args.sigma_loss = str(loss_sigma)
    input_args.optimizer = str(optimizer)
    input_args.metrics = str(input_args.metrics)
    input_args.weighted_metrics = str(input_args.weighted_metrics)
    input_args.lr = str(lr_initial) + " " + str(input_args.lr)
    input_args.lr_sigma = str(lr_sigma_initial) + " " + str(input_args.lr)
    # save log with all the setup
    with open(f'{path_to_save}/compile_and_train_config.yaml', 'w') as fp:
        yaml.dump(asdict(input_args), fp)

    # Define the Keras TensorBoard callbacks.
    experiment_name = path_to_save[-path_to_save[::-1].index('/'):]
    logdir = f"./logs_tb/{experiment_name}/fit/{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    tensorboard_callback = cb.TensorBoard(log_dir=logdir)
    callbacks = [cb.EarlyStopping(**input_args.early_stop_params),
                 cb.ModelCheckpoint(filepath=f"{path_to_save}/best_by_test",
                                    save_freq='epoch',
                                    monitor='val_loss_E', verbose=input_args.verbose,
                                    save_best_only=True, mode='min'),
                 cb.ModelCheckpoint(filepath=f"{path_to_save}/best_by_train",
                                    save_freq=int(steps_per_epoch // input_args.checkpoint_cutting),
                                    save_best_only=True,
                                    monitor='loss_E', mode='min',
                                    verbose=input_args.verbose),
                 tensorboard_callback,
                 VerboseSave(f'{path_to_save}/verbose.txt',
                             freq=int(steps_per_epoch // input_args.verbose_cutting)),
                 cb.TerminateOnNaN(),
                 cb.CSVLogger(f"{path_to_save}/history.csv", separator=',')
                 ]

    '''
    Train the model!
    '''
    history = model.fit(train_dataset, steps_per_epoch=steps_per_epoch, epochs=input_args.num_of_epochs,
                        validation_data=test_dataset,
                        callbacks=callbacks, verbose=input_args.verbose)
    # Save the very last model instance
    model.save(f'{path_to_save}/last')

    return history, test_dataset #delete test_dataset!
