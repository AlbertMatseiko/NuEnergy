import tensorflow as tf
from dataclasses import asdict
from pydantic.dataclasses import dataclass
from typing import Any, List, Union

tfl = tf.keras.layers
SEED = 42
GU = tf.keras.initializers.GlorotUniform(seed=SEED)
Ort = tf.keras.initializers.Orthogonal(seed=SEED)


"""Custom activation"""
def shifted_relu(x, t=2., a=0.5):
    return tf.keras.activations.relu(a*x+t)
ShiftedRelu = tf.keras.layers.Activation(shifted_relu)


"""RNN Layers!"""
@dataclass
class RnnInput:
    units: int = 32
    return_sequences: bool = False
    activation: str = 'tanh'
    recurrent_activation: str = 'sigmoid'
    dropout: float = 0.1
    recurrent_dropout: float = 0.1
    kernel_initializer: Any = GU
    recurrent_initializer: Any = Ort
    merge_mode: str = 'mul'


# BidirLSTM layer
class BidirLayer(tfl.Layer):
    def __init__(self, input_hp: RnnInput = RnnInput(), **kwargs):
        super(BidirLayer, self).__init__(**kwargs)
        self.input_hp = input_hp
        self.lstm_layer = tfl.LSTM(**({k: v for k, v in asdict(self.input_hp).items()
                                       if k not in ["merge_mode"]}))
        self.bidir = tfl.Bidirectional(self.lstm_layer,
                                       merge_mode=self.input_hp.merge_mode)

    def get_config(self):
        config = super().get_config()
        config.update(asdict(self.input_hp))
        return config

    def call(self, x, seq_mask=None):
        if seq_mask is not None:
            return self.bidir(x, mask=tf.cast(seq_mask, tf.bool)[:, :, 0]), seq_mask
        else:
            return self.bidir(x)

"""Last Dense Layers in Encoder part!"""
@dataclass
class DenseInput:
    units: int = 2
    activation: Any = (tfl.LeakyReLU())
    dropout: float = 0.1
    kernel_initializer: Any = GU

class DenseBlock(tfl.Layer):
    def __init__(self, input_hp: DenseInput = DenseInput(), **kwargs):
        self.input_hp = input_hp
        if self.input_hp.activation == 'shifted_relu':
            self.input_hp.activation = ShiftedRelu
        inp_dict = asdict(self.input_hp)
        dropout = inp_dict.pop('dropout')
        self.dense_layer = tfl.Dense(**inp_dict)
        self.dropout_layer = tfl.Dropout(dropout)
        super(DenseBlock, self).__init__(**kwargs)

    def get_config(self):
        config = super().get_config()
        config.update(asdict(self.input_hp))
        return config

    def call(self, x, *args, **kwargs):
        x_out = self.dropout_layer(x)
        x_out = self.dense_layer(x_out)
        return x_out


"""Building a Universal ENCODER"""
@dataclass
class EncoderBlockInput:
    rnn_start_inputs: List[RnnInput] = None
    rnn_end_inputs: List[RnnInput] = None
    dense_blocks: List[DenseInput] = None

class EncoderBlock(tfl.Layer):
    def __init__(self, input_hp: EncoderBlockInput = EncoderBlockInput(), **kwargs):
        super(EncoderBlock, self).__init__(**kwargs)
        self.input_hp = input_hp

        # will contain tuples: (layer, batch_norm)
        self.start_layers_list = []
        self.final_layers_list = []

        if self.input_hp.rnn_start_inputs is not None:
            for rnn_inp in self.input_hp.rnn_start_inputs:
                assert rnn_inp.return_sequences
                self.start_layers_list.append((BidirLayer(rnn_inp), tfl.BatchNormalization(axis=-1)))

        if self.input_hp.rnn_end_inputs is not None:
            for rnn_inp in self.input_hp.rnn_end_inputs[:-1]:
                assert rnn_inp.return_sequences
                self.start_layers_list.append((BidirLayer(rnn_inp), tfl.BatchNormalization(axis=-1)))
        
        if self.input_hp.rnn_end_inputs is not None:
            for rnn_inp in self.input_hp.rnn_end_inputs[-1:]:
                assert not rnn_inp.return_sequences
                self.final_layers_list.append((BidirLayer(rnn_inp), tfl.BatchNormalization(axis=-1)))
        
        if self.input_hp.dense_blocks is not None:
            for dense_inp in self.input_hp.dense_blocks:
                self.final_layers_list.append((DenseBlock(dense_inp), tfl.BatchNormalization(axis=-1)))
                assert dense_inp.units > 1

    def get_config(self):
        config = super().get_config()
        config.update(asdict(self.input_hp))
        return config

    def call(self, x_input, *args, **kwargs):
        mask = x_input[:, :, -1:]
        x = x_input[:, :, :-1]

        # apply layers that consider mask
        for layer, batch_norm in self.start_layers_list:
            x, mask = layer(x, mask)
            if batch_norm is not None:
                x = batch_norm(x)

        # may apply final layers to encoded array
        for layer, batch_norm in self.final_layers_list:
            x = layer(x)
            if batch_norm is not None:
                x = batch_norm(x)
        return x


"""Dense Layers for Energy and Sigma Regression"""
@dataclass
class DenseRegressionInput:
    dense_blocks: List[DenseInput]

class DenseRegression(tfl.Layer):
    def __init__(self, input_hp: DenseRegressionInput, **kwargs):
        super(DenseRegression, self).__init__(**kwargs)
        self.input_hp = input_hp
        self.layers_list = []
        self.batch_norm_list = []

        for i, dense_inp in enumerate(self.input_hp.dense_blocks):
            self.layers_list.append(DenseBlock(dense_inp))
            if i < len(self.input_hp.dense_blocks) - 1:
                self.batch_norm_list.append(tfl.BatchNormalization(axis=-1))
            else:
                #assert dense_inp.dropout == 0.  # check that last layer has no dropout
                assert dense_inp.units == 1  # check that last layer is regression
                self.batch_norm_list.append(None)

    def get_config(self):
        config = super().get_config()
        config.update(asdict(self.input_hp))
        return config

    def call(self, x, *args, **kwargs):

        # apply layers
        for layer, bath_norm in zip(self.layers_list, self.batch_norm_list):
            x = layer(x)
            if bath_norm is not None:
                x = bath_norm(x)

        return x


@dataclass
class TwoTapesModelInput:
    encoder_inp: EncoderBlockInput
    energy_inp: DenseRegressionInput
    sigma_inp: Union[DenseRegressionInput, None]

class TwoTapesModel(tf.keras.Model):

    def __init__(self, input_hp: TwoTapesModelInput, **kwargs):
        super(TwoTapesModel, self).__init__(**kwargs)
        self.input_hp = input_hp

        # init the encoder layer
        self.encoder = EncoderBlock(self.input_hp.encoder_inp)


        # init the Energy branch
        self.energy = DenseRegression(self.input_hp.energy_inp)

        # init the Sigma branch if necessary
        if self.input_hp.sigma_inp is not None:
            self.sigma = DenseRegression(self.input_hp.sigma_inp)
        else:
            self.sigma = None

        self.loss_E_tracker = tf.keras.metrics.Mean(name="loss_E")
        self.loss_sigma_tracker = tf.keras.metrics.Mean(name="loss_sigma")

    def get_config(self):
        config = super().get_config()
        config.update(asdict(self.input_hp))
        return config

    def call(self, inputs):
        x = self.encoder(inputs)
        x_E = tf.identity(x)
        E_out = self.energy(x_E)

        if self.sigma is not None:
            x_sigma = tf.identity(x)
            sigma_out = self.sigma(x_sigma, E_out)
            outputs = tfl.Concatenate(axis=-1,
                                      name=f'OutputsConcat')([E_out, sigma_out])
        else:
            outputs = E_out

        return outputs

    def compile(self, optimizer_E, optimizer_sigma, loss_E, loss_sigma=None,
                metrics=None, weighted_metrics=None, **kwargs):  # optimizer, loss): #
        super().compile()

        if metrics is None:
            metrics = []
        if weighted_metrics is None:
            weighted_metrics = []
        self.metrics_given = metrics
        self.weighted_metrics_given = weighted_metrics

        # make optimizer for Energy branch and corresponding weights
        self.loss_E = loss_E
        self.optimizer_E = optimizer_E
        self.train_vars_energy = self.encoder.trainable_weights + self.energy.trainable_weights

        # make optimizer for Sigma branch and corresponding weights (if necessary)
        if self.sigma is not None:
            self.loss_sigma = loss_sigma
            self.optimizer_sigma = optimizer_sigma
            self.train_vars_sigma = self.sigma.trainable_weights

    def train_step(self, data):
        inputs, target, weights = data
        assert weights.shape[0] == inputs.shape[0]
        assert weights.shape[1] == 1
        if self.sigma is not None:
            with tf.GradientTape(watch_accessed_variables=False) as tape_energy:
                tape_energy.watch(self.train_vars_energy)
                with tf.GradientTape(watch_accessed_variables=False) as tape_sigma:
                    tape_sigma.watch(self.train_vars_sigma)
                    preds = self.__call__(inputs)
                    loss_E_value = self.loss_E(target, preds, sample_weight=weights)
                    self.loss_E_tracker.update_state(loss_E_value)
                    if self.loss_sigma is not None:
                        loss_sigma_value = self.loss_sigma(target, preds, sample_weight=weights)
                        self.loss_sigma_tracker.update_state(loss_sigma_value)

            grads_E = tape_energy.gradient(loss_E_value, self.train_vars_energy)
            self.optimizer_E.apply_gradients(zip(grads_E, self.train_vars_energy))

            if self.loss_sigma is not None:
                grads_sigma = tape_sigma.gradient(loss_sigma_value, self.train_vars_sigma)
            else:
                grads_sigma = tape_sigma.gradient(loss_E_value, self.train_vars_sigma)
            self.optimizer_sigma.apply_gradients(zip(grads_sigma, self.train_vars_sigma))

        else:
            with tf.GradientTape(watch_accessed_variables=False) as tape_energy:
                tape_energy.watch(self.train_vars_energy)
                preds = self.__call__(inputs)
                loss_E_value = self.loss_E(target, preds, sample_weight=weights)
                grads_E = tape_energy.gradient(loss_E_value, self.train_vars_energy)
                self.optimizer_E.apply_gradients(zip(grads_E, self.train_vars_energy))
                self.loss_E_tracker.update_state(loss_E_value)

        to_return = {self.loss_E_tracker.name: self.loss_E_tracker.result()}
        if (self.sigma is not None) and (self.loss_sigma is not None):
            to_return[self.loss_sigma_tracker.name] = self.loss_sigma_tracker.result()
        for m in self.metrics_given:
            m.update_state(target, preds)
            to_return[m.name] = m.result()
        for m in self.weighted_metrics_given:
            m.update_state(target, preds, sample_weight=weights)
            to_return[m.name] = m.result()
        return to_return

    @property
    def metrics(self):
        """
         We list our `Metric` objects here so that `reset_states()` can be
         called automatically at the start of each epoch
         or at the start of `evaluate()`.
         If you don't implement this property, you have to call
         `reset_states()` yourself at the time of your choosing.
         """
        if (self.sigma is not None) and (self.loss_sigma is not None):
            list_of_metrics = [self.loss_E_tracker, self.loss_sigma_tracker] + self.metrics_given + self.weighted_metrics_given
        else:
            list_of_metrics = [self.loss_E_tracker] + self.metrics_given + self.weighted_metrics_given
        return list_of_metrics

    def test_step(self, data):
        # Unpack the data
        inputs, target, weights = data
        # Compute predictions
        preds = self.__call__(inputs, training=False)

        loss_E_value = self.loss_E(target, preds, sample_weight=weights)
        self.loss_E_tracker.update_state(loss_E_value)
        to_return = {self.loss_E_tracker.name: self.loss_E_tracker.result()}
        if (self.sigma is not None) and (self.loss_sigma is not None):
            loss_sigma_value = self.loss_sigma(target, preds, sample_weight=weights)
            self.loss_sigma_tracker.update_state(loss_sigma_value)
            to_return[self.loss_sigma_tracker.name] = self.loss_sigma_tracker.result()

        for m in self.metrics_given:
            m.update_state(target, preds, sample_weight=weights)
            to_return[m.name] = m.result()
        for m in self.weighted_metrics_given:
            m.update_state(target, preds, sample_weight=weights)
            to_return[m.name] = m.result()
        return to_return
    
    
"""Initialize the best architecture"""
    
encoder_inp = EncoderBlockInput(
    rnn_start_inputs = [
            # 1
            {'units': 64,
            'return_sequences': True,
            'dropout': 0.2,
            'recurrent_dropout': 0.}
        ],
    rnn_end_inputs = [
            # 1
            {'units': 64,
            'return_sequences': False,
            'dropout': 0.2,
            'recurrent_dropout': 0.}
        ]
    )
energy_inp = DenseRegressionInput(dense_blocks=[
    {'units': 128,
    'dropout': 0.2},
    {'units': 64,
    'dropout': 0.2},
    {'units': 1,
    'activation': 'linear',
    'dropout': 0.1}
    ]
)                                
sigma_inp = DenseRegressionInput(dense_blocks=[
    # 1
    {'units': 128,
    'dropout': 0.2},
    # 2
    {'units': 64,
    'dropout': 0.2},
    # 3
    {'units': 1,
    'activation': 'shifted_relu',
    'dropout': 0.1}
    ]
)    
model_inp = TwoTapesModelInput(encoder_inp=encoder_inp, energy_inp=energy_inp, sigma_inp=sigma_inp)
model = TwoTapesModel(model_inp)
model.build((None, None, 6))

""" After the model is trained on MC data, predictions for data (exp_data) can be obtained as follows """
preds = model.predict(exp_data)

if __name__=="__main__":
    print(model.summary())
