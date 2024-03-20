import sys

path_to_project = '/NuEnergy'
sys.path.append(path_to_project)

import numpy as np
import tensorflow as tf
from dataclasses import asdict, field
from pydantic.dataclasses import dataclass
from typing import Any, List

try:
    from .activations import ShiftedRelu
except:
    from activations import ShiftedRelu

tfl = tf.keras.layers
SEED = 42
GU = tf.keras.initializers.GlorotUniform(seed=SEED)
Ort = tf.keras.initializers.Orthogonal(seed=SEED)


class GlobalAveragePooling1DMasked(tfl.GlobalAveragePooling1D):
    def call(self, x, mask=None):
        if mask is not None:
            mask = tf.cast(mask, tf.float32)
            return tf.reduce_sum(x * mask, axis=1) / tf.reduce_sum(mask, axis=1)
        else:
            return super().call(x)


class GlobalMaxPooling1DMasked(tfl.GlobalMaxPooling1D):
    def call(self, x, mask=None):
        if mask is not None:
            x = tf.where(tf.cast(mask, tf.bool), x, -np.inf)
            return tf.reduce_max(x, axis=1)
        else:
            return super().call(x)


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


@dataclass
class ConvInput:
    filters: int = 10
    kernel_size: int = 3
    strides: int = 1
    activation: Any = tfl.LeakyReLU(alpha=0.3)
    dropout: float = 0.1
    kernel_initializer: Any = GU
    padding: str = "valid"


# 1D Convolution with mask
class MaskedConv1D(tfl.Layer):
    def __init__(self, input_hp: ConvInput = ConvInput(), **kwargs):
        self.input_hp = input_hp
        self.conv = tfl.Conv1D(**({k: v for k, v in asdict(self.input_hp).items() if k not in ["dropout"]}))
        super(MaskedConv1D, self).__init__(**kwargs)

    def get_config(self):
        config = super().get_config()
        config.update(asdict(self.input_hp))
        return config

    def call(self, x, mask):
        strides = self.input_hp.strides
        k = self.input_hp.kernel_size
        x_local = x * tf.cast(mask, tf.float32)
        # Make my own 'same' padding. TF returns bullshit.
        pad_left = (k - 1) // 2
        pad_right = (k - 1) // 2 + (k - 1) % 2
        x_local = tf.pad(x_local, ((0, 0), (pad_left, pad_right), (0, 0)))
        x_local = self.conv(x_local)
        x_local = tfl.Dropout(self.input_hp.dropout)(x_local)
        # Recalculate the mask so 'True' values corresponds to positions of hits as if there wasn't masking in conv.
        signal_sizes = tf.cast(tf.reduce_sum(mask, axis=1), tf.int32)
        new_sizes = (signal_sizes - 1) // strides + 1
        mask_new = tf.transpose(tf.sequence_mask(new_sizes,
                                                 maxlen=tf.shape(x_local)[1],
                                                 dtype=tf.float32
                                                 ),
                                [0, 2, 1])
        x_local = x_local * mask_new
        return x_local, mask_new


@dataclass
class ResBlockInput:
    #default = ConvInput(strides=1, custom_name="ConvID")
    id: ConvInput
    
    #default = ConvInput(strides=2, custom_name="ConvCD")
    cd: ConvInput
    
    #default = ConvInput(strides=2, custom_name="ConvSKIP")
    skip: ConvInput


class ResBlock(tfl.Layer):
    def __init__(self, input_hp: ResBlockInput, **kwargs): # = ResBlockInput()
        super(ResBlock, self).__init__(**kwargs)

        self.input_hp = input_hp

        self.conv_id = MaskedConv1D(self.input_hp.id)
        self.conv_cd = MaskedConv1D(self.input_hp.cd)
        self.conv_skip = MaskedConv1D(self.input_hp.skip)

        self.norm_id = tfl.BatchNormalization(axis=-1)
        self.norm_cd = tfl.BatchNormalization(axis=-1)
        self.norm_skip = tfl.BatchNormalization(axis=-1)

    def get_config(self):
        config = super().get_config()
        config.update(asdict(self.input_hp))
        return config

    def call(self, x, mask):
        x_skip = x
        mask_skip = mask

        ### forward convs
        # identical dimensions
        x, mask = self.conv_id(x, mask)
        x = self.norm_id(x)

        # change dimensions
        x, mask = self.conv_cd(x, mask)
        x = self.norm_cd(x)

        ### skip conv
        x_skip, mask_skip = self.conv_skip(x_skip, mask_skip)
        x_skip = self.norm_skip(x_skip)

        ### Concat
        # just to be sure
        length = tf.minimum(tf.shape(x_skip)[1], tf.shape(x)[1])
        x = tfl.Concatenate(axis=-1,
                            name=f'ResNetConcat')([x[:, :length, :], x_skip[:, :length, :]])
        mask, mask_skip = mask[:, :length, :], mask_skip[:, :length, :]
        return x, mask


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
        x_out = self.dense_layer(x)
        x_out = self.dropout_layer(x_out)
        return x_out


@dataclass
class EncoderBlockInput:
    rnn_start_inputs: List[RnnInput]
    res_block_inputs: List[ResBlockInput]
    rnn_end_inputs: List[RnnInput]
    pooling: bool
    dense_blocks: List[DenseInput]


class EncoderBlock(tfl.Layer):
    def __init__(self, input_hp: EncoderBlockInput, **kwargs):
        super(EncoderBlock, self).__init__(**kwargs)
        self.input_hp = input_hp

        # will contain tuples: (layer, batch_norm)
        self.start_layers_list = []
        self.final_layers_list = []

        for rnn_inp in self.input_hp.rnn_start_inputs:
            assert rnn_inp.return_sequences
            self.start_layers_list.append((BidirLayer(rnn_inp), tfl.BatchNormalization(axis=-1)))

        for res_block_inp in self.input_hp.res_block_inputs:
            self.start_layers_list.append((ResBlock(res_block_inp), None))  # batch norm is already in resblocks

        for rnn_inp in self.input_hp.rnn_end_inputs[:-1]:
            assert rnn_inp.return_sequences
            self.start_layers_list.append((BidirLayer(rnn_inp), tfl.BatchNormalization(axis=-1)))

        # either pooling or rnn to encode. Not both!
        assert (self.input_hp.pooling + (len(self.input_hp.rnn_end_inputs) > 0)) == 1
        for rnn_inp in self.input_hp.rnn_end_inputs[-1:]:
            assert not rnn_inp.return_sequences
            self.final_layers_list.append((BidirLayer(rnn_inp), tfl.BatchNormalization(axis=-1)))
        if self.input_hp.pooling:
            self.final_layers_list.append((GlobalAveragePooling1DMasked(), tfl.BatchNormalization(axis=-1)))

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
                assert dense_inp.dropout == 0.  # check that last layer has no dropout
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


if __name__ == "__main__":
    inp = ConvInput()
    print(tfl.Conv1D(**{k: v for k, v in asdict(inp).items() if k not in ["dropout"]}))
