import sys

path_to_project = '/NuEnergy'
sys.path.append(path_to_project)

import numpy as np
import tensorflow as tf
from dataclasses import asdict, field
from pydantic.dataclasses import dataclass
from pydantic import Field
from typing import Any, List

try:
    from .activations import ShiftedRelu
except:
    from activations import ShiftedRelu

tfl = tf.keras.layers
SEED = 42
GU = tf.keras.initializers.GlorotUniform(seed=SEED)
Ort = tf.keras.initializers.Orthogonal(seed=SEED)


"""Pooling layers!"""
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


"""CNN Layers!"""
@dataclass
class ConvInput:
    filters: int = 10
    kernel_size: int = 3
    strides: int = 1
    activation: Any = tfl.LeakyReLU(alpha=0.3)
    dropout: float = 0.1
    kernel_initializer: Any = GU


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
        ## just to be sure
        length = tf.minimum(tf.shape(x_skip)[1], tf.shape(x)[1])
        
        x = tfl.Concatenate(axis=-1,
                            name=f'ResNetConcat')([x[:, :length, :], x_skip[:, :length, :]])

        mask, mask_skip = mask[:, :length, :], mask_skip[:, :length, :]
        return x, mask


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


"""Multihead Attention Layers!"""
# projecting to Qs, Ks, Vs
class qkv_projector(tf.keras.layers.Layer):

    def __init__(self, qk_dim, v_dim):
        super().__init__()
        self.qk_dim = qk_dim
        self.v_dim = v_dim

    def build(self, input_shape):
        num_fs = input_shape[-1]
        self.proj_matrix_Q = tf.Variable( initial_value=tf.keras.initializers.GlorotUniform()( shape=(num_fs,self.qk_dim) ), trainable=True )
        self.proj_matrix_K = tf.Variable( initial_value=tf.keras.initializers.GlorotUniform()( shape=(num_fs,self.qk_dim) ), trainable=True )
        self.proj_matrix_V = tf.Variable( initial_value=tf.keras.initializers.GlorotUniform()( shape=(num_fs,self.v_dim) ), trainable=True )
        self.bias_Q = tf.Variable( initial_value=tf.keras.initializers.GlorotUniform()( shape=(self.qk_dim,) ), trainable=True )
        self.bias_K = tf.Variable( initial_value=tf.keras.initializers.GlorotUniform()( shape=(self.qk_dim,) ), trainable=True )
        self.bias_V = tf.Variable( initial_value=tf.keras.initializers.GlorotUniform()( shape=(self.v_dim,) ), trainable=True )

    def call(self, node, training=False):
        qs = tf.linalg.matmul( node, self.proj_matrix_Q ) + self.bias_Q
        ks = tf.linalg.matmul( node, self.proj_matrix_K ) + self.bias_K
        vs = tf.linalg.matmul( node, self.proj_matrix_V ) + self.bias_V
        return (qs,ks,vs)

# attention calculation
class NLPAttention(tf.keras.layers.Layer):

    def __init__(self):
        super().__init__()

    def build(self, input_shape):
        self.norm_softmax = 1./tf.math.sqrt( tf.cast( input_shape[1][-1], tf.float32 ) )

    def call(self, inputs, training=False):
        qs, ks, vs = inputs
        prods = tf.linalg.matmul( qs, ks, transpose_b=True )
        att_scores = tf.nn.softmax( prods*self.norm_softmax )
        messgs = tf.linalg.matmul( att_scores, vs )
        return messgs

# multihead
class multiheadAttentionNLP(tf.keras.layers.Layer):

    def __init__(self, num_heads, qk_dim, v_dim, out_dim, dr_rate):#num_heads, qk_dim, v_dim, out_dim, dr_rate):
        super().__init__()
        self.qk_dim = qk_dim
        self.v_dim = v_dim
        self.num_heads = num_heads
        self.out_dim = out_dim    
        self.dr_rate = dr_rate
        
        self.input_hp = dict(qk_dim=qk_dim, v_dim=v_dim, num_heads=num_heads, out_dim=out_dim, dr_rate=dr_rate)
        
        self.prog_layer = qkv_projector(self.num_heads*self.qk_dim, self.num_heads*self.v_dim)
        self.att_layer = NLPAttention()
        self.dropout = tf.keras.layers.Dropout( self.dr_rate )
    
    def get_config(self):
        config = super().get_config()
        config.update(asdict(self.input_hp))
        return config
    
    def build(self, input_shape):
        qk_shape = input_shape[:-1]+(self.qk_dim,)
        v_shape =  input_shape[:-1]+(self.v_dim,)
        self.att_layer.build((qk_shape,qk_shape,v_shape))
        self.matrix_out = tf.Variable( initial_value=tf.keras.initializers.GlorotUniform()( shape=(self.num_heads*self.v_dim, self.out_dim ) ), trainable=True )
        self.prog_layer.build(input_shape)
        self.bias = tf.Variable( initial_value=tf.keras.initializers.GlorotUniform()( shape=(self.out_dim,) ), trainable=True )
        
    def call(self, nodes, *args, **kwargs):
        x = self.dropout(nodes)
        (qs,ks,vs) = self.prog_layer(x)
        qs = tf.stack( tf.split(qs, self.num_heads, axis=-1), axis=0 )
        ks = tf.stack( tf.split(ks, self.num_heads, axis=-1), axis=0 )
        vs = tf.stack( tf.split(vs, self.num_heads, axis=-1), axis=0 )
        msgs = self.att_layer((qs,ks,vs))
        msgs = tf.concat( tf.unstack(msgs, axis=0), axis=-1 )
        res = tf.linalg.matmul(msgs,self.matrix_out) + self.bias
        return res

@dataclass
class TransformerEncLayerInput:
    num_heads: int = 8
    qk_dim: int = 64
    v_dim: int = 64
    out_dim: int = 64
    mha_dr_rate: float = 0.1
    
    # Feed Forward params
    ff_units: list[int] = field(default_factory=lambda: [64, 64])
    ff_dr_rate: float = 0.1
    ff_activation: Any = (tfl.LeakyReLU())

class TransformerEncLayer(tf.keras.layers.Layer):
    def __init__(self, input_hp: TransformerEncLayerInput = TransformerEncLayerInput()):#num_heads, qk_dim, v_dim, out_dim, dr_rate):
        super().__init__()
        # MultiHeadAttention layers
        self.input_hp = input_hp
        self.mha_layer = multiheadAttentionNLP(input_hp.num_heads, input_hp.qk_dim, input_hp.v_dim, input_hp.out_dim, 
                                               input_hp.mha_dr_rate)
        #self.add_mha = tf.keras.layers.Add()
        self.norm_mha = tf.keras.layers.BatchNormalization()
        # Feed Forward layers
        self.layers_ff = []
        for un in input_hp.ff_units:
            self.layers_ff.append(DenseBlock(DenseInput(units=un, dropout=input_hp.ff_dr_rate, activation=input_hp.ff_activation)))
        self.norm_ff = tf.keras.layers.BatchNormalization()
        #self.add_ff = tf.keras.layers.Add()
    
    def get_config(self):
        config = super().get_config()
        config.update(asdict(self.input_hp))
        return config
    
    def call(self, data, mask, *args, **kwargs):
        # MHA
        messgs = self.mha_layer(data)
        x = tf.concat([data,messgs], axis=-1)
        mha_output = self.norm_mha(x)
        # FF
        for layer in self.layers_ff:
            x = layer(mha_output)
        x = tf.concat([x,mha_output], axis=-1)
        x = self.norm_ff(x)
        return x*mask, mask


"""Building up a Universal ENCODER"""
@dataclass
class EncoderBlockInput:
    rnn_start_inputs: List[RnnInput] = None
    res_block_inputs: List[ResBlockInput] = None
    transf_inputs: List[TransformerEncLayerInput] = None
    rnn_end_inputs: List[RnnInput] = None
    pooling: bool = False
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

        if self.input_hp.res_block_inputs is not None:
            for res_block_inp in self.input_hp.res_block_inputs:
                self.start_layers_list.append((ResBlock(res_block_inp), None))  # batch norm is already in resblocks

        if self.input_hp.rnn_end_inputs is not None:
            for rnn_inp in self.input_hp.rnn_end_inputs[:-1]:
                assert rnn_inp.return_sequences
                self.start_layers_list.append((BidirLayer(rnn_inp), tfl.BatchNormalization(axis=-1)))

        if self.input_hp.transf_inputs is not None:
            for transf_inp in self.input_hp.transf_inputs:
                self.start_layers_list.append((TransformerEncLayer(transf_inp), None)) # batch norm is already in transf block
        
        # either pooling or rnn to encode. Not both!
        assert (int(self.input_hp.pooling) + int(self.input_hp.rnn_end_inputs is not None)) == 1
        if self.input_hp.rnn_end_inputs is not None:
            for rnn_inp in self.input_hp.rnn_end_inputs[-1:]:
                assert not rnn_inp.return_sequences
                self.final_layers_list.append((BidirLayer(rnn_inp), tfl.BatchNormalization(axis=-1)))
        
        if self.input_hp.pooling:
            self.final_layers_list.append((GlobalAveragePooling1DMasked(), tfl.BatchNormalization(axis=-1)))
        
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


"""Dense Layers for Energy and Sigma Regression!"""
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


if __name__=='__main__':
    inp = EncoderBlockInput(rnn_end_inputs = [RnnInput()])
    enc = EncoderBlock(inp)
    print(enc.get_config())