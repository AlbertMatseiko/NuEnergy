import tensorflow as tf

## input shape should always be (some batch dims, nodes, features)

## NLP-like

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

    def __init__(self, num_heads, qk_dim, v_dim, out_dim, dr_rate):
        super().__init__()
        self.qk_dim = qk_dim
        self.v_dim = v_dim
        self.prog_layer = qkv_projector(num_heads*qk_dim, num_heads*v_dim)
        self.num_heads = num_heads
        self.out_dim = out_dim       
        self.att_layer = NLPAttention()
        self.dropout = tf.keras.layers.Dropout( dr_rate )

    def build(self, input_shape):
        qk_shape = input_shape[:-1]+(self.qk_dim,)
        v_shape =  input_shape[:-1]+(self.v_dim,)
        self.att_layer.build((qk_shape,qk_shape,v_shape))
        self.matrix_out = tf.Variable( initial_value=tf.keras.initializers.GlorotUniform()( shape=(self.num_heads*self.v_dim, self.out_dim ) ), trainable=True )
        self.prog_layer.build(input_shape)
        self.bias = tf.Variable( initial_value=tf.keras.initializers.GlorotUniform()( shape=(self.out_dim,) ), trainable=True )
        
    def call(self, nodes, training=False):
        x = self.dropout(nodes)
        (qs,ks,vs) = self.prog_layer(x)
        qs = tf.stack( tf.split(qs, self.num_heads, axis=-1), axis=0 )
        ks = tf.stack( tf.split(ks, self.num_heads, axis=-1), axis=0 )
        vs = tf.stack( tf.split(vs, self.num_heads, axis=-1), axis=0 )
        msgs = self.att_layer((qs,ks,vs))
        msgs = tf.concat( tf.unstack(msgs, axis=0), axis=-1 )
        res = tf.linalg.matmul(msgs,self.matrix_out) + self.bias
        return res

## scalar product

class scalarProductAttentiont(tf.keras.layers.Layer):

    def __init__(self):
        super().__init__()
        
    def build(self, input_shape):
        num_fs = input_shape[-1]
        self.product_matrix = tf.Variable( initial_value=tf.keras.initializers.GlorotUniform()( shape=(num_fs,num_fs) ), trainable=True )
        self.norm_const = 1./tf.math.sqrt( tf.cast( num_fs, tf.float32 ) )

    def call(self, nodes, training=False):
        x = tf.linalg.matmul( nodes, self.product_matrix )
        prods = tf.linalg.matmul( x, nodes, transpose_b=True )
        att_scores = tf.nn.softmax( prods*self.norm_const )
        messgs = tf.linalg.matmul( att_scores, nodes )
        return messgs

class multiheadAttentionDirect(tf.keras.layers.Layer):

    def __init__(self, num_heads, out_dim, dr_rate):
        super().__init__()
        self.num_heads = num_heads
        self.out_dim = out_dim     
        self.att_layers = [ scalarProductAttentiont() for i in range(num_heads) ]
        self.dropout = tf.keras.layers.Dropout( dr_rate )

    def build(self, input_shape):
        self.matrix_out = tf.Variable( initial_value=tf.keras.initializers.GlorotUniform()( shape=(self.num_heads*input_shape[-1], self.out_dim ) ), trainable=True )
        self.bias = tf.Variable( initial_value=tf.keras.initializers.GlorotUniform()( shape=(self.out_dim,) ), trainable=True )
        for layer in self.att_layers:
            layer.build(input_shape)
    
    def call(self, nodes, training=False):
        x = self.dropout(nodes)
        msgs = tf.concat( [ layer(x) for layer in self.att_layers ], axis=-1 )
        res = tf.linalg.matmul(msgs,self.matrix_out) + self.bias
        return res

## graph-like

class GraphAttention(tf.keras.layers.Layer):

    def __init__(self, use_delta, hidden_dim):
        super().__init__()
        self.use_delta = use_delta
        self.a = tf.Variable( tf.keras.initializers.Ones()( shape=(hidden_dim,) ), trainable=True )
        self.att_layer = tf.keras.layers.Dense( hidden_dim )
        self.activation = tf.keras.layers.LeakyReLU( alpha=0.2 )

    def build(self, input_shape):
        if self.use_delta:
            dense_shape = input_shape[:-1]+(3*input_shape[-1],)
        else:
            dense_shape = input_shape[:-1]+(2*input_shape[-1],)
        self.att_layer.build(dense_shape)
        self.norm_const = 1./tf.math.sqrt( tf.cast(input_shape[-1], tf.float32) )

    def eval_att(self, nodes, training=False):
        scores = tf.tensordot( self.a, self.activation( self.att_layer(nodes) ), axes=[[-1],[-1]] )
        att_scores = tf.nn.softmax( scores*self.norm_const )
        return att_scores

    def call(self, nodes, training=False):
        num = tf.shape(nodes)[-2]
        n1 = tf.expand_dims(nodes, axis=-2)
        n1 = tf.repeat(n1, axis=2, repeats=num)
        n2 = tf.transpose(n1, perm=[0,2,1,3])
        nns = tf.concat( (n1,n2,n1-n2), axis=-1)
        scores = self.eval_att(nns)
        messgs = tf.linalg.matmul(scores, nodes)
        return messgs

# slower but better suited for graph nns
class GraphAttention_gather(tf.keras.layers.Layer):

    def __init__(self, use_delta, hidden_dim):
        super().__init__()
        self.use_delta = use_delta
        self.a = tf.Variable( tf.keras.initializers.Ones()( shape=(hidden_dim,) ), trainable=True )
        self.att_layer = tf.keras.layers.Dense( hidden_dim )
        self.activation = tf.keras.layers.LeakyReLU( alpha=0.2 )

    def build(self, input_shape):
        data_shape = input_shape[0]
        if self.use_delta:
            dense_shape = data_shape[:-1]+(3*data_shape[-1],)
        else:
            dense_shape = data_shape[:-1]+(2*data_shape[-1],)
        self.att_layer.build(dense_shape)
        self.norm_const = 1./tf.math.sqrt( tf.cast(data_shape[-1], tf.float32) )

    def eval_att(self, nodes, training=False):
        scores = tf.tensordot( self.a, self.activation( self.att_layer(nodes) ), axes=[[-1],[-1]] )
        att_scores = tf.nn.softmax( scores*self.norm_const )
        return att_scores

    def call(self, inputs, training=False):
        nodes, adjs = inputs
        ins = tf.transpose( tf.gather( nodes, adjs, axis=1 ), perm=[0,1,2,4,3] )
        delta = ins[...,0:1] - ins[...,1:2]
        ins = tf.concat( [ins,delta], axis=-1 )
        ins = tf.concat( tf.unstack( ins, axis=-1 ), axis=-1 )
        scores = self.eval_att(ins)
        messgs = tf.linalg.matmul(scores, nodes)
        return messgs
        
class GraphAttentionQKV(tf.keras.layers.Layer):

    def __init__(self, use_delta, hidden_dim):
        super().__init__()
        self.use_delta = use_delta
        self.a = tf.Variable( tf.keras.initializers.Ones()( shape=(hidden_dim,) ), trainable=True )
        self.att_layer = tf.keras.layers.Dense( hidden_dim )
        self.activation = tf.keras.layers.LeakyReLU( alpha=0.2 )

    def build(self, input_shape):
        shape = input_shape[0]
        if self.use_delta:
            dense_shape = shape[:-1]+(3*shape[-1],)
        else:
            dense_shape = shape[:-1]+(2*shape[-1],)
        self.att_layer.build(dense_shape)
        self.norm_const = 1./tf.math.sqrt( tf.cast(shape[-1], tf.float32) )

    def eval_att(self, nodes, training=False):
        scores = tf.tensordot( self.a, self.activation( self.att_layer(nodes) ), axes=[[-1],[-1]] )
        att_scores = tf.nn.softmax( scores*self.norm_const )
        return att_scores

    def call(self, inputs, training=False):
        (qs,ks,vs) = inputs
        num = tf.shape(qs)[-2]
        qs = tf.expand_dims(qs, axis=-2)
        ks = tf.expand_dims(ks, axis=1)
        qs = tf.repeat(qs, axis=2, repeats=num)
        ks = tf.repeat(ks, axis=1, repeats=num)
        nns = tf.concat( (qs,ks,qs-ks), axis=-1)
        scores = self.eval_att(nns)
        messgs = tf.linalg.matmul(scores, vs)
        return messgs

# multihead
class multiheadAttentionGraph(tf.keras.layers.Layer):

    def __init__(self, num_heads, use_delta, hidden_dim, out_dim, dr_rate):
        super().__init__()
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim 
        self.att_layers = [ GraphAttention(use_delta, hidden_dim) for i in range(num_heads) ]
        self.dropout = tf.keras.layers.Dropout( dr_rate )

    def build(self, input_shape):
        self.matrix_out = tf.Variable( initial_value=tf.keras.initializers.GlorotUniform()( shape=(self.num_heads*input_shape[-1], self.out_dim ) ), trainable=True )
        self.bias = tf.Variable( initial_value=tf.keras.initializers.GlorotUniform()( shape=(self.out_dim, ) ), trainable=True )
        for layer in self.att_layers:
            layer.build(input_shape)
            
    def call(self, nodes, training=False):
        x = self.dropout(nodes)
        msgs = tf.concat( [ layer(x) for layer in self.att_layers ], axis=-1 )
        res = tf.linalg.matmul(msgs,self.matrix_out) + self.bias
        return res

class multiheadAttentionGraphQKV(tf.keras.layers.Layer):

    def __init__(self, num_heads, use_delta, hidden_dim, out_dim, qk_dim, v_dim, dr_rate):
        super().__init__()
        self.qk_dim = qk_dim
        self.v_dim = v_dim
        self.prog_layer = qkv_projector(num_heads*qk_dim, num_heads*v_dim)
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim    
        self.att_layers = [ GraphAttentionQKV(use_delta, hidden_dim) for i in range(num_heads) ]
        self.dropout = tf.keras.layers.Dropout( dr_rate )

    def build(self, input_shape):
        self.matrix_out = tf.Variable( initial_value=tf.keras.initializers.GlorotUniform()( shape=(self.num_heads*self.hidden_dim, self.out_dim ) ), trainable=True )
        self.bias = tf.Variable( initial_value=tf.keras.initializers.GlorotUniform()( shape=(self.out_dim, ) ), trainable=True )
        qk_shape = input_shape[:-1]+(self.qk_dim,)
        v_shape =  input_shape[:-1]+(self.v_dim,)
        for layer in self.att_layers:
            layer.build((qk_shape,qk_shape,v_shape))
            
    def call(self, nodes, training=False):
        x = self.dropout(nodes)
        qs, ks, vs = self.prog_layer(x)
        qs = tf.stack( tf.split(qs, self.num_heads, axis=-1), axis=0 )
        ks = tf.stack( tf.split(ks, self.num_heads, axis=-1), axis=0 )
        vs = tf.stack( tf.split(vs, self.num_heads, axis=-1), axis=0 )
        msgs = tf.concat( [ self.att_layers[i]((qs[i],ks[i],vs[i])) for i in range(self.num_heads) ], axis=-1 )
        res = tf.linalg.matmul(msgs,self.matrix_out) + self.bias
        return res