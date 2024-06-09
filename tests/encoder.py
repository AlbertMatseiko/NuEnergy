import tensorflow as tf

class dense_block(tf.keras.layers.Layer):

    def __init__(self, unit, dr_rate, use_bn):
        super().__init__()
        self.dense = tf.keras.layers.Dense(unit)
        self.dropout = tf.keras.layers.Dropout(dr_rate)
        self.activation = tf.keras.layers.LeakyReLU( alpha=0.2 )
        if use_bn:
            self.norm_layer = tf.keras.layers.BatchNormalization()
        else:
            self.norm_layer = tf.keras.layers.LayerNormalization()
        
    def call(self, inputs, training=False):
        x = self.dropout(inputs, training=training)    
        x = self.dense(x)
        x = self.activation(x)
        x = self.norm_layer(x, training=training)
        return x

# updating layer
class predict_layer(tf.keras.layers.Layer):

    def __init__(self, units, dr_rate, use_bn):
        super().__init__()
        self.layers = []
        for un in units:
            self.layers.append( dense_block(un, dr_rate, use_bn) )
                
    def call(self, inputs, training=False):
        x = inputs
        for layer in self.layers:
            x = layer(x, training=training)
        preds = tf.nn.softmax(x)
        return preds

class update_layer(tf.keras.layers.Layer):

    def __init__(self, units, dr_rate, use_bn):
        super().__init__()
        self.layers = []
        for un in units:
            self.layers.append( dense_block(un, dr_rate, use_bn) )
        self.add = tf.keras.layers.Add()
        if use_bn:
            self.norm_layer = tf.keras.layers.BatchNormalization()
        else:
            self.norm_layer = tf.keras.layers.LayerNormalization()
                
    def call(self, inputs, training=False):
        x = inputs
        for layer in self.layers:
            x = layer(x, training=training)
        # concat
        #updt = tf.concat((x,inputs), axis=-1)
        # add
        updt = self.add([x,inputs])
        updt = self.norm_layer(updt)
        return updt

class EncoderLayer(tf.keras.layers.Layer):
  
    def __init__(self, attention_layer, updater_layer, use_bn):
        super().__init__()
        self.attention_layer = attention_layer
        self.upd_layer = updater_layer
        if use_bn:
            self.layernorm = tf.keras.layers.BatchNormalization()
        else:
            self.layernorm = tf.keras.layers.LayerNormalization()
        self.add = tf.keras.layers.Add()
    
    def call(self, inputs, training=False):
        data, mask = inputs
        messgs = self.attention_layer(data, training=training)
        # var 1: add, remove self.add if not needed
        updts = self.add( [data,messgs] ) 
        # var 2: concat
        #updts = tf.concat( (data,messgs), axis=-1 )
        updts = self.layernorm(updts, training=training)
        updated = self.upd_layer( updts, training=training )*mask
        return updated
    
class Encoder(tf.keras.layers.Layer):
    
    def __init__(self, enc_layers):
        super().__init__()
        self.enc_layers = enc_layers
        self.depth = len(enc_layers)
    
    def call(self, inputs, training=False):
        x, mask = inputs
        for i in range(self.depth):
            x = self.enc_layers[i]((x,mask), training=training)
        return x

# agregating with mask
class avg_agregate_masked(tf.keras.layers.Layer):

    def __init__(self):
        super().__init__()
                
    def call(self, inputs, training=False):
        (data, mask) = inputs
        avg = tf.math.reduce_sum( data, axis=1 ) / tf.math.reduce_sum( mask, axis=1 )
        return avg

class EncoderClassifier(tf.keras.Model):

    def __init__(self, pre_layer, encoder, agregator, predict_layer):
        super().__init__()
        self.pre_layer = pre_layer
        self.encoder = encoder
        self.agregator = agregator
        self.predict_layer = predict_layer

    def call(self, inputs, training=False):
        dt_params, wfs_flat, recos, dt_bundle = inputs
        mask = dt_params[:,:,-1:]
        data = dt_params[:,:,:-1]
        x = self.pre_layer(dt_params, training=training)
        x = self.encoder((x,mask), training=training)
        x = self.agregator(((x,mask)), training=training)
        preds = self.predict_layer(x, training=training)
        return preds
    
# to create:
    
num_heads = [8,8,8,8]
qk_dims = [128,128,128,128]
v_dims = [128,128,128,128]
out_dims = [128,128,128,128]
dr_rate = 0.

att_layers = [ multiheadAttentionNLP(num_heads=nh, qk_dim=qd, v_dim=vd, out_dim=od, dr_rate=dr_rate) for (nh,qd,vd,od) in zip(num_heads,qk_dims,v_dims,out_dims) ]

upd_units = [ [512,128], [512,128], [512,128], [512,128] ]
upd_layers = [ update_layer(units, dr_rate, True) for units in upd_units ]

assert len(upd_layers)==len(att_layers)

enc_layers = [ EncoderLayer(att, upd, True)  for (att,upd) in zip(att_layers,upd_layers) ]

encoder = Encoder(enc_layers)

pre_layer = dense_block(128, 0.0, True)
agregator = avg_agregate_masked()
#agregator = tf.keras.layers.GlobalAveragePooling1D()
predicor = predict_layer([64,2], 0.1, True)

classifier = EncoderClassifier(pre_layer, encoder, agregator, predicor)

# with master node

class EncoderLayer_mn(tf.keras.layers.Layer):
  
    def __init__(self, attention_layer, updater_layer, mn_updater, mn_msg_creator, use_bn):
        super().__init__()
        self.attention_layer = attention_layer
        self.upd_layer = updater_layer
        self.mn_msg_creator = mn_msg_creator
        self.mn_updater = mn_updater
        if use_bn:
            self.layernorm = tf.keras.layers.BatchNormalization()
        else:
            self.layernorm = tf.keras.layers.LayerNormalization()
        self.add = tf.keras.layers.Add()
    
    def call(self, inputs, training=False):
        data, mask, mn = inputs
        # update nodes
        messgs = self.attention_layer(data, training=training)
        # var 1: add, remove self.add if not needed
        updts = self.add( [data,messgs] ) 
        # var 2: concat
        #updts = tf.concat( (data,messgs), axis=-1 )
        updts = self.layernorm(updts, training=training)
        updated = self.upd_layer( updts, training=training )*mask
        # undate master node
        nodes_mn_msg = self.mn_msg_creator((updated, mask))
        mn_upd = self.mn_updater( tf.concat((mn,nodes_mn_msg), axis=-1) )
        return updated, mn_upd
    
class Encoder_mn(tf.keras.layers.Layer):
    
    def __init__(self, enc_layers):
        super().__init__()
        self.enc_layers = enc_layers
        self.depth = len(enc_layers)
    
    def call(self, inputs, training=False):
        x, mask, mn = inputs
        for i in range(self.depth):
            x, mn = self.enc_layers[i]((x,mask,mn), training=training)
        return x, mn
    
class EncoderClassifier_mn(tf.keras.Model):

    def __init__(self, pre_layer, encoder, agregator, predict_layer):
        super().__init__()
        self.pre_layer = pre_layer
        self.encoder = encoder
        self.agregator = agregator
        self.predict_layer = predict_layer

    def call(self, inputs, training=False):
        dt_params, wfs_flat, recos, dt_bundle = inputs
        mask = dt_params[:,:,-1:]
        data = dt_params[:,:,:-1]
        mn = tf.zeros( (tf.shape(recos)[0],1) )
        x = self.pre_layer(dt_params, training=training)
        x, mn = self.encoder((x,mask,mn), training=training)
        #x = self.agregator(((x,mask)), training=training)
        #preds = self.predict_layer(x, training=training)
        preds = self.predict_layer(mn, training=training)
        return preds