from load_from_config import dataset_from_config, model_from_config
import os
import tensorflow as tf
import numpy as np

from nn.losses import MyLoss
from nn.metrics import nlogE_MAE, nSigmaMAE

# set gpu
os.environ['CUDA_VISIBLE_DEVICES'] = "1"

# function for counting parameters of model
def count_params(model):
    print([v.get_shape() for v in model.trainable_weights])
    trainableParams = np.sum([np.prod(v.get_shape()) for v in model.trainable_weights])
    nonTrainableParams = np.sum([np.prod(v.get_shape()) for v in model.non_trainable_weights])
    return trainableParams, nonTrainableParams

model, _ = model_from_config(MODEL_FUNC_NAME="TwoTapesModel")
model.build(input_shape=(None, None, 6))
print(model.summary())
print(count_params(model))

optimizer = tf.keras.optimizers.Adam
model.compile(optimizer_E=optimizer(learning_rate=0.01),
              optimizer_sigma=optimizer(learning_rate=0.01),
              loss_E=MyLoss(), loss_sigma=None,
              metrics=[nlogE_MAE(weighted=False), nSigmaMAE(weighted=False)],
              weighted_metrics=[nlogE_MAE(weighted=True), nSigmaMAE(weighted=True)]
              )
(ds, _a, _b), _ = dataset_from_config(regime='train')
history = model.fit(ds, steps_per_epoch=100, epochs=2)
print(history.history)
