from typing import Union
from .nn.custom_layers import *

# try:
#     from .nn.custom_layers import *
# except:
#     from nn.custom_layers import *

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
        # We list our `Metric` objects here so that `reset_states()` can be
        # called automatically at the start of each epoch
        # or at the start of `evaluate()`.
        # If you don't implement this property, you have to call
        # `reset_states()` yourself at the time of your choosing.
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