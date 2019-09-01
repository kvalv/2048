import tensorflow as tf
from tensorflow.keras import layers

from ray.rllib.models.tf.tf_modelv2 import TFModelV2
from ray.rllib.models import ModelCatalog

__all__ = ["register_agent"]


class Agent(TFModelV2):
    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        super(Agent, self).__init__(
            obs_space, action_space, num_outputs, model_config, name
        )
        self.inputs = layers.Input(shape=obs_space.shape, name="inputs")

        x = layers.Flatten()(self.inputs)
        x = layers.Dense(256)(x)
        x = layers.Dense(256)(x)
        x = layers.Dense(256)(x)
        x = layers.Dense(256)(x)
        layer_out = layers.Dense(self.num_outputs)(x)
        value_out = layers.Dense(1)(x)

        self.base_model = tf.keras.Model(self.inputs, [layer_out, value_out])
        self.register_variables(self.base_model.variables)

        self.obs_space = obs_space

    def forward(self, input_dict, state, seq_lens):
        model_out, self._value_out = self.base_model(input_dict["obs"])
        return model_out, state

    def value_function(self):
        return tf.reshape(self._value_out, [-1])


def register_agent():
    name = "custom_model"
    ModelCatalog.register_custom_model(name, Agent)
    print("Registered agent as {name}")
