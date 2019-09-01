import tensorflow as tf
from tensorflow.keras import layers

from ray.rllib.models.tf.tf_modelv2 import TFModelV2
from ray.rllib.models import ModelCatalog
from ray.rllib.models.tf.fcnet_v2 import FullyConnectedNetwork


__all__ = ["register_agent"]


class Agent(TFModelV2):
    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        super(Agent, self).__init__(
            obs_space, action_space, num_outputs, model_config, name
        )

        # TODO: find better way?
        self.inputs = layers.Input(shape=(4, 4), name="inputs")

        x = layers.Flatten()(self.inputs)
        x = layers.Dense(256, activation="relu")(x)
        x = layers.Dense(256, activation="relu")(x)
        x = layers.Dense(256, activation="relu")(x)
        x = layers.Dense(256, activation="relu")(x)
        self.logits = layers.Dense(self.num_outputs, activation=None)(x)

        self.value_out = layers.Dense(1)(x)

        self.base_model = tf.keras.Model(self.inputs, [self.logits, self.value_out])
        self.register_variables(self.base_model.variables)

        self.obs_space = obs_space

    def forward(self, input_dict, state, seq_lens):
        intent_vector, self._value_out = self.base_model(input_dict["obs"]["obs"])
        action_logits = intent_vector * input_dict["obs"]["valid_action_mask"]

        return action_logits, state

    def value_function(self):
        return tf.reshape(self._value_out, [-1])


def register_agent():
    name = "custom_model"
    ModelCatalog.register_custom_model(name, Agent)
    print("Registered agent as {name}")
