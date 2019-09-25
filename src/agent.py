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

        K.set_image_data_format("channels_last")  # workers probably have channels_first
        if K.image_data_format() == "channels_last":
            n_rows, n_columns, n_channels = obs_space.original_space["board"].shape
            self.inputs = layers.Input(
                shape=(n_rows, n_columns, n_channels), name="inputs"
            )
        else:
            n_channels, n_rows, n_columns = obs_space.original_space["board"].shape
            self.inputs = layers.Input(
                shape=(n_channels, n_rows, n_columns), name="inputs"
            )

        x = KL.Conv2D(32, kernel_size=(1, 1), strides=(1, 1), padding="same")(
            self.inputs
        )
        x = KL.Conv2D(32, kernel_size=(2, 2), strides=(2, 2), padding="same")(x)
        x = KL.Flatten()(x)
        x = KL.Dense(64)(x)
        # x = layers.Conv2D(32, kernel_size=(1, 1), strides=(1, 1), padding="same")(
        #     self.inputs
        # )
        # x = layers.Conv2D(64, kernel_size=(2, 2), strides=(1, 1), padding="same")(x)
        # x = layers.Flatten()(x)
        # x = layers.Dense(64)(x)
        # self.logits = layers.Dense(self.num_outputs, activation=None)(x)

        # x = layers.Dense(64)(x)
        # self.value_out = layers.Dense(1)(x)
        # __import__("pdb").set_trace()

        # x = layers.Flatten()(self.inputs)
        # x = layers.Dense(512, activation="relu")(x)
        # x = layers.Dense(512, activation="relu")(x)
        # x = layers.Dense(512, activation="relu")(x)
        self.logits = layers.Dense(self.num_outputs, activation=None)(x)

        x = layers.Dense(64)(x)
        self.value_out = layers.Dense(1)(x)

        self.base_model = tf.keras.Model(self.inputs, [self.logits, self.value_out])
        self.base_model.summary()
        self.register_variables(self.base_model.variables)

        self.obs_space = obs_space

    def forward(self, input_dict, state, seq_lens):
        intent_vector, self._value_out = self.base_model(input_dict["obs"]["board"])
        action_logits = intent_vector * input_dict["obs"]["valid_action_mask"]

        return action_logits, state

    def value_function(self):
        return tf.reshape(self._value_out, [-1])


def register_agent():
    name = "custom_model"
    ModelCatalog.register_custom_model(name, Agent)
    print("Registered agent as {name}")
