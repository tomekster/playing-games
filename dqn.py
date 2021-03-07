from tensorflow.keras import layers
from tensorflow import keras

class Model:
	def __init__(self, input_shape, output_size):
		self.input_shape = input_shape
		self.output_size = output_size

	def build(self):
		inputs = layers.Input(shape=self.input_shape)
		layer1 = layers.Conv2D(32, 8, strides=4, activation="relu")(inputs)
		layer2 = layers.Conv2D(64, 4, strides=2, activation="relu")(layer1)
		layer3 = layers.Conv2D(64, 3, strides=1, activation="relu")(layer2)

		layer4 = layers.Flatten()(layer3)

		layer5 = layers.Dense(512, activation="relu")(layer4)
		action = layers.Dense(self.output_size, activation="linear")(layer5)
		return keras.Model(inputs=inputs, outputs=action)
