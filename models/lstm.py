import typing
import tensorflow as tf

from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, RNN
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import Sequential

class Lstm(tf.keras.Model):
	"""
	LSTM implementation of https://www.kaggle.com/code/shivamb/beginners-guide-to-text-generation-using-lstms/notebook

	The model is sequential with an embedding layer (input), LSTM layer, Dropout and a Dense layer (fully connected output layer)

	:param vocab_size: Length of the input vocabulary
	:param embedding: embedding layer
	:param embedding_dim: embedding dimensions 
	:param lstm_units: lstm dimension
	:param dropout: fraction of inputs to drop
	:param dense_activation: activation type
	"""

	def __init__(self, vocab_size: int, embedding, embedding_dim: int, lstm_units: int, dropout: float, dense_activation: str):
		super(Lstm, self).__init__()
		self.model = Sequential()
		self.embedding_layer = embedding
		self.lstm_layer = LSTM(lstm_units)
		self.dropout_layer = Dropout(dropout)
		self.output_layer = Dense(vocab_size, activation=dense_activation)
		

	def call(self, inputs: tf.data.Dataset, training=False):
		"""
		Adds embedding, lstm, dropout and dense layer to the sequential model

		Returns the model
		"""
		# x = inputs
		# x = self.embedding_layer(x, training=training)
		# x = self.lstm_layer(x, training=training)
		# x = self.dropout_layer(x, training=training)
		# x = self.output_layer(x, training=training)

		# return x
		self.model.add(self.embedding_layer)
		self.model.add(self.lstm_layer)
		self.model.add(self.dropout_layer)
		self.model.add(self.output_layer)

		return self.model(inputs, training=training)