import typing
import tensorflow as tf

from pathlib import Path

from models.evaluator import Evaluator

class Trainer(Evaluator):
	"""
	CLass to train models for text generation

	:param model: Model to train
	:param optimizer: Optimizer to use when training
	:param train_loss: Loss functio
	:param train_metric: Training metric
	"""

	def __init__(self, model, train_data: tf.data.Dataset, val_data: tf.data.Dataset,
				 optimizer: tf.keras.optimizers.Optimizer = tf.keras.optimizers.Adam(),
                 train_loss: tf.keras.losses.Loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                 valid_loss: tf.keras.losses.Loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                 train_metric: tf.keras.metrics.Metric = tf.keras.metrics.SparseCategoricalAccuracy(),
                 valid_metric: tf.keras.metrics.Metric = tf.keras.metrics.SparseCategoricalAccuracy()):

		super().__init__(model, val_data, valid_loss, valid_metric)

		self.train_data = train_data
		self.optimizer = optimizer
		self.train_loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True) #### Check why proper train loss is not passed as argument
		self.train_metric = train_metric


	def fit(self, epochs: int):
		"""
		Train for specified number of epochs

		:param epochs: number of epochs to train for
		"""
		for epoch in range(epochs):
			# Reset metrics at the start of every epoch
			# self.train_loss.reset_states()
			self.train_metric.reset_states()
			# self.valid_loss.reset_states()
			self.valid_metric.reset_states()

			self.fit_one(epoch)


	@tf.function
	def train_step(self, train_inputs: tf.Tensor, train_targets: tf.Tensor):
		"""
		Performs a single train step on the training inputs and targets

		:param train_inputs: Training inputs
		:param train_targets: Training targets
		"""

		raise NotImplementedError("Subclass Trainer class to execute train_step specific to your model")


	def fit_one(self, epoch: int = 1):
		"""
		Fit model for one epoch

		:param epoch: Current epoch
		"""
		print("Epoch number - ", epoch+1)
		for step, (inputs, targets) in enumerate(self.train_data):
			loss_value_train = self.train_step(inputs, targets)

			# Log every 200 batches.
			if step % 200 == 0:
				print("Training loss (for one batch) at step %d: %.4f"% (step, float(loss_value_train)))

		# loss_value_valid = self.evaluate(epoch) ### fix valid data argument

		# print(f'Valid loss for epoch {epoch+1} is {loss_value_valid}')










