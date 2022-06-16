import argparse
import tensorflow as tf
import numpy as np
from pathlib import Path
from tensorflow.keras.layers import Embedding

from models.lstmtrainer import LSTMTrainer
from models.lstm import Lstm
from utils.dataset import list_of_sentences, dataset_for_training
from embeddings.glove import create_embedding_matrix

def train(data_dir: str, log_dir: str, embedding: str, embedding_dir: str, batch_size: int, epochs: int, seed: int, debug: bool=False):
	"""
	Trains a text generation model using the specified data from a data directory

	:param data_dir: Directory where data is located
	:param log_dir: Directory where training logs should be saved
	:param embedding_dir: Embeddings directory
	:param embedding: type of word embeddings
	:param epochs: number of epochs to train for
	:param seed: random state seed
	:param debug: uses less data for faster debugging
	"""

	# Set seed
	tf.random.set_seed(seed)

	# Load dataset
	headlines = list_of_sentences(data_dir)

	# Debug
	if debug:
		print('Debugging')
		# Uses just 100 sentences
		headlines = headlines[:100]

	# Prepare dataset for training
	train_dataset, val_dataset, tokenizer = dataset_for_training(headlines, seed, batch_size)

	# Vocabulary
	vocab_size = len(tokenizer.get_vocabulary())

	# Hyperparameters
	embedding_dim = 100
	lstm_units = 256
	dropout = 0.1
	dense_activation = 'softmax'

	# Embedding
	if embedding == 'none':
		embedding = Embedding(vocab_size, embedding_dim, mask_zero=True)
	elif embedding == 'glove100':
		print(f'Using {embedding} embeddings')
		embedding_matrix = create_embedding_matrix(embedding_dir=embedding_dir, embedding_dim=100, tokenizer=tokenizer)
		embedding = Embedding(vocab_size,
							  100,
							  embeddings_initializer=tf.keras.initializers.Constant(embedding_matrix),
							  trainable=False,
							  mask_zero=True)

	# Instiantiate model
	model = Lstm(vocab_size=vocab_size, embedding=embedding, embedding_dim=embedding_dim,
				 lstm_units=lstm_units, dropout=dropout, dense_activation=dense_activation)

	# Optimizer
	optimizer = tf.keras.optimizers.Adam()

	# Loss functions and metrics
	train_loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)
	valid_loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)
	train_metric = tf.keras.metrics.SparseCategoricalAccuracy()
	valid_metric = tf.keras.metrics.SparseCategoricalAccuracy()

	# Trainer
	trainer = LSTMTrainer(model=model, train_data=train_dataset,
						  val_data=val_dataset,optimizer=optimizer,
						  train_loss=train_loss, valid_loss=valid_loss,
						  train_metric=train_metric, valid_metric=valid_metric)

	# Train
	trainer.fit(epochs)

	print('---- Training complete ----')



if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--data_dir', type=str,
						default='dataset/',
						help='Dataset directory')
	parser.add_argument('--logs_dir', type=str, default='',
						help='Path to training logs')
	parser.add_argument('--embedding', type=str, default='none', help='Type of word embeddings to use', choices=['none', 'glove100'])
	parser.add_argument('--embedding_dir', type=str, default='', help='Embeddings directory')
	parser.add_argument('--batch_size', type=int, default=64, help='The desired batch size to use when training')
	parser.add_argument('--epochs', type=int, default=1, help='Number of epochs to train for')
	parser.add_argument('--seed', type=int, default=271, help='random state seed')   
	parser.add_argument('--debug', type=bool, default=False, help='Uses less data to help debug quickly')
	args = parser.parse_args()
	train(args.data_dir, args.logs_dir, args.embedding, args.embedding_dir, args.batch_size, args.epochs, args.seed, args.debug)




