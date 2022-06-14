import typing
import os
import pandas as pd
import tensorflow as tf

from sklearn.model_selection import train_test_split

from .preprocessing import clean_dataset, tokenize_input, create_ngrams, split_inputs_labels

def list_of_sentences(data_dir: str) -> typing.List[str]:
	"""
	Extracts headlines from each csv file and creates a list of all headlines

	Returns a list of strings (a list of headlines(sentences))

	:data_dir: Directory where all the csv files are located
	"""
	all_headlines = []
	for filename in os.listdir(data_dir):
		csv_file = pd.read_csv(data_dir + '/' + filename)
		headlines = csv_file.headline.values
		all_headlines.extend(headlines) # creates a list of sentences from all the csv files
	return all_headlines


def dataset_for_training(headlines: typing.List[str], seed: tf.int64, batch_size: tf.int64):
	"""
	1. Pre-processes data for training
	2. Tokenizes input
	3. Creates ngrams
	4. Splits data into inputs and targets/labels
	5. Splits data into training and validation sets
	6. Dynamically pads sequences to max sequence length in each batch

	Returns training and validation dataset to be used for training a tensorflow model
	along with the tokenizer

	:headlines: list of headlines
	:seed: to recreate experiments
	:batch_size: the number of samples in a batch
	"""

	# 1. Pre-processes data for training
	clean_headlines = clean_dataset(headlines)[:100]

	# 2. Tokenizes input
	tokenizer = tokenize_input(clean_headlines)

	# 3. Creates ngrams
	ngram_sequences = create_ngrams(clean_headlines, tokenizer)

	# 4. Splits data into inputs and targets/labels
	inputs, targets = split_inputs_labels(ngram_sequences)

	# 5. Creating training and validation sets
	x_train, x_val, y_train, y_val = train_test_split(inputs, targets, shuffle=True,
													  test_size=0.2, random_state=seed)
	# batch_size = 128

	# Creating tf dataset from generator (for variable length inputs 'from_tensor_slices' doesn't work)
	# train dataset
	train_dataset = tf.data.Dataset.from_generator(lambda: ((x, y) for (x,y) in zip(x_train, y_train)),
												   output_types=(tf.as_dtype(x_train[0].dtype), tf.as_dtype(y_train[0].dtype)),
												   output_shapes=([None,], ()))
	# Valid dataset
	val_dataset = tf.data.Dataset.from_generator(lambda: ((x, y) for (x,y) in zip(x_val, y_val)),
												 output_types=(tf.as_dtype(x_val[0].dtype), tf.as_dtype(y_val[0].dtype)),
												 output_shapes=([None,], ()))


	# 6. Dynamic padding pads the sequences to the maximum sequence length of each batch
	# Prepare the training dataset.
	train_dataset = train_dataset.shuffle(buffer_size=1024).padded_batch(batch_size)

	# Prepare the validation dataset.
	val_dataset = val_dataset.padded_batch(batch_size)

	return train_dataset, val_dataset, tokenizer









