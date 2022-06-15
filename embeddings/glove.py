import os
import typing
import numpy as np
from pathlib import Path


def create_embedding_matrix(embedding_dir: str, embedding_dim: int, tokenizer) -> dict:
	"""
	Loads glove embeddings and Creates a matrix with the input data vocabulary and glove embeddings
	if the word is in the embeddings  or else it is considered unknown and assigned the 
	embeddings of [UNK] i.e, zeros

	:param embeddings_dir: Directory where embeddings are located
	:param embedding_dim: dimensions of the embedding
	:param tokenizer: Input data tokenizer
	"""

	def load_glove_embeddings(embeddings_dir: str) -> dict:
		"""
		Loads glove embeddings and returns a dictionary of words and their embeddings (n-D vector)

		:param embeddings_dir: Directory where embeddings are located
		"""
		path_to_glove_file = os.path.join(os.path.expanduser("~"), embeddings_dir)

		embeddings_index = {}
		with open(path_to_glove_file) as f:
			for line in f:
				word, coefs = line.split(maxsplit=1)
				coefs = np.fromstring(coefs, "f", sep=" ")
				embeddings_index[word] = coefs

		return embeddings_index


	# Load embeddings
	embeddings_index = load_glove_embeddings(embedding_dir)

	# Create a dictionary, mapping words in the input data to their tokens, created by tokenizer
	vocab = tokenizer.get_vocabulary()
	word_tokens = dict(zip(vocab, range(len(vocab))))

	num_tokens = len(vocab) # Vocab size
	
	# Prepare embedding matrix
	embedding_matrix = np.zeros((num_tokens, embedding_dim))
	for word, i in word_tokens.items():
		embedding_vector = embeddings_index.get(word)
		if embedding_vector is not None:
			# Words not found in embedding index will be all-zeros.
			# This includes the representation for "padding" and "OOV"
			embedding_matrix[i] = embedding_vector

	return embedding_matrix
