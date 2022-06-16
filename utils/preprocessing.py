import contractions
import typing
import re
import tensorflow as tf

from tensorflow.keras.layers.experimental.preprocessing import TextVectorization



def clean_dataset(sentences: typing.List[str]) -> typing.List[str]:
	"""
	Cleans the dataset
	1. Removes null values, in this dataset 'Unknown' (found after data exploration)
	Skip for now - correct spellings # may change spellings of unknown names like 'Swetha Srikari' into 'seth shikari'
	2. expand contractions
	3. remove ascii characters (Deal with - son's, doctor's later)

	Returns a clean list of sentences after pre-processing

	:param sentences: list of sentences to be cleaned
	"""

	# 1. Removes 'Unknown's' from all sentences
	clean_sentences = [i for i in sentences if i != 'Unknown']

	# 2. Expands contracted words like I'm, shouldn't to I am, should not
	def expand_contractions(inp_sentence: str) -> str:
		"""
		Expands all contracted words and returns the 'expanded' sentence

		Returns sentences after expanding contracted words

		:inp_sentence: string of words (sentence) to be processed
		"""


		# creating an empty list
		expanded_words = []   
		for word in inp_sentence.split():
		  # using contractions.fix to expand the shortened words
		  expanded_words.append(contractions.fix(word))

		return " ".join(expanded_words)

	clean_sentences = [expand_contractions(lines) for lines in clean_sentences]


	# Removes ascii characters from the input text with removing apostrphe in son's, doctor's
	def handle_special_chars(inp_sentence: str) -> str:
		"""
		This function handles special characters in the input sentences
		1. Removes non-ascii characters
		2. splits punctuation and words separately, ex - 'steps,' -> 'steps', ',' to reduce vocab size
		3. removes apostrophe at the beginning or end of a word (to reduce vocab)

		Returns sentences after handling special characters

		:inp_sentences: the sentences to be processed
		"""
		# This step is skipped because glove embeddings don't represent apostrophe
		# inp_sentence = re.sub('’', "'", inp_sentence) # Change '‘' (non-ascii) character to "'" so that it is not removed during next step
		inp_sentence = inp_sentence.encode('utf-8').decode('ascii', 'ignore') # Remove non-ascii characters
		new = []
		# Remove apostrophe at the beginning and the end of a word,
		# might remove apostrophes like democrats' decision, students' homework
		# splits punctuation as well
		new += [re.sub(r"(^')|('$)", '', s) for s in re.findall(r"[\w']+|[.,!?;~`@#$%^&*()-_+={|}:<>/]", inp_sentence)]
		inp_sentence = " ".join(new)

		return inp_sentence

	clean_sentences = [handle_special_chars(i) for i in clean_sentences]

	return clean_sentences


def tokenize_input(sentences: typing.List[str]):
	"""
	Tokenizes the input - assigns a unique integer to each unique word and
	returns a RaggedTensor instead of a dense Tensor, 
	where each sequence may have a different length

	Returns a tokenizer

	:param sentences: list of sentences after pre-processing
	"""

	# Tokenizing and vectorizing the text for the model to understand
	# Converts to lowercase and does not strips punctuation and 
	tokenizer = TextVectorization(standardize='lower', output_mode='int', ragged=True) 
	tokenizer.adapt(sentences)

	return tokenizer

# Creating n-gram sequences
def create_ngrams(sentences: typing.List[str], tokenizer) -> typing.List[tf.Tensor]:
	"""
	Returns all n-grams for each input sequence

	:param sentences: list of sentences after pre-processing
	"""
	input_sequences = []
	for headline in sentences:
		tokens = tokenizer(headline)
		input_sequences.extend([tokens[:i+1] for i in range(1, len(tokens))])
	return input_sequences


def split_inputs_labels(input_sequences: typing.List[tf.Tensor]):
	"""
	Returns input sequences (all tokens except last one) and it's labels (last token)

	:input_sequences: list of n-gram tokens
	"""
	predictors, labels = [], []
	predictors.extend([i[:-1] for i in input_sequences])
	labels.extend([i[-1] for i in input_sequences])

	return predictors, labels