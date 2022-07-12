# Text-generation
Exploring different language models and word embeddings to generate text (next word prediction)

# Dataset
The text generation project utilizes data collected from New York times. The data contains information about the comments made on the articles published in New York Times between January and May of 2017 and 2018. The articles csv file contains 16 features about more than 9,000 articles. 

The most common words used in these artcles can summed up using WordCloud

![WordCloud](/images/wordcloud.png)

For text generation, sentences from headline and snippet columns are used.

The csv files can be downloaded from here - https://www.kaggle.com/code/shivamb/beginners-guide-to-text-generation-using-lstms/data

# Data exploration
The [Data analysis](https://colab.research.google.com/drive/1HNRpc6PxcjBO_-swXn2DVgFUwpYtOLN2#scrollTo=TdVxoIfAjCb8) Colab notebook helps to understand more about the dataset. 

# Requirements
To run the `train.py` script and train the models, the libraries listed in the `requiremnets.txt` have to be installed.

# Training
To run the model, run the `train.py` script within the `Text-generation` folder with appropriate arguments.

```
$ python train.py --data_dir <path to dataset> --logs_dir <path to save model logs> --embedding <word embeddings to use> --embedding_dir <path to embeddings> --batch_size <batch_size> --epochs <number of epochs to train for> --seed <random state seed> --debug <for debugging>
```

# References
Text pre-processing - https://www.exxactcorp.com/blog/Deep-Learning/text-preprocessing-methods-for-deep-learning

Expanding word contractions - https://www.geeksforgeeks.org/nlp-expand-contractions-in-text-processing/

Tensorflow's ragged tensor to handle inputs of different length - https://www.tensorflow.org/guide/ragged_tensor

Tensorflow dataset - https://www.tensorflow.org/guide/data

Tensorflow's text generation - https://www.tensorflow.org/text/tutorials/text_generation

Tensorflow training - https://www.tensorflow.org/guide/keras/train_and_evaluate

Writing training loop from scratch - https://www.tensorflow.org/guide/keras/writing_a_training_loop_from_scratch

Saving and loading models - https://www.tensorflow.org/tutorials/keras/save_and_load

Tensorflow embeddings - https://www.tensorflow.org/text/guide/word_embeddings

GloVe embeddings - https://nlp.stanford.edu/projects/glove/

Using GloVe as keras Embedding layer - https://keras.io/examples/nlp/pretrained_word_embeddings/#load-pretrained-word-embeddings

WordCloud - https://www.numpyninja.com/post/nlp-text-data-visualization