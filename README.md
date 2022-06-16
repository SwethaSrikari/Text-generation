# Text-generation
Exploring different language models and word embeddings to generate text (next word prediction)

# Dataset
The text generation project utilizes data collected from New York times. The data contains information about the comments made on the articles published in New York Times in Jan-May 2017 and Jan-April 2018. The articles csv file contains 16 features about more than 9,000 articles.

For text generation, sentences from headline and snippet columns are used.

The csv files can be downloaded from here - https://www.kaggle.com/code/shivamb/beginners-guide-to-text-generation-using-lstms/data

# Data exploration
The [Data analysis](https://colab.research.google.com/drive/1HNRpc6PxcjBO_-swXn2DVgFUwpYtOLN2#scrollTo=TdVxoIfAjCb8) Colab notebook helps to understand more about the dataset. 

# Training
To run the model, run the `train.py` script within the `Text-generation` folder with appropriate arguments.

```
$ python train.py --data_dir <path to dataset> --logs_dir <path to save model logs> --embedding <word embeddings to use> --embedding_dir <path to embeddings> --batch_size <batch_size> --epochs <number of epochs to train for> --seed <random state seed>
```