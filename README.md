# Sentimental Analysis in Pytorch
## Requirements
Pytorch will be used for buildng all the models

We will be using torchtext library from Pytorch to process our text
``` bash
pip install torchtext
```
We will also use spacy to tokenize our data.To install spaCy
``` bash
python -m spacy download en
```

We will use the transformers library, which can be installed via:

```bash
pip install transformers
```
We will be using the the movie reviwes from IMDB datasets which has only two classes : positive and negative sentiment. 20k examaples are used for training, 5k examplea are used for vaidation and 25k examples are used for testng.


In Single_double_bidirectional.py randomly initiliased embeding were used, Glove_RNN.py and Glove_Biirectional_LSTM used GLove embedding and fially in Transformer_GRU.py we use the BERT embedding.
