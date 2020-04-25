import numpy as np
import pandas as pd
import re
import os
import tensorflow as tf
from tensorflow import keras
from keras.utils import to_categorical
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential, load_model
from keras.layers import LSTM, Dense, GRU, Embedding
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
import sys
from gru_model import generate_sequence

def text_cleaner(text):
    # lower case text
    newString = text.lower()
    newString = re.sub(r"'s\b","",newString)
    # remove punctuations
    newString = re.sub("[^a-zA-Z]", " ", newString)
    long_words=[]
    # remove short word
    for i in newString.split():
        if len(i)>=3:
            long_words.append(i)
    return (" ".join(long_words)).strip()


# Count the arguments
arguments = len(sys.argv) - 1

if arguments == 0:
    print("Please type an input string, model desired ('gru' or 'lstm'), and the input string in quotes!\n\n")

if (0 < arguments < 3):
    print("Not enough arguments detected. Please type an input string, model desired ('gru' or 'lstm'), and the input string in quotes!\n\n")

if arguments == 3:
    input_text = sys.argv[1]
    print(type(input_text))

    if sys.argv[2] == 'gru':
        model = load_model('gru_model.h5')
    else:
        print("PUT LSTM MODEL HERE")

    file = open("corpus.txt", "r")
    text_data =  file.read()

    cleaned_text = text_cleaner(text_data)
    chars = sorted(list(set(cleaned_text)))
    mapping = dict((c, i) for i, c in enumerate(chars))

    new_length = int(sys.argv[3])

    print(generate_sequence(model, mapping, 30, input_text.lower(), new_length))

if arguments > 3:
    print("Too many arguments! Remember, only one input string at a time!\n\n")
