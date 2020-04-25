import numpy as np
import pandas as pd
import re
import os
import tensorflow as tf
from tensorflow import keras
from keras.utils import to_categorical
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import LSTM, Dense, GRU, Embedding
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split


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

def create_sequence(text):
    length = 30
    sequences = list()
    for i in range(length, len(text)):
        # select sequence of tokens
        seq = text[i-length:i+1]
        # store
        sequences.append(seq)
    print('Total Sequences: %d' % len(sequences))
    return sequences

def encode_sequence(seq):
    sequences = list()
    for line in seq:
        # integer encode line
        encoded_seq = [mapping[char] for char in line]
        # store
        sequences.append(encoded_seq)
    return sequences

def generate_sequence(model, mapping, seq_length, seed_text, n_chars):
	in_text = seed_text

    # generate a fixed number of characters
	for _ in range(n_chars):
		# encode the characters as integers
		encoded = [mapping[char] for char in in_text]
		# truncate sequences to a fixed length
		encoded = pad_sequences([encoded], maxlen=seq_length, truncating='pre')
		# predict character
		yhat = model.predict_classes(encoded, verbose=0)
		# reverse map integer to character
		out_char = ''
		for char, index in mapping.items():
			if index == yhat:
				out_char = char
				break
		# append to input
		in_text += char

	return in_text


if __name__ == '__main__':
    file = open("corpus.txt", "r")
    text_data =  file.read()

    cleaned_text = text_cleaner(text_data)
    # create sequences
    sequences = create_sequence(cleaned_text)

    # create a character mapping index
    chars = sorted(list(set(cleaned_text)))
    mapping = dict((c, i) for i, c in enumerate(chars))

    # encode the sequences
    sequences = encode_sequence(sequences)

    # vocabulary size
    vocab = len(mapping)
    sequences = np.array(sequences)
    # create X and y
    X, y = sequences[:,:-1], sequences[:,-1]
    # one hot encode y
    y = to_categorical(y, num_classes=vocab)
    # create train and validation sets
    X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    print('Train shape:', X_tr.shape, 'Val shape:', X_val.shape)

    #define model
    model = Sequential()
    model.add(Embedding(vocab, 150, input_length=30, trainable=True))
    model.add(GRU(150, recurrent_dropout=0.1, dropout=0.1))
    model.add(Dense(vocab, activation='softmax'))
    print(model.summary())

    # compile the model
    model.compile(loss='categorical_crossentropy', metrics=['acc'], optimizer='adam')
    # fit the model
    model.fit(X_tr, y_tr, epochs=30, verbose=2, validation_data=(X_val, y_val))

    model.save('GRUmodel.h5')
    #model = tf.keras.models.load_model('model.h5')


    #input_text = "You really like "
    #print(len(input_text))

    #print(generate_sequence(model, mapping, 30, input_text.lower(), 50))
