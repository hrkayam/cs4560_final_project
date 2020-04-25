import numpy as np
import pandas as pd
import re
from keras.utils import to_categorical
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import LSTM, Dense, GRU, Embedding, Dropout, Activation, RNN, Concatenate, multiply
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
import os
import tensorflow as tf
import winsound
from custom_layer import UserEmbedCell


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

file = open("smallcorpus.txt", "r")
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
X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=0.1, random_state=42)

print('Train shape:', X_tr.shape, 'Val shape:', X_val.shape)


hidden_size = 50
num_users = 100
user_vector = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0]

#define model
first = Sequential()
first.add(Embedding(vocab, 128, input_length=30, trainable=True))
first.add(Dense(vocab, activation='softmax'))

second = Sequential()
dense_2 = Embedding(len(user_vector), 30, trainable=True)
second.add(dense_2)
weights = dense_2.get_weights()
user_embedding = np.dot(user_vector, weights[0])
second.add(RNN(UserEmbedCell(128, user_embedding)))
second.add(Activation('softmax'))

model = multiply([first, second])

# model = Sequential()
# model.add(Embedding(vocab, 50, input_length=30, trainable=True))
# model.add(LSTM(150, recurrent_dropout=0.1, dropout=0.1))
# model.add(Dense(vocab, activation='softmax'))

# model = Sequential()
# model.add(Embedding(vocab, hidden_size, input_length=30, trainable=True))
# model.add(LSTM(hidden_size, recurrent_dropout=0.1, dropout=0.1, return_sequences=False))
# # model.add(Dense(vocab, activation='tanh'))
# # model.add(LSTM(hidden_size, recurrent_dropout=0.1, dropout=0.1, return_sequences=False))
# # model.add(LSTM(150, recurrent_dropout=0.2, dropout=0.2))
# model.add(Dense(vocab, activation='tanh'))
# model.add(Dense(vocab, activation='softmax'))


checkpoint_path = "cp-{epoch:04d}.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

cp_callback = ModelCheckpoint(
    filepath=checkpoint_path,
    verbose=1,
    save_best_only=1,
    save_weights_only=True,
    period=5)

# compile the model
model.compile(loss='categorical_crossentropy', metrics=['acc'], optimizer='adam')
# fit the model
model.fit(X_tr, y_tr, epochs=20, verbose=2, validation_data=(X_val, y_val), callbacks=[cp_callback])

model.summary()


frequency = 2500  # Set Frequency To 2500 Hertz
duration = 500  # Set Duration To 1000 ms == 1 second
winsound.Beep(frequency, duration)

model_ckpt = model
model_ckpt.load_weights("cp-0020.ckpt")

input_text = "knock knock"
print(len(input_text))

outputs = [layer.output for layer in model.layers]
print(generate_sequence(model, mapping, 30, input_text.lower(), 100))
print(outputs)
