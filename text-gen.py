#!/usr/bin/env python

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from __future__ import with_statement

__author__ = "Sangjee Dondrub"
__email__ = "sangjeedondrub@live.com"
__license__ = "MIT"
__date__ = "2020 - 04 - 09"


import os
import time
import re

import argparse
import numpy as np

import tensorflow as tf

parser = argparse.ArgumentParser(description="Parse bool")

parser.add_argument('--use_syllable', nargs='*', default=None)
config = parser.parse_args()

use_syllable = config.use_syllable is not None

mode = 'syllable' if use_syllable else 'char'

train_text_path = 'data/lu-drub-gong-gyan.txt'
BATCH_SIZE = 64
# Buffer size to shuffle the dataset
BUFFER_SIZE = 10000
# The embedding dimension
embedding_dim = 256
# Number of RNN units
rnn_units = 1024
EPOCHS = 1
checkpoint_dir = f'./training_checkpoints-{mode}'
# Name of the checkpoint files
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}")

text = open(train_text_path, 'rb').read().decode(encoding='utf-8')

if use_syllable:
    text = re.sub('[^\u0f00-\u0fff ]*', '', text)
    text = text.replace('་', ' ').replace('༎', ' ༎ ').replace('།', ' ། ')
    text = re.sub(' {2,}', ' ', text).split()


vocab = sorted(set(text))

char2idx = {u: i for i, u in enumerate(vocab)}
idx2char = np.array(vocab)
text_as_int = np.array([char2idx[c] for c in text])

# The maximum length sentence we want for a single input in characters
seq_length = 100

# Create training examples / targets
char_dataset = tf.data.Dataset.from_tensor_slices(text_as_int)


def split_input_target(chunk):
    input_text = chunk[:-1]
    target_text = chunk[1:]
    return input_text, target_text


vocab_size = len(vocab)
print(f'vocab_size: {vocab_size}')
sequences = char_dataset.batch(seq_length+1, drop_remainder=True)
dataset = sequences.map(split_input_target)
dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)


def build_model(vocab_size, embedding_dim, rnn_units, batch_size):
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(vocab_size, embedding_dim,
                                  batch_input_shape=[batch_size, None]),
        tf.keras.layers.GRU(rnn_units,
                            return_sequences=True,
                            stateful=True,
                            recurrent_initializer='glorot_uniform'),
        tf.keras.layers.Dense(vocab_size)
    ])
    return model


model = build_model(
    vocab_size=len(vocab),
    embedding_dim=embedding_dim,
    rnn_units=rnn_units,
    batch_size=BATCH_SIZE)


def loss(labels, logits):
    return tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True)


model.compile(optimizer='adam', loss=loss)


checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_prefix,
    save_weights_only=True)


history = model.fit(dataset, epochs=EPOCHS, callbacks=[checkpoint_callback])
tf.train.latest_checkpoint(checkpoint_dir)
model = build_model(vocab_size, embedding_dim, rnn_units, batch_size=1)

model.load_weights(tf.train.latest_checkpoint(checkpoint_dir))
model.build(tf.TensorShape([1, None]))


def generate_text(model, starter):
    num_generate = 1000 if use_syllable else 2000
    input_eval = [char2idx[s] for s in starter]
    input_eval = tf.expand_dims(input_eval, 0)
    text_generated = []

    # Low temperatures results in more predictable text.
    # Higher temperatures results in more surprising text.
    # Experiment to find the best setting.
    temperature = 0.8

    # Here batch size == 1
    model.reset_states()
    for _ in range(num_generate):
        predictions = model(input_eval)
        # remove the batch dimension
        predictions = tf.squeeze(predictions, 0)

        # using a categorical distribution to predict the character returned by the model
        predictions = predictions / temperature
        predicted_id = tf.random.categorical(
            predictions, num_samples=1)[-1, 0].numpy()

        input_eval = tf.expand_dims([predicted_id], 0)

        text_generated.append(idx2char[predicted_id])
    if not use_syllable:
        return (starter + ''.join(text_generated))
    else:
        return '་'.join(starter) + '་'.join(text_generated)


if use_syllable:
    starter = ['དབུ', 'མ']
else:
    starter = 'དབུ་མ་'


results = generate_text(model, starter=starter).replace('\n', ' ')
print(results)

out = f'sample.{mode}.txt'

with open(out, 'w', encoding='utf-8') as f:
    f.write(results)
