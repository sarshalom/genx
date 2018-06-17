import pandas as pd
import numpy as np
import os
import keras

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

from sklearn.preprocessing import LabelEncoder


def build_genx_model(n_users, n_items, n_words, embedding_size, word_embedding_size, rev_length, lstm_size):

    # item input
    item_input = keras.layers.Input(shape=(1,), name='item-input')
    item_embedding = keras.layers.Embedding(n_items, embedding_size, name='item-embedding')(item_input)
    item_vec = keras.layers.Flatten(name='item-flatten')(item_embedding)

    # user input
    user_input = keras.layers.Input(shape=(1,), name='user-input')
    user_embedding = keras.layers.Embedding(n_users, embedding_size, name='user-embedding')(user_input)
    user_vec = keras.layers.Flatten(name='user-flatten')(user_embedding)

    # user-item score
    score = keras.layers.dot(axes=[1, 1], name="user-item-score", inputs=[user_vec, item_vec])

    # text input
    text_input = keras.layers.Input(shape=(rev_length,), name='text-input')
    text_embedding = keras.layers.Embedding(n_words+2, word_embedding_size, mask_zero=True, name='word-embedding')(text_input)

    # LSTM generator
    lstm_init = keras.layers.multiply(name="user-item-mul", inputs=[user_vec, item_vec])
    lstm_init = keras.layers.Dense(lstm_size, activation='relu')(lstm_init)                    # so init size is correct

    lstm_generator = keras.layers.LSTM(lstm_size, return_sequences=True)(text_embedding, initial_state=[lstm_init, lstm_init])
    text_output = keras.layers.TimeDistributed(keras.layers.Dense(n_words+2, activation='softmax'))(lstm_generator)

    model = keras.Model(inputs=[user_input, item_input, text_input], outputs=[score, text_output])
    model.compile(keras.optimizers.Adam(1e-2),
                  ['mean_squared_error', 'categorical_crossentropy'],
                  loss_weights=[1, .1])
    model.summary()
    return model


def get_data(DATA_FILE, n_words_max=1000, max_sequence_len=20):
    dataset = pd.read_csv(DATA_FILE, header=None, sep='\t', nrows=None)
    dataset.columns = ['user_id', 'item_id', 'rating', 'review']

    print(dataset.head())

    # Tokenize
    tok = Tokenizer(num_words=n_words_max,
                    filters='',
                    lower=True,
                    split=" ",
                    char_level=False)
    tok.fit_on_texts(dataset.review.values)

    reviews = tok.texts_to_sequences(dataset.review.values)

    # Pad sequences (right)
    reviews = pad_sequences(reviews, maxlen=max_sequence_len, padding='post')

    # users, items
    users = LabelEncoder().fit_transform(dataset.user_id)
    items = LabelEncoder().fit_transform(dataset.item_id)

    return users, items, reviews, dataset.rating.values


if __name__ == "__main__":
    DATA_FILE = os.path.expanduser("~/Downloads/all-sample.txt")
    users, items, reviews, ratings = get_data(DATA_FILE)

    model = build_genx_model(n_users=users.max()+1,
                             n_items=items.max()+1,
                             n_words=1000,
                             embedding_size=20,
                             word_embedding_size=20,
                             rev_length=20,
                             lstm_size=10)

    rev_out = np.array([np.eye(1002)[rev] for rev in reviews])
    model.fit([users, items, reviews], [ratings, rev_out], epochs=1)

    # TODO: change model output method to something that will work with real data
    # TODO: shift target word sequenes by 1 ..


