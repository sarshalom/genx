import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import keras
from keras.optimizers import Adam
from keras.utils.vis_utils import model_to_dot


def split_data(dataset):
    return train_test_split(dataset, test_size=0.2)


def learn(train, n_users, n_items, n_hidden_size):
    item_input = keras.layers.Input(shape=[1], name='Item')
    item_embedding = keras.layers.Embedding(n_items + 1, n_hidden_size, name='item-Embedding')(item_input)
    item_vec = keras.layers.Flatten(name='Flattenitems')(item_embedding)

    user_input = keras.layers.Input(shape=[1], name='User')
    user_vec = keras.layers.Flatten(name='FlattenUsers')(
        keras.layers.Embedding(n_users + 1, n_hidden_size, name='User-Embedding')(user_input))

    prod = keras.layers.merge([item_vec, user_vec], mode='dot', name='DotProduct')
    model = keras.Model([user_input, item_input], prod)
    model.compile('adam', 'mean_squared_error')
    model.fit([train.user_id, train.item_id], train.rating, epochs=100, verbose=0)
    return model


def run(file_path):
    dataset = pd.read_csv(file_path, header=None, sep='\t')
    dataset.columns = ['user_id', 'item_id', 'rating', 'review']
    train_set, test_set = split_data(dataset)
    n_users, n_items = len(dataset.user_id.unique()), len(dataset.item_id.unique())
    n_hidden_size = 50
    model = learn(train_set, n_users, n_items, n_hidden_size)


run("/Users/osarshalom/Downloads/all-sample.txt")
