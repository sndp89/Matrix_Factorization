# Matrix Factorization using Keras on GPU
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


from sklearn.model_selection import train_test_split
dataset = pd.read_csv("recomm.txt", sep=" ",)
dataset.columns.get_values()
dataset.head()
len(dataset.user.unique()), len(dataset.movie_ID.unique())
dataset.user = dataset.user.astype('category').cat.codes.values
dataset.movie_ID = dataset.movie_ID.astype('category').cat.codes.values

from sklearn.model_selection import train_test_split
train, test = train_test_split(dataset, test_size=0.2)
train.head()
test.head()

import keras
from IPython.display import SVG
from keras.optimizers import Adam
from keras.utils.vis_utils import model_to_dot
from keras.utils import multi_gpu_model
n_users, n_movies = len(dataset.user.unique()), len(dataset.movie_ID.unique())
n_latent_factors = 3

movie_input = keras.layers.Input(shape=[1],name='Movie')
movie_embedding = keras.layers.Embedding(n_movies + 1, n_latent_factors, name='Movie-Embedding')(movie_input)
movie_vec = keras.layers.Flatten(name='FlattenMovies')(movie_embedding)

user_input = keras.layers.Input(shape=[1],name='User')
user_vec = keras.layers.Flatten(name='FlattenUsers')(keras.layers.Embedding(n_users + 1, n_latent_factors,name='User-Embedding')(user_input))

prod = keras.layers.merge([movie_vec, user_vec], mode='dot',name='DotProduct')
model = keras.Model([user_input, movie_input], prod)
parallel_model = multi_gpu_model(model, gpus=4)
parallel_model = model.compile('adam', 'mean_squared_error')

parallel_model.summary()
history = model.fit([train.user, train.movie_ID], train.rating, epochs=100, verbose=0)

pd.Series(history.history['loss']).plot(logy=True)
plt.xlabel("Epoch")
plt.ylabel("Train Error")

y_hat = np.round(model.predict([test.user, test.movie_ID]),0)
y_true = test.rating

from sklearn.metrics import mean_absolute_error
mean_absolute_error(y_true, y_hat)

