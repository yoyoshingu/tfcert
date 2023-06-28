import numpy as np
import matplotlib.pyplot as plt

xs = np.array([-1.0, 0.0, 1.0, 2.0, 3.0, 4.0], dtype=float)
ys = np.array([-3.0, -1.0, 1.0, 3.0, 5.0, 7.0], dtype=float)

def  tf101fit(x, y, epochs):
  history={"loss":[], "w":[], "b":[]}
  hismse=[];   hisw=[];   hisb=[]
  w=1; b=1
  for i in range(1, epochs):
    for x_i, y_i in zip(x, y):
      y_hat = x_i *w + b
      err = y_i - y_hat
      w_rate = x_i
      w = w + w_rate * err
      b = b + 1 * err
    #history["loss"].append(chkmse(x,y,w,b))
    history['w'].append(w)
    hisb.append(b)
  print(f'{w=}', f'{b=}')
  return history

history = tf101fit(xs, ys, 10)
plt.plot(history['w'])
plt.show()

import tensorflow as tf
from tensorflow import keras

model = tf.keras.Sequential([keras.layers.Dense(units=1, input_shape=[1])])
model.compile(optimizer='sgd', loss='mean_squared_error')
hist = model.fit(xs, ys, epochs=100, verbose=2)

model.summary()
model.weights
plt.plot(hist.history['loss'])
plt.show()