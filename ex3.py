import sys
import numpy as np


sigmoid = lambda x: 1 / (1 + np.exp(-x))


def softmax(x):
  e_x = np.exp(x - np.max(x))
  return e_x / e_x.sum()

def fprop(x, y, params):
  # Follows procedure given in notes
  W1, b1, W2, b2 = [params[key] for key in ('W1', 'b1', 'W2', 'b2')]
  x.shape = (len(x), 1)
  z1 = np.dot(W1, x) + b1
  h1 = sigmoid(z1)
  z2 = np.dot(W2, h1) + b2
  h2 = softmax(z2)
  ret = {'x': x, 'y': y, 'z1': z1, 'h1': h1, 'z2': z2, 'h2': h2}
  for key in params:
    ret[key] = params[key]
  return ret

def bprop(fprop_cache):
  # Follows procedure given in notes
  x, y, z1, h1, z2, h2 = [fprop_cache[key] for key in ('x', 'y', 'z1', 'h1', 'z2', 'h2')]
  y_copy = np.zeros(10)
  y_copy[int(y)] = 1
  y_copy.shape = (len(y_copy), 1)
  dz2 = h2 - y_copy                                #  dL/dz2
  dW2 = np.dot(dz2, h1.T)                       #  dL/dz2 * dz2/dw2
  db2 = dz2                                     #  dL/dz2 * dz2/db2
  dz1 = np.dot(fprop_cache['W2'].T,
    dz2) * sigmoid(z1) * (1-sigmoid(z1))   #  dL/dz2 * dz2/dh1 * dh1/dz1
  dW1 = np.dot(dz1, x.T)                        #  dL/dz2 * dz2/dh1 * dh1/dz1 * dz1/dw1
  db1 = dz1                                     #  dL/dz2 * dz2/dh1 * dh1/dz1 * dz1/db1
  return {'b1': db1, 'W1': dW1, 'b2': db2, 'W2': dW2}

def update(derivatives,params,eps):
  for key in params:
    params[key] -= np.dot(derivatives[key], eps)


def train(x, y, params):
  eps=0.1
  i = 0
  for epoch in range(15):
    if (i%4 == 0):
      eps /= 2
    i += 1
    randomize = np.arange(len(x))
    np.random.shuffle(randomize)
    x = x[randomize]
    y = y[randomize]
    for x_i,y_i in zip(x,y):
      fprop_cache = fprop(x_i, y_i, params)
      derivatives = bprop(fprop_cache)
      update(derivatives,params,eps)



if __name__ == '__main__':
  # Initialize random parameters and inputs

  train_x = np.loadtxt(sys.argv[1])
  train_y = np.loadtxt(sys.argv[2])
  test_x = np.loadtxt(sys.argv[3])

  train_x /= 255
  test_x /=255

  randomize = np.arange(len(train_x))
  np.random.shuffle(randomize)
  train_x = train_x[randomize]
  train_y = train_y[randomize]

  W1 = np.random.uniform(0, 0.0001, (110, 784))
  b1 = np.random.uniform(0, 0.0001, (110, 1))
  W2 = np.random.uniform(0, 1, (10, 110))
  b2 = np.random.uniform(0, 1, (10, 1))

  params = {'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2}
  train(train_x, train_y, params)

  f = open("test_y", "w+")
  for x in test_x:
    W1, b1, W2, b2 = [params[key] for key in ('W1', 'b1', 'W2', 'b2')]
    x.shape = (len(x), 1)
    z1 = np.dot(W1, x) + b1
    h1 = sigmoid(z1)
    z2 = np.dot(W2, h1) + b2
    h2 = softmax(z2)
    y_hat = np.argmax(h2)
    f.write(f"{y_hat}\n")
  f.close()
