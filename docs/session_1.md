# Session 1: Introduction

## Tensorflow

`Tensorflow` is an open source library developed by `Google`.
The main purpose of `Tensorflow` is to give us the power of
manipulating mathematical expressions on numerical `tensors`.
Somehow like `NumPy` but with some extra features, like:

* Calculating gradient
* Can run on `GPU`, `TPU` and `CPU`
* Computation can be distributed on different machines pretty easily
* It has an API for other programming languages like `c++` and `Java-script`.

## Keras

`Keras` is a high level API for building and training **Deep learning** models.
It was designed to be a stand-alone project.
But with the help of `tensorflow`, `PyTorch` and `Jax` it can run on top of different hardware.

## Google Colab

## Load notebook from GitHub to Colab

## Deep learning Hello World

```python
from tensorflow.keras.datasets import mnist
from tensorflow import keras
from tensorflow.keras import layers

# prepare data
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

train_images = train_images.reshape((60000, 28 * 28))
train_images = train_images.astype("float32") / 255
test_images = test_images.reshape((10000, 28 * 28))
test_images = test_images.astype("float32") / 255

# define our model
model = keras.Sequential([
    layers.Dense(512, activation="relu"),
    layers.Dense(10, activation="softmax")
])

model.compile(optimizer="rmsprop",
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])

# train our model
model.fit(train_images, train_labels, epochs=5, batch_size=128)

# test our model
test_loss, test_acc = model.evaluate(test_images, test_labels)

```

## Prepare Data

```python
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

train_images = train_images.reshape((60000, 28 * 28))
train_images = train_images.astype("float32") / 255
test_images = test_images.reshape((10000, 28 * 28))
test_images = test_images.astype("float32") / 255
```

## Define Model

```python

model = keras.Sequential([
    layers.Dense(512, activation="relu"),
    layers.Dense(10, activation="softmax")
])

model.compile(optimizer="rmsprop",
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])

```

## Train

```python
model.fit(train_images, train_labels, epochs=5, batch_size=128)
```

## Test

```python
test_loss, test_acc = model.evaluate(test_images, test_labels)
```

## Kaggle

## Assignment

* Change the dataset to a new dataset
  from available datasets in keras
    * [MNIST](https://keras.io/api/datasets/mnist/)
    * [CIFAR10](https://keras.io/api/datasets/cifar10/)
    * [CIFAR100](https://keras.io/api/datasets/cifar100/)
    * [Fashion MNIST](https://keras.io/api/datasets/fashion_mnist/)
* Find a dataset in [Kaggle](https://www.kaggle.com/)
    * Image
    * Classification