# Session 1: Introduction

## Tensorflow

## Keras

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

## Define Model

## Train

## Test

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