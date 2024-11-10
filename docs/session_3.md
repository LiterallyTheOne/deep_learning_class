# Session 3: Model in keras and transfer learning

## Different models in Keras

There are 3 different ways to define a model in `Keras` :

* Sequential
* Functional
* Subclassing

In this session we focus on the easiest and the most readable way, Sequential.

```{note}
source: https://keras.io/api/models/model/
```

## Sequential model

To define a sequential model, we can simply use `keras.Sequential`.

```python
keras.Sequential(layers=None, trainable=True, name=None)
```

It takes a list of layers that we are going to learn about some of them in this
session.

```{note}
sources:
https://keras.io/api/models/sequential/
https://keras.io/guides/sequential_model/
```

## Input layer

`Input layer` is the layer that we are using
to tell `Keras` what the shape of our input is.
For example, if we want to say the shape of our input is `(80, 190, 3)`, we can
use the code below:

```python
input_layer = keras.layers.Input(shape=(80, 190, 3))
```

So let's add it to our sequential model: 

```python
model = keras.Sequential(
    [
        keras.layers.Input(shape=(80, 190, 3)),
    ],
)
```

```{note}

```

## Dense layer

`Dense layer` (fully connected layer) is a layer which
all the neurons of this layer is connected to the neurons
of the previous layer.
To define a `Dense layer` in `Keras` we can simpy use
`keras.layers.Dense`.
It requires us to define how many neurons we want, also
we can give the activation function that we want to it as well.
For example, if we want to have 10 neurons with the `ReLU` activation,
we can use the code below:

```python
dense_layer = keras.layers.Dense(10, activation="relu")
```

```{note}
source: https://keras.io/api/layers/core_layers/dense/ 
```

## Output layer

`Output layer` is the layer that we are using to generate
our output respect to our problem.
In **classification** problems we mostly use
`Dense layer` with `softmax` as its activation.
For example, if we have 4 classes we can define an output layer like below

```python
keras.layers.Dense(4, activation="softmax"),
```

Let's add it to our sequential model:

```python
model = keras.Sequential(
    [
        keras.layers.Input(shape=(80, 190, 3)),
        keras.layers.Dense(4, activation="softmax"),
    ],
)
```



