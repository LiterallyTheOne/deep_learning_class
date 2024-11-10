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

## Flatten layer

`Flatten layer` is simply flatten the output of the previous layer. 
If we have `5` data which are with the shape of `(8, 9)`, the output
of a `flatten layer` would be `5` data with the shape of `(72,)`.
To use a flatten layer we can simply use the code below:

```python
flatten_layer = keras.layers.Flatten()
```

Since the output of our input layer is `(80, 90, 3)`, we should flatten
this output in order to give it to our `dense layer`.
So let's add our `flatten layer` to our sequential model like this.

```python
model = keras.Sequential(
    [
        keras.layers.Input(shape=(80, 190, 3)),
        keras.layers.Flatten(),
        keras.layers.Dense(4, activation="softmax"),
    ],
)
```

```{note}
source: https://keras.io/api/layers/reshaping_layers/flatten/ 
```

## Fit the model

To fit the model, firstly we should use the `compile` function to determine our
`loss function`, `optimizer` and `metrics`.
We will be discussing about them in the next session further more.
But for now use the code below to compile the model.

```python
model.compile(
    optimizer="adam",
    loss=keras.losses.SparseCategoricalCrossentropy(),
    metrics=["accuracy"],
)
```

After compiling the model, we are ready to use the `fit` function.
This function helps us to train our model.
Some important arguments of this function are:

* x: Input data
* y: Target data (labels)
* batch_size: number of data in each batch
* epochs: number of iterations through all batches
* validation_data: validation data
* callbacks: list of callbacks (we are going to discuss about it more)

To train our model with train data in 10 epochs we can use the code below:

```python
model.fit(train_data, epochs=10)
```

```{note}
source: https://keras.io/api/models/model_training_apis/
```

## Evaluate the model

To evaluate our model on the test data we can use the code below:

```python
model.evaluate(test_data)
```

## Split a tensorflow dataset

To split a tensorflow dataset we can use the technique of `take` and `skip`.
Imagine we have a dataset containing 150 batches.
If we want to get 100 data as our first part and 50 data for the second part we can use the code below:

```python
first_part = data.take(100)
second_part = data.skip(100)
```

This technique is super practical when we want to split our `train` dataset to `train` and `validation`.
We can use it like the below example:

```python
new_train_data_size = np.ceil(train_data.cardinality().numpy() * 0.8)

new_train = train_data.take(new_train_data_size)
new_valid = train_data.skip(new_train_data_size)
```

In the example above we split 80 percent of our `train` dataset to `new_train`
and the 20 perecnt remaining to `new_valid`.


## Fit our model with having validation data

To fit our model with validation data we can simply pass it to the `validation_data` argument:

```python
model.fit(new_train, epochs=10, validation_data=(new_valid))
```


## Transfer learning

Transfer learning is a technic that we are using a pretrained model (called base model), on a new dataset with different purpose.
We simply don't train the base model and we only train the layers that we manually add.
To get prepared for the transfer learning:
* Load the model without its classification layers
* Put the training of the base model to `false`
* Change the input layer according to our dataset input
* Change the output layer according to our number of classes

For example for `MobileNetV2`, we can load it like below:

```python
base_model = keras.applications.MobileNetV2(include_top=False)

base_model.trainable = False
```

```{note}
Different models available in `Keras`: https://keras.io/api/applications/
```

After loading it we can simply add it to our simple sequiential model that we made before as a layer:

```python
model = keras.Sequential(
    [
        keras.layers.Input(shape=(80, 190, 3)),
        base_model,
        keras.layers.Flatten(),
        keras.layers.Dense(4, activation="softmax"),
    ],
)
```

Now we are ready to train our model.

