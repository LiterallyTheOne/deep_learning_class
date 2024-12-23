# Session 2: Load data from Kaggle and make it to a dataset

In this session we are going to get a dataset from
colab, then extract it and learn how [`image_dataset_from_directory`](https://keras.io/api/data_loading/image/) works.

## Load dataset to colab

Now there is a better way to do that, using `kagglehub`.
To download a dataset you don't need to have an `api token`.
for example, if we want to download a dataset of 
[Tom and Jerry Image classification](https://www.kaggle.com/datasets/balabaskar/tom-and-jerry-image-classification)
We can simply use the code below (Make sure `kagglehub` is installed).

```python
import kagglehub

# Download latest version
path = kagglehub.dataset_download("balabaskar/tom-and-jerry-image-classification")

print("Path to dataset files:", path)
```

It would download the dataset and put it in `.cache` directory.


## `image_dataset_from_directory`

[`image_dataset_from_directory`](https://keras.io/api/data_loading/image/) is a built-in function in `Keras`, we are using it to load our dataset.
Each dataset have a different structure, 
but mostly, they follow the following strcuture:

```text
main_directory/
...class_a/
......a_image_1.jpg
......a_image_2.jpg
...class_b/
......b_image_1.jpg
......b_image_2.jpg
```

There is a `main_directory`, in this directory there are
some sub-directories which represent each `class`.
Finally in each sub-directory there are the data we needed.

There are some arguments that we can use:

```python
keras.utils.image_dataset_from_directory(
    directory,
    labels="inferred",
    label_mode="int",
    class_names=None,
    color_mode="rgb",
    batch_size=32,
    image_size=(256, 256),
    shuffle=True,
    seed=None,
    validation_split=None,
    subset=None,
    interpolation="bilinear",
    follow_links=False,
    crop_to_aspect_ratio=False,
    pad_to_aspect_ratio=False,
    data_format=None,
    verbose=True,
)
```

```{note}
https://keras.io/api/data_loading/image/
```

## `tf.data.Dataset`

It is designed to have so much efficency when working with data,
specially when you are dealing with a large-scale dataset. 
The output of `image_dataset_from_directory` would be a
`tensorflow dataset`.  
So every function that we can use on `dataset` is apply able
to the `dataset` that we get from `image_dataset_from_directory`.
Some important functions are:

* `as_numpy_iterator()`
* `map(map_func, num_parallel_calls=None, deterministic=None, name=None)`
* `unbatch()`
* `batch()`
* `rebatch()`
* `take()`
* `skip()`
* `cardinality()`

```{note}
https://www.tensorflow.org/api_docs/python/tf/data/Dataset
```

## train dataset

This is the dataset we are using to train our model with.
This dataset is usually large.

## validation dataset

This is the dataset we are using to validate our model.
We don't train our model with this dataset.
During training we use this dataset to see how our traning is processed.
This usually is smaller than the **train dataset**.

## test dataset

To evaluate our model, we use test dataset.
The size of test dataset is usually simillar to validation dataset.
These are the unseen data that is not being used on the process of training.

## show a batch of data with `matplotlib`

So now we load our datasets, let's show them with matplotlib.
There are multiple ways on doing that.
One of the easiest way is to first cast our `tensorflow dataset`
to a `numpy_iterator`.
Then, use `next()` function to get one batch of it.
After that, iterate through the batch and show the images with
their labels.
For example:

```python
one_batch = next(train_data.as_numpy_iterator())

fig, axes = plt.subplots(3, 4)

axes_ravel = axes.ravel()

for i, (image, label) in enumerate(zip(one_batch[0], one_batch[1])):
    axes_ravel[i].imshow(image.astype("uint8"))
    axes_ravel[i].set_axis_off()
    axes_ravel[i].set_title(f"{label}")

```

