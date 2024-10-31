# Session 2: Load data from colab to Kaggle

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
but mostly they follow the following strcuture:

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

## Load dateset in train, validation and test format

## Prepare data to feed it to model

## Fit the model with train and validation

## Test the model with test

