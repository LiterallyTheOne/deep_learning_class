# Session 2: Load data from colab to Kaggle

In this session we are going to get a dataset from
colab, then extract it and learn how [`image_dataset_from_directory`](https://keras.io/api/data_loading/image/) works.

## Load dataset to colab

In the previous session we have chosen a dataset and download it to
google colab using `kaggle api`.

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

## Extract the dataset

When we get the data from `kaggle api`, it is zipped,
we can use the code below to unzip it:

```sh
! unzip your_dataset.zip
```

options of `unzip`:

* `-q`: perform operations quietly 
* `-qq`: perform operations even more quietly 


## Load dateset in train, validation and test format

## Prepare data to feed it to model

## Fit the model with train and validation

## Test the model with test

