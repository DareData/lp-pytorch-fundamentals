# Lesson 03: ML with your own data

In this lesson, you'll learn how to use your own tabular data to train a model. You've seen a little bit of this in Assignment 1, where you had to load your data from CSV files and do a few simple pre-processing steps before feeding it to the linear model, while most examples in the lessons up until now were using synthetic data just for learning purposes. 

In real world scenarios though, you work with custom (and sometimes very diverse) data and you need to load and pre-process it to make it useful. So, in this lesson, you'll see what tools PyTorch has that can help you with that.

Before that though, you need to learn about a concept you may have heard of before: **batches**.

## Batches

If you've never heard of them, batches are just subsets of your data. For example, if you have 1000 samples, you can split them into 10 batches of 100 samples each. There are several reasons to do that, most of them highlighted [in this short article](https://medium.com/analytics-vidhya/when-and-why-are-batches-used-in-machine-learning-acda4eb00763).

This short introduction is necessary as you'll see batches being talked about in the next section and it's something you'll use when working with significant amounts of data.

## Preparing Data with PyTorch

A significant part of the work you'll do in ML is preparing your data for training. This includes loading it from files, transforming it into tensors and then creating batches. PyTorch has a few tools that can help you with that, namely the Dataset and DataLoader classes.

As you might have noticed, a lot of the PyTorch standard is to create classes that inherit from the base PyTorch classes and then override some of their methods. This is also the case for the Dataset class, which you can use to create your own dataset class.

The benefit is that, when creating a Dataset class that wraps your data, all you'll have to do before training is instantiate the class and feed it to the model, as PyTorch knows how to handle the rest in the background.

That said, read the following article, which will teach you all about this data pre-processing pipeline in PyTorch, and try the code yourself in the [file](./data_prep.ipynb), <ins>but only</ins> for sections *Data preparation – the simplest scenario* and *How to create your own Dataset?*. Ignore the *Preparing the conda environment* section (not relevant for us) <ins>and</ins> the *Retrieving data from files* section, which deals with a huge data source that's uncessary for this lesson.

[Data preparation with Dataset and DataLoader in PyTorch](https://aigeekprogrammer.com/data-preparation-with-dataset-and-dataloader-in-pytorch/)

After this lesson, you should now know how to build a Dataset class and how to use it to feed your data to the model for training. Now, you're ready to do <ins>Assignment 2</ins>, where you'll use the knowledge from the previous lesson and this one to build a classifying neural network whilst also preparing your data to train it.

## Data Splitting

In the article above, you've seen how to create a Dataset class and how to use it to feed your data to the model. However, you may have noticed that the article didn't cover how to split your data into training and testing sets, which, as you have seen previously, is essential. So, how would you do that with PyTorch? Luckily, there's a function for that called `random_split`, which you can use to split your data into training and testing sets. Check out [this short example](https://www.projectpro.io/recipes/split-dataset-pytorch) on how to use the function. 

The first answer to [this Stack Overflow question](https://stackoverflow.com/questions/50544730/how-do-i-split-a-custom-dataset-into-training-and-test-datasets) is also very clear and it shows how you'd split the dataset with percentages rather than absolute numbers. For more information, check the [documentation page](https://pytorch.org/docs/stable/data.html#torch.utils.data.random_split) for the function.

## Assignment 2

If you feel like you didn't put much into practice in this lesson, don't worry, you'll do it in the assignment. In it, you'll have to build a neural network to classify if a patient has had a heart attack or not based on their medical data, while also creating a Dataset prior to that. So, go ahead and check it out!

## Extra Resources

If you wish to to explore the part about loading images from a folder, which you didn't to in the article above, check out this [PyTorch Documentation Tutorial](https://pytorch.org/tutorials/beginner/data_loading_tutorial.html) on this topic, altough some of the things you may find redundant with the article above. Besides this one, [the following article](https://medium.com/analytics-vidhya/creating-a-custom-dataset-and-dataloader-in-pytorch-76f210a1df5d) also covers the topic of pre-processing images, so you can check it out as well.

This other PyTorch article also covers this topic so you can explore it as well, if you feel curious: [Datasets & DataLoaders](https://pytorch.org/tutorials/beginner/basics/data_tutorial.html).

Lastly, notice that the `random_split` function described above returns a `Subset` object, which is a subclass of the `Dataset` class. To access the actual dataset, you'll need to do `train_set.dataset` and to get the `features` Tensor, for example, you'll get it with `train_set.dataset.features` So, if you want to learn more about the Subset, take a look at [the documentation](https://pytorch.org/docs/stable/data.html#torch.utils.data.Subset).