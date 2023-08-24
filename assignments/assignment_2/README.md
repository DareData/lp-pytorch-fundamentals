# Assignment 2: Heart Attack Prediction with Neural Networks

In this assignment, your task will be to create a neural network that takes in patient data and predicts whether they have/had a heart attack or not. Like the previous one, this assignment has 3 parts:
- Creating the Dataset class
- Creating the Model class
- Training and testing the model

The assignment will guide you step by step through the whole process, with some hints and tips along the way to make sure you're in the right track.
After you're done with this main part, you can try to improve your model with a few extras that are listed at the bottom of this page.


## 1. Processing the data with PyTorch

The data you will use is a dataset of patients with heart disease. It was taken from [Kaggle](https://www.kaggle.com/datasets/bharath011/heart-disease-classification-dataset) and it's stored in a CSV file in the [data/assignment_2](../data/assignment_2/) folder. It has 1319 rows and 9 columns - 8 numerical medical data and one classification label, the presence of a heart attack (verify this when you load the data from the file with pandas).

If you wish to visualize the data, just create a Notebook, import pandas, read the CSV file and print the DataFrame. You can also use the `describe()` method to get some statistics about the data.

To prepare the data for training, you will need to build a custom Dataset class that will read the CSV file and return the data in the form of PyTorch Tensors. This class should be implemented in the [dataset.py](dataset.py) file. The layout is built for you already.
To do so, the following steps should be implemented:

1. Load the data from the CSV file with pandas' `read_csv` function.

2. Transform the `class` column's values from `negative, positive` to `0, 1` respectively, using the `map` function.

3. Transform the dataframe into a PyTorch Tensor with `torch.tensor()`, making sure the datatype is `float32` (and using the `.values` attribute).

4. With Tensor indexing, separate the last column, the labels, from the rest of the data, which corresponds to the features.

5. Define the `__len__` method to return the number of rows in the dataset.

6. Define the `__getitem__` method to return a tuple of the features and the label for a given index.

After these steps, you should have a Dataset class that can be used to create a Dataset object. You can test it by creating a Dataset object and printing its length and the first item with the `__getitem__` method.

## 2. Creating the model

Now, it's time to create a neural network model like you've seen in Chapter 02. Initially, we want a network with the following architecture:
- 1 input layer with *N* inputs and 16 outputs (where *N* is the number of features)
- 1 hidden layer with 16 inputs and 8 outputs
- 1 output layer with 8 inputs and M outputs (where *M* is the number of classes)
- ReLU activation functions for each of the layers except the output layer, which will return logits (raw values)

The model class should be implemented in the [model.py](model.py) file, where you already have the blueprint. The following steps should be implemented:

1. In the `__init__` method, define the layers of the model. You can use the `nn.Linear` class to define the layers. Make sure you use the correct input and output sizes for each layer.

2. In the `forward` method, define the forward pass of the model. You can use the `nn.ReLU` class to define the activation function.

After these steps, you should have a model class that can be used to create a model object. You can test it by creating a model object and printing it to see what it looks like or printing the model's parameters.

## 3. Training and testing the model

Now that you created your dataset and model classes, it's time to use the data to train the model. You can do this in a Jupyter Notebook or using a Python script, whichever you prefer. Don't forget to import the classes you created in the previous steps.

The following steps should be implemented:

1. Instantiate the dataset class and print its shape to make sure it's correct.

2. Instantiate the model class and print it to make sure it's correct.

3. Define the loss function and the optimizer. You should use Binary Cross Entropy with Logits (BCEWithLogitsLoss) and the Adam optimizer. Feel free to experiment with other loss functions and optimizers, but beware that if you want to use a different loss function, you will need to change the activation function of the output layer to a sigmoid function.

4. Define an accuracy function, as you've seen in the book in Lesson 02.

5. Define the epochs and batch size hyperparameters. You can use 1000 epochs and a batch size of 32 to start with, but feel free to experiment with other values. The most common batch sizes for small datasets like ours are 8, 16, 32 and 64 but it's up to you.

6. Using the `random_split` function, create the training and testing subsets as seen in lesson 03. You can use 80% of the data for training and 20% for testing.

7. Create the training and testing data loaders with the `DataLoader` class with the batch size defined prior (make sure you enable shuffling).

8. Create the training loop. To do this with batches instead of just epochs, check this [documentation example](https://pytorch.org/tutorials/beginner/introyt/trainingyt.html#the-training-loop), which is very similar to what you need to do. 
Before each batch loop, set the model to train mode and define a training loss variable and an accuracy variable, that will store the loss over one epoch and the accuracy. For each batch, you should:
- Make a prediction with the model (remember to squeeze)
- Since the output comes in logits, apply sigmoid and rounding to get the predicted class
- Calculate the loss with the loss function (remember to pass the logits) and add it to the training loss variable
- Calculate the accuracy with the accuracy function and add it to the accuracy variable
- Clear the optimizer's gradients
- Backpropagate the loss
- Update the parameters with the optimizer

After each epoch, you should:
- Divide the training loss and accuracy by the number of batches to get the average loss and accuracy over the epoch
- With `inference_mode()`, create a testing loop that does the same as the training loop, but without backpropagation and with the testing data loader
- Print the training loss and accuracy, and testing loss and accuracy every 50 epochs or so (change at will)

9. Run the training loop and see how the loss and accuracy change over time. If you've done everything correctly, the loss should decrease and the accuracy should increase over time, getting close or over 90% accuracy.

## Extras

If you followed the steps above, you should have a working model that can predict whether a patient has/had a heart attack or not. However, there are a few things you can do to improve your model and get a better understanding of what you're building. Feel free to try as many of these extras as you want.

### Visualizing the training

As you've done in assignment 1, you can add a visualization of the training process. You can use the `matplotlib` library to plot the loss and accuracy over time. You can also plot the loss and accuracy for the training and testing sets in the same plot, to see how they compare. For this, you'll need to store the loss and accuracy at every epoch OR every batch, depending on how detailed you want your plot to be.

### Improving the model

There are a few things you can do to improve your model. You can try to change the number of layers, the number of neurons in each layer, the activation functions, etc. What we challenge you to do is change your model class to make it more flexible. You can do this by adding parameters to the `__init__` method that allow you to change the number of layers, the number of neurons in each layer, the activation functions, etc. This way, you can create different models with different architectures and test them to see which one performs better.

For example, you could make the `__init__` method accept an arbitrary list of integers, each one corresponding to the number of neurons in a layer. That is, to create a network with 4 layers of 100, 80, 60, 40 neurons, you'd input `[100, 80, 60, 40]` when instantiating the class. To create a dynamic list of layers inside a model instead of just creating `layer1, layer2, layer3`, etc., you can use PyTorch's [ModuleList](https://pytorch.org/docs/stable/generated/torch.nn.ModuleList.html), which allows you to get around this. You can also explore the [Sequential](https://pytorch.org/docs/stable/generated/torch.nn.Sequential.html) class to create a model with a dynamic number of layers.

Another thing you can do is accept a parameter that defines which activation function to use. This parameter can be text and you can do an if-else chain to check which activation function to use and default to one of them if the input is invalid.

### Improving the training

There are also a few things you can do to improve your training process. You can try to change the loss function, the optimizer, the batch size, the learning rate, and, especially, the architecture of the model, which will be very easy if you implement the suggestions above.

However, you can also normalize your data. As you may have noticed when you printed the data, some columns have numbers hundreds of times larger than others. This can cause problems when training the model, so it's a good idea to normalize the data. You can do this, once again, through Min-Max or Standardization, for example, both of which were described in the Extras section of [Assignment 1](../assignment_1/README.md). It would be a good idea to normalize during the data processing you do in the Dataset class, so that the data is already normalized when you train the model.

