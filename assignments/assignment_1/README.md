# Assignment 1: Simple Linear Regression

This assignment will challenge you to apply the things you've learned in chapters 00 and 01, i.e., the fundamentals of PyTorch. You will be asked to create a simple linear regression model to predict sales from advertising data. The assignment is composed of 3 parts:
- Preparing the data
- Creating the model
- Training and testing the model

Aside from this, you can complement your work with a few extras that'll help you grasp the concepts of ML with PyTorch better.

> **Note**: Before you start, make sure you have <ins>**set up your environment correctly**</ins> by following the instructions in the [setup file](../../setup.ipynb).

For this project, you'll need a few imports from PyTorch and other libraries. The essential ones are below, while others may be added in the Extras section:

```python
import torch
from torch import nn
import pandas as pd
from sklearn.model_selection import train_test_split
```

You can do this assignment in a Jupyter Notebook or using a few Python scripts. Although the latter is what is used in the real world, the former is more convenient for learning purposes and should be easier for you to see what's going on.

## 1. Preparing the data

The data you will use is a shows, in each row, the advertising budget for a product in three different media (TV, radio, newspaper) and the sales for that product. The data is stored in a CSV file in `../data/assignment_1/advertising.csv`. It has 200 rows and one column for each type of advertising media and one column for sales. All values are floating point numbers and all cells are filled.

To prepare the data for training, you will need to perform a few steps:

1. Read the CSV file with the pandas library and print it to see what it looks like.

2. Transform the DataFrame into a PyTorch Tensor with torch.tensor(). Make sure the datatype is `float32`. Hint: use the `.values` attribute of the Pandas DataFrame.

3. Using Tensor indexing, separate the last column, which is the sales and corresponds to the target variable **y**, from the rest of the data, which corresponds to the features **X**.

4. Using the `train_test_split` function from sklearn, separate the data into training and testing sets. Use a test size of 0.2 and a random state of 42. Information on how to use this function can be found [here](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html).

After these steps, you should have 4 tensors, `X_train, X_test, y_train, y_test`, each with the corresponding data. Print the shapes of each tensor to make sure the number of rows in each tensor is correct, which should be 160 for training and 40 for testing.

## 2. Creating the model

Now, it's time to create a linear regression model like you've seen in Chapter 01. The following steps should be implemented:

1. Create a `LinearRegression()` class as you've learned before. The regression "layer" can be implemented with PyTorch's `nn.Linear` class **OR** by using `nn.Parameter`, as seen in the guide book. The input size should be 3, corresponding to the number of features, and the output size should be 1, corresponding to the number of outputs.

2. Define some hyperparameters for training: the number of epochs and the learning rate of the optimizer. You can use 1000 epochs and a learning rate of 0.01 to start with, but feel free to experiment with other values.

3. Define a loss function and an optimizer. You can use the [Mean Squared Error (MSE) loss](https://pytorch.org/docs/stable/generated/torch.nn.MSELoss.html) and the Adam or SGD optimizers. Feel free to experiment with other loss functions and optimizers.

4. Instantiate a model of the class you created and print it to see what it looks like.

## 3. Training and testing the model

Now, that you have a model, it's time to create the training loop and train the model. To achieve this, you must follow these steps:

1. Create a training loop that iterates over the number of epochs. For each epoch, you should:
	- Make a prediction with the model
	- Calculate the loss with the loss function
	- Clear the optimizer's gradients
	- Backpropagate the loss
	- Update the parameters with the optimizer
	- Print the loss at each *n* epochs, depending on how many epochs you have. Seeing 10 to 20 loss values should be enough.

2. Run the training loop and see how the loss changes over time. If you've done everything correctly, the loss should decrease over time.

3. Re-instantiate the model to erase this training and add a testing loop to the training loop. The testing loop should , make a prediction of the test set with the model and calculate the test loss. Make sure you use `torch.no_grad()` or `torch.inference_mode()` to disable gradient calculation in this step. Also, change the loss printing so that it prints the training and testing loss at each epoch.

4. Run the training loop again and see how the loss changes over time. If you've done everything correctly, both the training and testing loss should decrease over time, with the testing loss being higher than the training loss, as expected.

5. Save the model with `torch.save()` and print the model parameters to see what they look like. Don't forget to save the model's state dictionary, not the model itself.

## Extras

If you've done everything correctly, you should have a working linear regression model. However, there are a few things you can do to improve your model and your understanding of PyTorch. We've provided a list of extra things you can do to understand the concepts better and improve your model. Feel free to do as many as you want:

### Visualization

One of the most important things you can do to understand your model is to visualize its training and performance. You can do this by plotting the loss over time, plotting the predictions vs the actual values, and checking the model parameters. You can do this with the following steps:

1. Plot the loss over time. To do this, you should create a list to store the both the training and test loss values at each epoch. Then, you can plot the list with matplotlib. To do this, import the package with the command `import matplotlib.pyplot as plt` and use `plt.plot()` to plot the list. You can also use `plt.xlabel()`, `plt.ylabel()` and `plt.legend()` to make the graph more readable.

2. Plot the predicted values vs. real values. For this, after training, you should make a prediction of the test set with the model and plot the values with matplotlib. You can use `plt.scatter()` to plot the values and `plt.plot()` to plot the line of best fit, and the same commands as before to make the graph more readable.

3. Check the model parameters with `model.parameters()` and analyze them. Since we're doing linear regression, the model tries to find the weights that best fit the line $y= w_0 x_0 + w_1 x_1 + w_2 x_2 + b$. The weights depend on the magnitude of the features, but usually, when weights are near zero, it means that the feature is not important for the model. You can check the weights and see which feature is more important for the model.

### Improve training

We've built a very basic linear regression model and are using few samples to train it. However, there are a few things you can do to improve the training process and the model's performance. If you want to improve your model, you can try the following:

1. Scale your features with Min-Max or Standardization. This is a very important step in ML, as many times your features have very different order of magnitude among them, and standardizing the numerical values helps stabilize the calculations the model performs. You can find more information by reading [this article](https://medium.com/@soniaman809/feature-scaling-in-machine-learning-regularization-and-normalization-40d1091a45f8) and [this latter one](https://www.geeksforgeeks.org/ml-feature-scaling-part-2/), which is more code-driven and shows you how to do it.

2. Experiment with other loss functions, optimizers, and hyperparameters. There are many loss functions and optimizers you can use, and each one has its advantages and disadvantages. You can explore other [loss functions](https://pytorch.org/docs/stable/nn.html#loss-functions) and  [optimizers](https://pytorch.org/docs/stable/optim.html) and try to see which ones are adequate for the type of task we're doing. You can also experiment with other hyperparameters, such as the learning rate, the number of epochs, the batch size, etc.