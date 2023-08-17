### Assignment Plan
1. Read CSV with pandas
2. Transform DF into PyTorch Tensor with datatype float32
3. Separate the sales (y) from the rest of the data (X) with PyTorch indexing
4. Separate the data into training and testing sets with sklearn
5. Create a linear regression model with PyTorch (nn.Linear)
6. Define a loss function (MSE) and an optimizer (Adam or SGD)
7. Define hyperparameters learning rate, epochs (batch size later)
8. Create training loop and train
9. Create testing loop and train+test
10. Save model

### Extras

#### Visualization
1. Plot loss over time (record loss at each epoch)
2. Plot predictions vs actual
3. Check the model parameters with `model.parameters()` and analyze (higher = more important feature)

#### Improve training
1. Use TensorDataset and DataLoader to create batches
2. Normalize the data with Min-Max or Standardization
3. Experiment with other loss functions, optimizers, and hyperparameters

