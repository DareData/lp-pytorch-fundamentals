import torch
import pandas as pd
from sklearn.model_selection import train_test_split


def load_data(path):
    try:
        dataset = pd.read_csv(path)
    except FileNotFoundError:
        raise FileNotFoundError("Provide a proper file path to the dataset as an argument to this script.")
    print("========================================")
    print("Dataset loaded.")
    print("========================================")
    print(dataset.head())
    print("========================================")

    # Turn it into a torch tensor
    dataset = torch.tensor(dataset.values, dtype=torch.float32)
    y = dataset[:, -1].unsqueeze(1)
    X = dataset[:, :-1]

    print("========================================")
    print("Dataset converted to torch tensors.")
    print("========================================")
    print("X shape: ", X.shape)
    print("y shape: ", y.shape)
    print("========================================")

    # Split into train and test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print("========================================")
    print("Dataset split into train and test.")
    print("========================================")
    print("X_train shape: ", X_train.shape)
    print("y_train shape: ", y_train.shape)
    print("X_test shape: ", X_test.shape)
    print("y_test shape: ", y_test.shape)
    print("========================================")
    return X_train, X_test, y_train, y_test

