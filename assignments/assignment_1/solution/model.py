import torch

class LinearRegressionModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(in_features=3, out_features=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)
    


def train_model(model: torch.nn.Module, X_train: torch.Tensor, y_train: torch.Tensor, num_epochs: int = 1000, learning_rate: float = 0.01) -> None:
    """Trains a model using the provided data."""
    loss_function = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    print("========================================")
    print("Training model.")
    print("========================================")
    print(model)
    print("========================================")
    model.train()
    for epoch in range(num_epochs):
        y_pred = model(X_train)
        loss = loss_function(y_pred, y_train)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if (epoch + 1) % 50 == 0:
            print(f'Epoch: {epoch + 1}, Loss: {loss:.2f}')
    print("========================================")


def train_and_validate_model(
        model: torch.nn.Module,
        X_train: torch.Tensor,
        y_train: torch.Tensor,
        X_test: torch.Tensor,
        y_test: torch.Tensor,
        num_epochs: int = 1000,
        learning_rate: float = 0.01
    ) -> None:
    """Trains and validates a model using the provided data."""
    loss_function = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    print("========================================")
    print("Training and Validating model.")
    print("========================================")
    # Create empty loss lists to track values
    train_loss_values = []
    test_loss_values = []
    epoch_count = []
    for epoch in range(num_epochs):
        model.train()
        y_pred = model(X_train)
        loss = loss_function(y_pred, y_train)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if (epoch + 1) % 50 == 0:
            model.eval()
            with torch.inference_mode():
                y_pred_test = model(X_test)
                test_loss = loss_function(y_pred_test, y_test)
                epoch_count.append(epoch)
                train_loss_values.append(loss.detach().numpy())
                test_loss_values.append(test_loss.detach().numpy())
                print(f'Epoch: {epoch + 1}, Loss: {loss:.2f}, Test Loss: {test_loss:.2f}')
    print("========================================")
    return train_loss_values, test_loss_values, epoch_count


def save_model(model: torch.nn.Module) -> None:
    """Saves a model to the specified path."""
    torch.save(model.state_dict(), "model.pth")
    print("========================================")
    print("Model saved as 'model.pth'.")
    print("========================================")
