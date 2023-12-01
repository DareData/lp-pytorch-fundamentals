import data
import model
import sys
import torch
import visuals

if __name__ == "__main__":
    torch.manual_seed(42)
    path = sys.argv[1]
    try:
        X_train, X_test, y_train, y_test = data.load_data(path)
    except FileNotFoundError:
        raise FileNotFoundError("Provide a proper file path to the dataset as an argument to this script.")
    
    lr = model.LinearRegressionModel()
    model.train_model(lr, X_train, y_train)

    lr_test = model.LinearRegressionModel()
    train_loss_values, test_loss_values, epoch_count = model.train_and_validate_model(lr_test, X_train, y_train, X_test, y_test)

    model.save_model(lr_test)

    visuals.visualize_losses(train_loss_values, test_loss_values, epoch_count)
    y_pred = lr_test(X_test)
    visuals.visualize_pred_vs_actual(y_pred.detach().numpy(), y_test.detach().numpy())
    visuals.visualize_coefficients(lr_test.linear.weight.detach().squeeze().numpy())
