from matplotlib import pyplot as plt

def visualize_losses(train_loss_values, test_loss_values, epoch_count):
    plt.plot(epoch_count, train_loss_values, 'r--')
    plt.plot(epoch_count, test_loss_values, 'b-')
    plt.legend(['Training Loss', 'Test Loss'])
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.show()
    print("========================================")
    print("Visualized losses.")
    print("========================================")


def visualize_pred_vs_actual(y_pred, y_test):
    plt.scatter(y_test, y_pred)
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.show()
    print("========================================")
    print("Visualized predictions vs actual.")
    print("========================================")

def visualize_coefficients(coefs):
    plt.bar(range(len(coefs)), coefs)
    plt.xlabel('Coefficient Index')
    plt.ylabel('Coefficient Value')
    plt.show()
    print("========================================")
    print("Visualized coefficients.")
    print("========================================")