import numpy as np
from utils import load_cifar10
from model import NeuralNetwork

EPOCHS = 50
BATCH_SIZE = 128
LEARNING_RATE = 0.01
DECAY_RATE = 0.95
REG_LAMBDA = 0.001
HIDDEN_SIZE = 2560 # to change
ACTIVATION = 'relu' # 'relu', 'sigmoid', 'tanh'
NUM_CLASSES = 10

def test(model, X_test, y_test):
    y_test_onehot = np.eye(10)[y_test]

    _, A1 = model.forward_layer(X_test, model.W1, model.b1, activation=True)
    _, A2 = model.forward_layer(A1, model.W2, model.b2, activation=False)

    test_loss = model.compute_loss(y_test_onehot, A2)

    test_pred = np.argmax(A2, axis=1)
    test_accuracy = np.mean(test_pred == y_test)

    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")
    return test_loss, test_accuracy


if __name__ == "__main__":
    np.random.seed(42)

    file_folder = './data/cifar-10-batches-py'
    _, _, X_test, y_test = load_cifar10(file_folder)

    model = NeuralNetwork(input_size=32 * 32 * 3, hidden_size=HIDDEN_SIZE, output_size=10,
                          activation=ACTIVATION, reg_lambda=REG_LAMBDA)
    weights = np.load('./models/best_model.npz')
    model.W1, model_b1, model.W2, model.b2 = weights["W1"], weights["b1"], weights["W2"], weights["b2"]

    test_loss, test_accuracy = test(model, X_test, y_test)

    print("\nTesting completed.")
