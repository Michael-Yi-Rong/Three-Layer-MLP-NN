import numpy as np
from utils import load_cifar10
from model import NeuralNetwork

EPOCHS = 50
BATCH_SIZE = 128
LEARNING_RATE = 0.01
DECAY_RATE = 0.95
REG_LAMBDA = 0.001
HIDDEN_SIZE = 256
ACTIVATION = 'relu' # 'relu', 'sigmoid', 'tanh'
NUM_CLASSES = 10

def train(model, X_train, y_train, X_val, y_val, num_classes, learning_rate, epochs, batch_size):
    best_val_acc = 0.0
    history = {"train_loss": [], "val_loss": [], "val_accuracy": []}

    # training epochs
    for epoch in range(epochs):
        indices_ = np.arange(X_train.shape[0])
        np.random.shuffle(indices_)
        X_train, y_train = X_train[indices_], y_train[indices_]

        total_loss = 0.0
        num_batches = X_train.shape[0] // batch_size

        for i in range(num_batches):
            # each batch
            start, end = i * batch_size, (i + 1) * batch_size
            X_batch, y_batch = X_train[start:end], y_train[start:end]

            # forward
            Z1, A1 = model.forward_layer(X_batch, model.W1, model.b1, activation=True)
            Z2, A2 = model.forward_layer(A1, model.W2, model.b2, activation=False)

            # compute loss
            y_batch_onehot = np.eye(10)[y_batch]
            loss = model.compute_loss(y_batch_onehot, A2)
            total_loss += loss

            # backward propagation
            dA2 = A2 - y_batch_onehot
            dA1, dW2, db2 = model.backward_layer(dA2, model.W2, Z2, A1, activation=False)
            _, dW1, db1 = model.backward_layer(dA1, model.W1, Z1, X_batch, activation=True)

            # SGD
            model.W1 -= learning_rate * dW1
            model.b1 -= learning_rate * db1
            model.W2 -= learning_rate * dW2
            model.b2 -= learning_rate * db2

        # val
        Z1, A1 = model.forward_layer(X_val, model.W1, model.b1, activation=True)
        Z2, A2 = model.forward_layer(A1, model.W2, model.b2, activation=False)

        y_val_onehot = np.eye(num_classes)[y_val]
        val_loss = model.compute_loss(y_val_onehot, A2)
        val_accuracy = np.mean(np.argmax(A2, axis=1) == y_val)

        history["train_loss"].append(total_loss / num_batches)
        history["val_loss"].append(val_loss)
        history["val_accuracy"].append(val_accuracy)

        print(f"Epoch {epoch + 1}/{epochs}, Train Loss: {total_loss / num_batches:.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")

        learning_rate *= DECAY_RATE

    # np.savez('./models/model_weights.npz', W1=model.W1, b1=model.b1, W2=model.W2, b2=model.b2)
    return history


if __name__ == "__main__":
    np.random.seed(42)

    file_folder = './data/cifar-10-batches-py'
    X_train, y_train, _, _ = load_cifar10(file_folder)

    num_samples = X_train.shape[0]
    indices = np.arange(num_samples)
    np.random.seed(42)
    np.random.shuffle(indices)
    X_train, y_train = X_train[indices], y_train[indices]

    num_train = int(0.9 * num_samples)
    X_val, y_val = X_train[num_train:], y_train[num_train:]
    X_train, y_train = X_train[:num_train], y_train[:num_train]

    model = NeuralNetwork(input_size=32 * 32 * 3, hidden_size=HIDDEN_SIZE, output_size=10,
                          activation=ACTIVATION, reg_lambda=REG_LAMBDA)

    history = train(model, X_train, y_train, X_val, y_val, NUM_CLASSES, LEARNING_RATE, EPOCHS, BATCH_SIZE)

    print("\nTraining completed.")