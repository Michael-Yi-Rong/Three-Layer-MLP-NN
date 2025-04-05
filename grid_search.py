import numpy as np
import itertools
import os
import matplotlib.pyplot as plt
from utils import load_cifar10
from model import NeuralNetwork
from train import train

# parameter combinations
EPOCHS_LIST = [50]
BATCH_SIZE_LIST = [128, 256]
LEARNING_RATE_LIST = [0.01, 0.005]
DECAY_RATE_LIST = [1.0, 0.95]
REG_LAMBDA_LIST = [0.001]
HIDDEN_SIZE_LIST = [1280, 2560]
ACTIVATION_LIST = ['relu', 'sigmoid', 'tanh']
NUM_CLASSES = 10

np.random.seed(42)

os.makedirs("./models", exist_ok=True)
os.makedirs("./plots", exist_ok=True)

param_grid = list(itertools.product(EPOCHS_LIST, BATCH_SIZE_LIST, LEARNING_RATE_LIST,
                                    DECAY_RATE_LIST, REG_LAMBDA_LIST, HIDDEN_SIZE_LIST, ACTIVATION_LIST))

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

all_histories = {}

best_val_accuracy = 0
best_params = None
best_model = None

for params in param_grid:
    EPOCHS, BATCH_SIZE, LEARNING_RATE, DECAY_RATE, REG_LAMBDA, HIDDEN_SIZE, ACTIVATION = params
    param_str = f"E{EPOCHS}_B{BATCH_SIZE}_L{LEARNING_RATE}_D{DECAY_RATE}_R{REG_LAMBDA}_H{HIDDEN_SIZE}_A{ACTIVATION}"
    print(f"\nTraining with params: {params}")

    model = NeuralNetwork(input_size=32 * 32 * 3, hidden_size=HIDDEN_SIZE, output_size=10,
                          activation=ACTIVATION, reg_lambda=REG_LAMBDA)
    history = train(model, X_train, y_train, X_val, y_val, NUM_CLASSES, LEARNING_RATE, EPOCHS, BATCH_SIZE)
    all_histories[params] = history

    # np.savez(f'./models/{param_str}.npz', W1=model.W1, b1=model.b1, W2=model.W2, b2=model.b2)

    plt.figure(figsize=(12, 6), dpi=300)

    plt.suptitle(f"Training Results - {param_str}", fontsize=14, fontweight="bold")

    # loss curve
    plt.subplot(1, 2, 1)
    plt.plot(history["train_loss"], label="Train Loss")
    plt.plot(history["val_loss"], label="Val Loss", linestyle='dashed')
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title(f"Loss Curve")
    plt.legend()

    # accuracy curve
    plt.subplot(1, 2, 2)
    plt.plot(history["val_accuracy"], label="Validation Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.title(f"Validation Accuracy")
    plt.legend()

    plt.savefig(f'./plots/train/{param_str}.png')
    plt.close()

    # get the best model parameters
    val_accuracy = history["val_accuracy"][-1]
    if val_accuracy > best_val_accuracy:
        best_val_accuracy = val_accuracy
        best_params = params
        best_model = model

# save the best checkpoint
if best_model is not None:
    np.savez(f'./models/best_model.npz', W1=best_model.W1, b1=best_model.b1, W2=best_model.W2, b2=best_model.b2)

with open('./models/best_params.txt', 'w') as f:
    f.write(f"Best Parameters: {best_params}\n")
    f.write(f"Best Validation Accuracy: {best_val_accuracy}\n")

print(f"\nBest Parameters: {best_params}")
print(f"\nBest Validation Accuracy: {best_val_accuracy}")
print("\nTraining completed.")