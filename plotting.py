import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

def weight_hist(W, name):
    save_dir = "./plots/hist"
    os.makedirs(save_dir, exist_ok=True)
    plt.figure(dpi=300, figsize=(12, 8))
    plt.hist(W.flatten(), bins=50, alpha=0.7, color="b")
    plt.title(f"Histogram of {name}", fontsize=20)
    plt.xlabel("Weight Value", fontsize=18)
    plt.ylabel("Frequency", fontsize=18)
    plt.savefig(os.path.join(save_dir, f"{name}.png"), dpi=300, bbox_inches="tight")
    plt.close()

def weight_heatmap(W, name):
    save_dir = "./plots/heatmap"
    os.makedirs(save_dir, exist_ok=True)
    plt.figure(dpi=300, figsize=(12, 8))
    sns.heatmap(W, cmap="coolwarm", center=0)
    plt.title(f"Heatmap of {name}", fontsize=20)
    plt.xlabel("Neurons", fontsize=18)
    plt.ylabel("Input Features (Flattened Pixels)", fontsize=18)
    plt.savefig(os.path.join(save_dir, f"{name}.png"), dpi=300, bbox_inches="tight")
    plt.close()

data = np.load("./models/best_model.npz")
print(data.files) # ['W1', 'b1', 'W2', 'b2']

W1 = data["W1"]  # 输入层到隐藏层的权重
b1 = data["b1"]  # 输入层到隐藏层的偏置
W2 = data["W2"]  # 隐藏层到输出层的权重
b2 = data["b2"]  # 隐藏层到输出层的偏置

weight_hist(W1, 'W1')
weight_hist(b1, 'b1')
weight_hist(W2, 'W2')
weight_hist(b2, 'b2')

weight_heatmap(W1, 'W1')
weight_heatmap(b1, 'b1')
weight_heatmap(W2, 'W2')
weight_heatmap(b2, 'b2')