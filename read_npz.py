import numpy as np

data = np.load("./models/best_model.npz")
print(data.files) # ['W1', 'b1', 'W2', 'b2']

W1 = data["W1"]  # 输入层到隐藏层的权重
b1 = data["b1"]  # 输入层到隐藏层的偏置
W2 = data["W2"]  # 隐藏层到输出层的权重
b2 = data["b2"]  # 隐藏层到输出层的偏置

# print("W1:", W1)
# print("b1:", b1)
# print("W2:", W2)
# print("b2:", b2)

print("W1 shape:", W1.shape)
print("b1 shape:", b1.shape)
print("W2 shape:", W2.shape)
print("b2 shape:", b2.shape)
