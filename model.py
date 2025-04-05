import numpy as np

class NeuralNetwork:
    def __init__(self, input_size=32*32*3, hidden_size=256, output_size=10,
                 activation='relu', reg_lambda=0.01):
        """
        parameter initialization
        input_size: 输入层大小
        hidden_size1: 隐藏层大小
        output_size: 输出层大小
        activation: 使用的激活函数
        reg_lambda: L2正则化强度
        """

        np.random.seed(42)

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.activation = activation
        self.reg_lambda = reg_lambda

        # weights initialization
        self.W1 = np.random.randn(input_size, hidden_size) * np.sqrt(2.0 / input_size)
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, output_size) * np.sqrt(2.0 / hidden_size)
        self.b2 = np.zeros((1, output_size))

        # activation function selection
        if activation == 'relu':
            self.act_fn = lambda x: np.maximum(0, x)
            self.act_fn_grad = lambda x: (x > 0).astype(float)
        elif activation == 'sigmoid':
            self.act_fn = lambda x: np.where(x >= 0, 1 / (1 + np.exp(-x)), np.exp(x) / (1 + np.exp(x)))
            self.act_fn_grad = lambda x: self.act_fn(x) * (1 - self.act_fn(x))
        elif activation == 'tanh':
            self.act_fn = lambda x: np.tanh(np.clip(x, -50, 50))
            self.act_fn_grad = lambda x: 1 - np.clip(x, -1, 1) ** 2
        else:
            raise ValueError("Unsupported activation function!")

    def softmax(self, Z):
        expZ = np.exp(Z - np.max(Z, axis=1, keepdims=True)) # avoid overflow
        return expZ / np.sum(expZ, axis=1, keepdims=True)

    def forward_layer(self, X, W, b, activation=True):
        Z = X.dot(W) + b
        A = self.act_fn(Z) if activation else self.softmax(Z)
        return Z, A

    def backward_layer(self, dA, W, Z, A_prev, activation=True):
        dZ = dA * self.act_fn_grad(Z) if activation else dA
        dW = A_prev.T.dot(dZ) / A_prev.shape[0] + self.reg_lambda * W / A_prev.shape[0]
        db = np.sum(dZ, axis=0, keepdims=True) / A_prev.shape[0]
        dA_prev = dZ.dot(W.T)
        return dA_prev, dW, db

    def compute_loss(self, y_true, y_pred):
        N = y_true.shape[0]
        loss = -np.sum(y_true * np.log(y_pred + 1e-9)) / N # cross entropy loss
        reg_loss = (self.reg_lambda / (2 * N)) * (
                np.sum(np.clip(self.W1 ** 2, -1e10, 1e10)) +
                np.sum(np.clip(self.W2 ** 2, -1e10, 1e10))
        ) # reg loss
        return loss + reg_loss