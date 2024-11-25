import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import os
from matplotlib.patches import Circle
from functools import partial
import matplotlib

matplotlib.use('Agg')
result_dir = "results"
os.makedirs(result_dir, exist_ok=True)

# Define a simple MLP class
class MLP:
    def __init__(self, input_dim, hidden_dim, output_dim, lr, activation='tanh'):
        np.random.seed(0)
        self.lr = lr  # Learning rate
        self.activation_fn = activation  # Activation function
        # TODO: define layers and initialize weights
        self.W1 = np.random.randn(input_dim, hidden_dim) * np.sqrt(2 / input_dim)
        self.b1 = np.zeros((1, hidden_dim))
        self.W2 = np.random.randn(hidden_dim, output_dim) * np.sqrt(2 / hidden_dim)
        self.b2 = np.zeros((1, output_dim))

    def forward(self, X):
        # TODO: forward pass, apply layers to input X
        # TODO: store activations for visualization
        # Compute hidden layer input
        self.Z1 = np.dot(X, self.W1) + self.b1
        # Apply activation function for the hidden layer
        activation_functions = {
            'tanh': np.tanh(self.Z1),
            'relu': np.maximum(0, self.Z1),
            'sigmoid': np.clip(1 / (1 + np.exp(-self.Z1)), 1e-7, 1 - 1e-7)
        }
        if self.activation_fn in activation_functions:
            self.A1 = activation_functions[self.activation_fn]
        else:
            raise ValueError(f"Invalid activation function: {self.activation_fn}")
        # Compute output layer input
        self.Z2 = np.dot(self.A1, self.W2) + self.b2
        # Apply sigmoid activation for the output layer
        out = np.clip(1 / (1 + np.exp(-self.Z2)), 1e-7, 1 - 1e-7)
        return out


    def backward(self, X, y):
        # TODO: compute gradients using chain rule
        # TODO: update weights with gradient descent
        # TODO: store gradients for visualization
        # Number of samples
        m = X.shape[0]
        # Compute predictions (sigmoid output for the final layer)
        out = np.clip(1 / (1 + np.exp(-self.Z2)), 1e-7, 1 - 1e-7)
        # Compute gradients for the output layer
        dZ2 = out - y
        dW2 = np.dot(self.A1.T, dZ2) / m
        db2 = np.sum(dZ2, axis=0, keepdims=True) / m
        # Backpropagation through the activation function
        if self.activation_fn == 'tanh':
            dA1 = (1 - np.tanh(self.Z1) ** 2) * np.dot(dZ2, self.W2.T)
        elif self.activation_fn == 'relu':
            dA1 = (self.Z1 > 0).astype(float) * np.dot(dZ2, self.W2.T)
        elif self.activation_fn == 'sigmoid':
            sigmoid = np.clip(1 / (1 + np.exp(-self.Z1)), 1e-7, 1 - 1e-7)
            dA1 = sigmoid * (1 - sigmoid) * np.dot(dZ2, self.W2.T)
        else:
            raise ValueError(f"Invalid activation function: {self.activation_fn}")
        # Compute gradients for the hidden layer
        dW1 = np.dot(X.T, dA1) / m
        db1 = np.sum(dA1, axis=0, keepdims=True) / m
        # Update weights and biases using gradient descent
        self.W2 -= self.lr * dW2
        self.b2 -= self.lr * db2
        self.W1 -= self.lr * dW1
        self.b1 -= self.lr * db1

def generate_data(n_samples=100):
    np.random.seed(0)
    X = np.random.randn(n_samples, 2)
    y = (X[:, 0] ** 2 + X[:, 1] ** 2 > 1).astype(int).reshape(-1, 1)
    return X, y

def update(frame, mlp, ax_input, ax_hidden, ax_gradient, X, y):
    ax_hidden.clear()
    ax_input.clear()
    ax_gradient.clear()
    
    # Perform 10 steps of training per update
    for _ in range(10):
        mlp.forward(X)
        mlp.backward(X, y)
    
    # Hidden space plot
    hidden_features = mlp.A1
    ax_hidden.scatter(hidden_features[:, 0], hidden_features[:, 1], y.ravel(), c=y.ravel(), cmap='bwr', alpha=0.7)
    x_hidden = np.linspace(hidden_features[:, 0].min(), hidden_features[:, 0].max(), 50)
    y_hidden = np.linspace(hidden_features[:, 1].min(), hidden_features[:, 1].max(), 50)
    xx_hidden, yy_hidden = np.meshgrid(x_hidden, y_hidden)
    grid_hidden = np.c_[xx_hidden.ravel(), yy_hidden.ravel()]
    zz_hidden = mlp.forward(grid_hidden).reshape(xx_hidden.shape)
    ax_hidden.plot_surface(xx_hidden, yy_hidden, zz_hidden, cmap='bwr', alpha=0.5)
    ax_hidden.set_title(f"Hidden Space at Step {frame * 10}")
    ax_hidden.set_xlabel('Hidden Feature 1')
    ax_hidden.set_ylabel('Hidden Feature 2')
    ax_hidden.set_zlabel('Output')
    
    # Input space plot with binary decision boundary
    x1_range = np.linspace(-3, 3, 200)  # Further extended range
    x2_range = np.linspace(-3, 3, 200)  # Further extended range
    xx1, xx2 = np.meshgrid(x1_range, x2_range)
    grid = np.c_[xx1.ravel(), xx2.ravel()]
    decision_input = (mlp.forward(grid) > 0.5).astype(int).reshape(xx1.shape)  # Binary predictions
    
    # Binary colormap and decision boundary
    binary_cmap = plt.cm.colors.ListedColormap(['blue', 'red'])
    ax_input.contourf(xx1, xx2, decision_input, levels=[-0.5, 0.5, 1.5], cmap=binary_cmap, alpha=0.7)  # Binary regions
    ax_input.contour(xx1, xx2, decision_input, levels=[0.5], colors='white', linewidths=2)  # White decision boundary
    ax_input.scatter(X[:, 0], X[:, 1], c=y.ravel(), cmap='bwr', edgecolor='k')
    ax_input.set_title(f"Input Space at Step {frame * 10}")
    
    # Gradient graph with weights visualized (formerly plot_network)
    nodes = {
        'x1': [0.2, 0.95], 
        'x2': [0.2, 0.55], 
        'h1': [0.6, 0.95], 
        'h2': [0.6, 0.65], 
        'h3': [0.6, 0.35], 
        'y': [0.9, 0.65]
    }
    edges = [
        ('x1', 'h1', mlp.W1[0, 0]), ('x1', 'h2', mlp.W1[0, 1]), ('x1', 'h3', mlp.W1[0, 2]),
        ('x2', 'h1', mlp.W1[1, 0]), ('x2', 'h2', mlp.W1[1, 1]), ('x2', 'h3', mlp.W1[1, 2]),
        ('h1', 'y', mlp.W2[0, 0]), ('h2', 'y', mlp.W2[1, 0]), ('h3', 'y', mlp.W2[2, 0])
    ]
    max_thickness = 10  # Maximum edge thickness
    min_thickness = 1  # Minimum edge thickness
    
    # Normalize weights to scale thickness proportionally
    all_weights = [abs(edge[2]) for edge in edges]
    max_weight = max(all_weights)
    min_weight = min(all_weights)
    
    for n1, n2, weight in edges:
        normalized_weight = (abs(weight) - min_weight) / (max_weight - min_weight)
        line_width = normalized_weight * (max_thickness - min_thickness) + min_thickness
        ax_gradient.plot(
            [nodes[n1][0], nodes[n2][0]], 
            [nodes[n1][1], nodes[n2][1]], 
            color='purple', 
            alpha=0.8, 
            linewidth=line_width
        )
    for node, (x, y) in nodes.items():
        ax_gradient.scatter(x, y, color='blue', s=400)  # Slightly larger blue points
        ax_gradient.text(x, y - 0.05, node, color='black', fontsize=12, ha='center', va='center')  # Labels below points
    ax_gradient.set_xlim(0, 1)
    ax_gradient.set_ylim(0, 1)
    ax_gradient.axis('off')
    ax_gradient.set_title(f"Gradients at Step {frame * 10}")

def visualize(activation, lr, step_num):
    X, y = generate_data()
    mlp = MLP(input_dim=2, hidden_dim=3, output_dim=1, lr=lr, activation=activation)
    fig = plt.figure(figsize=(21, 7))
    ax_hidden = fig.add_subplot(131, projection='3d')
    ax_input = fig.add_subplot(132)
    ax_gradient = fig.add_subplot(133)
    # Show only every 10th step in the GIF
    ani = FuncAnimation(fig, partial(update, mlp=mlp, ax_input=ax_input, ax_hidden=ax_hidden, ax_gradient=ax_gradient, X=X, y=y), frames=step_num // 10, repeat=False)
    ani.save(os.path.join(result_dir, "visualize.gif"), writer='pillow', fps=10)
    plt.close()

if __name__ == "__main__":
    activation = "tanh"
    lr = 0.1
    step_num = 1000
    visualize(activation, lr, step_num)