import numpy as np

# Define the XOR Gate dataset
X = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])

y = [0, 1, 1, 0]  # XOR Gate outputs

# Activation function: Step function
def step_function(x):
    return 1 if x >= 0 else 0

# Define the manually chosen weights and biases for the hidden layer and output layer
# Hidden layer weights and bias
hidden_weights = np.array([[1, 1], [1, 1]])  # Two neurons in the hidden layer
hidden_bias = np.array([-0.5, -1.5])         # Bias for each hidden neuron

# Output layer weights and bias
output_weights = np.array([1, -1])           # Weights for the output neuron
output_bias = -0.5                           # Bias for the output neuron

# Forward pass function
def forward_pass(x):
    # Hidden layer computation
    hidden_input = np.dot(x, hidden_weights) + hidden_bias
    hidden_output = np.array([step_function(i) for i in hidden_input])

    # Output layer computation
    final_input = np.dot(hidden_output, output_weights) + output_bias
    final_output = step_function(final_input)

    return hidden_output, final_output

# Test the MLP on the XOR dataset
print("Testing MLP on XOR Gate:")
for i in range(len(X)):
    hidden_output, final_output = forward_pass(X[i])
    print(f"Input: {X[i]}, Hidden Layer Output: {hidden_output}, Final Output: {final_output}, Actual Output: {y[i]}")

