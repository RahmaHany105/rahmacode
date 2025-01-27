import numpy as np

X = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])

y = [0, 1, 1, 0]  

def step_function(x):
    return 1 if x >= 0 else 0

hidden_weights = np.array([[1, 1], [1, 1]])  
hidden_bias = np.array([-0.5, -1.5])         

output_weights = np.array([1, -1])          
output_bias = -0.5                           

def forward_pass(x):
    
    hidden_input = np.dot(x, hidden_weights) + hidden_bias
    hidden_output = np.array([step_function(i) for i in hidden_input])

    final_input = np.dot(hidden_output, output_weights) + output_bias
    final_output = step_function(final_input)

    return hidden_output, final_output

print("Testing MLP on XOR Gate:")
for i in range(len(X)):
    hidden_output, final_output = forward_pass(X[i])
    print(f"Input: {X[i]}, Hidden Layer Output: {hidden_output}, Final Output: {final_output}, Actual Output: {y[i]}")

