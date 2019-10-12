import numpy as np

def sigmoid(x):
    """
    Normalizing function.
    
    Returns a value between 0 and 1.
    """
    return 1 / (1 * np.exp(-x))

def sigmoid_derivative(x):
    """

    """
    return x * (1 - x)


training_inputs = np.array(
    [
        [0,0,1],
        [1,1,1],
        [1,0,1],
        [0,1,1]
    ]
)

# Transposted to become a 4x1 matrix
training_outputs = np.array(
    [
        [0, 1, 1, 0]
    ]
).T

# Initalize our weights:
np.random.seed(1)

# random values from -1 to 1
synaptic_weights = 2 * np.random.random((3, 1)) - 1

print("Random starting of synaptic weights:\n {}".format(synaptic_weights))

outputs = None
for iteration in range(20000):

    input_layer = training_inputs

    outputs = sigmoid(np.dot(input_layer, synaptic_weights))

    # Calulate error
    error = training_outputs - outputs

    adjustments = error * sigmoid_derivative(outputs)

    synaptic_weights += np.dot(input_layer.T, adjustments)


print("synaptic_weights after training:\n {}".format(synaptic_weights))

print("Outputs after training:\n {}".format(outputs))
