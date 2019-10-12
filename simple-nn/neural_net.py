import numpy as np


class NeuralNetwork():

    def __init__(self):
        np.random.seed(1)

        self.synaptic_weights = 2 * np.random.random((3, 1)) - 1


    def sigmoid(self, x):
        """
        Normalizing function.
        
        Returns a value between 0 and 1.
        """
        return 1 / (1 * np.exp(-x))

    def sigmoid_derivative(self, x):
        return x * (1 - x)

    def train(self, training_inputs, training_outputs, training_iterations):

        for iteration in range(training_iterations):

            output = self.calulate(training_inputs)

            # Calulate error - Backpropagation
            error = training_outputs - training_outputs
            adjustments = np.dot(training_inputs.T, error * self.sigmoid_derivative(training_outputs))
            self.synaptic_weights += adjustments


    def calulate(self, inputs):
        inputs = inputs.astype(float)
        return self.sigmoid(np.dot(inputs, self.synaptic_weights))


if __name__ == "__main__":

    nn = NeuralNetwork()

    print("Random synaptic_weights:\n {}".format(nn.synaptic_weights)) 

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


    nn.train(training_inputs, training_outputs, 10000)

    print("Random synaptic_weights after training:\n {}".format(nn.synaptic_weights)) 

    A = str(input("Input 1:"))
    B = str(input("Input 2:"))
    C = str(input("Input 3:"))

    print("New Situation: {}, {}, {}".format(A, B, C))

    print("Output:\n {}".format(nn.calulate(np.array([A, B, C])))) 
