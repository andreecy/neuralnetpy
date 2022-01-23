from turtle import forward
import numpy as np

# return value between 0 and 1
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

class NeuralNetwork():
    def __init__(self, input_nodes: int, hidden_nodes: int, output_nodes: int):
        self.input_nodes = input_nodes
        self.hidden_nodes = hidden_nodes
        self.output_nodes = output_nodes

        # create synapse weights for input-hidden and hidden-output layer
        # random value between -1 and 1
        self.synapse0 = 2 * np.random.random((input_nodes, hidden_nodes)) - 1
        self.synapse1 = 2 * np.random.random((hidden_nodes, output_nodes)) - 1

        print("Weight0 start")
        print(self.synapse0)
        print("Weight1 start")
        print(self.synapse1)
        print("---------------")
        
        # create bias for hidden and output layer
        # random value between -1 and 1
        self.bias0 = 2 * np.random.random((1, hidden_nodes)) - 1
        self.bias1 = 2 * np.random.random((1, output_nodes)) - 1

        self.learning_rate = 0.1

    # Feed Forward algorithm
    def forward(self, inputs):
        # Generating the Hidden layer outputs
        # matrix dot product of input-hidden weight, add hidden bias
        hidden = np.dot(inputs, self.synapse0)
        # hidden = hidden + self.bias0
        
        # activation function
        hidden = sigmoid(hidden)

        # Generating the Output layer outputs
        # matrix dot product of hidden-output weight, add output bias
        outputs = np.dot(hidden, self.synapse1)
        # outputs = outputs + self.bias1
        # activation function
        outputs = sigmoid(outputs)
        return outputs
    
    # same as forward method
    def think(self, inputs):
        return self.forward(inputs)
        
    # same as forward method
    def predict(self, inputs):
        return self.forward(inputs)
    
    def train(self, inputs, targets):
        # Generating the Hidden layer outputs
        # matrix dot product of input-hidden weight, add hidden bias
        hidden = np.dot(inputs, self.synapse0)
        # hidden = hidden + self.bias0
        # activation function
        hidden = sigmoid(hidden)
        
        # Generating the Output layer outputs
        # matrix dot product of hidden-output weight, add output bias
        outputs = np.dot(hidden, self.synapse1)
        # outputs = outputs + self.bias1
        # activation function
        outputs = sigmoid(outputs)

        # calculate errors
        # ERROR = TARGETS - OUTPUT
        output_errors = targets - outputs
        # Calculate Gradient descent, find best hidden-output weights
        gradient = sigmoid_derivative(outputs) * output_errors
        # gradient = gradient * self.learning_rate
        weight_ho_delta = np.dot(hidden.T, gradient)
        # adjust synapse weight hidden-output with delta
        self.synapse1 += weight_ho_delta
        # adjust the bias
        # self.bias1 = np.add(self.bias1, gradient)
        # self.bias1 = self.bias1 + gradient

        # Calculate hidden layer errors
        hidden_errors = output_errors * self.synapse1.T
        # Calculate hidden gradient
        hidden_gradient = sigmoid_derivative(hidden) * hidden_errors
        # hidden_gradient = hidden_gradient * self.learning_rate
        # Calculate input-hidden deltas
        weight_ih_delta = np.dot(inputs.T, hidden_gradient)
        # adjust synapse weight input-hidden with delta
        self.synapse0 += weight_ih_delta
        # adjust the bias
        # self.bias0 = np.add(self.bias0, hidden_gradient)
        # self.bias0 = self.bias0 + hidden_gradient