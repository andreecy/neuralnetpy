import numpy as np

# return value between -1 ~ 1
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

training_inputs = np.array([
    [0, 0, 1],
    [1, 1, 1],
    [1, 0, 1],
    [0, 1, 1]
])

training_outputs = np.array([[0, 1, 1, 0]]).T
print("Trainig Data:")
print("Input:")
print(training_inputs)
print("Output:")
print(training_outputs)
print()

synapse_weights = 2 * np.random.random((3,1)) - 1
# print("synapse_weights start")
# print(synapse_weights)

print("Training process...")
for i in range(10000):
    input_layer = training_inputs
    # sum (input * weights), and return with sigmoid function
    outputs = sigmoid(np.dot(input_layer, synapse_weights))

    error = training_outputs - outputs
    # gradient descent, find best weights
    adjustments = error * sigmoid_derivative(outputs)
    synapse_weights += np.dot(input_layer.T, adjustments)

print("Weights after training:")
print(synapse_weights)

print("Outputs after training:")
print(outputs)

print()
question = np.array([[1, 1, 0]])
print("Question:")
print(question)
answer = sigmoid(np.dot(question, synapse_weights))
print("Answer:")
print(answer)