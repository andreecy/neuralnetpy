import numpy as np
from neuralnetwork import NeuralNetwork

model = NeuralNetwork(3, 2, 1)
inputs = np.array([
    [0, 0, 1],
    [0, 1, 0],
    [1, 0, 1],
    [1, 1, 0],
])
targets = np.array([[0, 0, 1, 1]]).T

print("Training Data")
print(inputs)
print("Training Target")
print(targets)

for i in range(20000):
    model.train(inputs, targets)

print("weights0 after tranining")
print(model.synapse0)
print("weights1 after training")
print(model.synapse1)

print("target output after training")
output = model.forward(inputs)
print(output)

print("---------------")
# Question
question = np.array([[0, 1, 1 ]])
print("Question:")
print(question)
# Prediction
prediction = model.think(question)
print("Prediction:")
print(prediction)