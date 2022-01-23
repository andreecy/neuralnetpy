# Neural Network
Just a basic Neural Network module

## Usage Example

### Importing Module
```python
from neuralnetwork import NeuralNetwork
import numpy as np
```
### Initiating
Initiate `NeuralNetwork` class with parameters: 

`NeuralNetwork(input_nodes: int, hidden_nodes: int, output_nodes: int)`
```python
model = NeuralNetwork(3, 2, 1)
```
Set 3 inputs nodes, 2 hidden nodes, and 1 output node


### Training
For this example, we want to train our machine with data below. Basically, our expected output are just the first number of inputs.

```
[0, 0, 1] = 0
[0, 1, 0] = 0
[1, 0, 1] = 1
[1, 1, 0] = 1
```

Set training data for inputs and target outputs, and train with `train(inputs, targets)` method through some number of iterations, until you statisfied. This example: `20000` iterations

```python
# Training data inputs
inputs = np.array([
    [0, 0, 1],
    [0, 1, 0],
    [1, 0, 1],
    [1, 1, 0],
])
# Training data target outputs
targets = np.array([[0, 0, 1, 1]]).T

# iterate to train
for i in range(20000):
    model.train(inputs, targets)
```

### Predicting
After a training above, now we test our machine to predict an output for given input. 

Example: `[0, 1, 1]` should produce `0`, because 0 is the first number.

```python
# Question
question = np.array([[0, 1, 1 ]])
print("Question:")
print(question)
# Prediction
prediction = model.think(question)
print("Prediction:")
print(prediction)
```

### Result
This is the terminal result
```
Training Data
[[0 0 1]
 [0 1 0]
 [1 0 1]
 [1 1 0]]
Training Target
[[0]
 [0]
 [1]
 [1]]
...
target output after training
[[0.00344796]
 [0.00344796]
 [0.99647399]
 [0.99647399]]
---------------
Question:
[[0 1 1]]
Prediction:
[[0.00332841]]
```
From `[0, 1, 1]` input, we predict: `0.003` (almost nearly zero) 

Woohoo! You just build your machine learning

How we know our accurracy?

The "target output after training" above should match nearly the "training target" actual data.

Full example is located in `examples/basic.py` file

## Project Structure
```
/
├── neuralnetwork.py    # core module
├── pyproject.toml
├── README.rst
├── example             # containing examples of usage
│   ├── basic.py        # basic usage example
│   └── __init__.py
└── tests
    ├── __init__.py
    └── *.py
```