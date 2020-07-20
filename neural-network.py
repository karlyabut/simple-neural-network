import numpy as np

#normalizing function that returns any value between 0 - 1
def sigmoid(x):
  return 1 / (1 + np.exp(-x))

training_inputs = np.array([[0,0,1],
                           [1,1,1],
                           [1,0,1],
                           [0,1,1]])

training_outputs = np.array([[0,1,1,0]]).T

np.random.seed(1)

#create 3x1 because we have 3 input and 1 output, random values from -1 to 1
synaptic_weights = 2 * np.random.random((3, 1)) - 1

print('Random starting synaptic weights: ')
print(synaptic_weights)


for iteration in range(1):
  
  input_layer = training_inputs

  outputs = sigmoid(np.dot(input_layer, synaptic_weights))

print('Outputs after training: ')
print(outputs)