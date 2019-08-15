import json
import numpy as np

class NN():
    def __init__(self, layersize = [1, 1]):
        np.random.seed(1)

        if len(layersize) < 2:
            raise Exception('Invalid Parameters')

        self.weights = []
        for i in range(0, len(layersize)-1):
            if layersize[i] < 1 or layersize[i+1] < 1:
                raise Exception('Invalid Parameters')
            self.weights.append(2 * np.random.random((layersize[i], layersize[i+1])) - 1)

    def sigmoid(self, s):
        return 1/(1+np.exp(-s))

    def sigmoidDerive(self, s):
        return s * (1 - s)

    def save(self, filename):
        try:
            data = {}
            data['weights'] = self.weights
            with open(filename, 'w') as outfile:
                json.dump(data, outfile)
        except Exception as e:
            print(e)
            return False
        return True

    def load(self, filename):
        try:
            with open(filename) as f:
                data = json.load(f)
            self.weights = data['weights']
        except Exception as e:
            print(e)
            return False
        return True

    def forward(self, inputs):
        out = []
        for w in self.weights:
            if len(out) == 0:
                out.append(self.sigmoid(np.dot(inputs.astype(float), w)))
            else:
                out.append(self.sigmoid(np.dot(out[-1].astype(float), w)))
        return out

    def backward(self, inputs, results, outputs):
        deltas = None
        for i in range(len(results)-1, -1, -1):
            if i == len(results)-1:
                error = outputs - results[i]
                deltas = error*self.sigmoidDerive(results[i])
            else:
                error = delta.dot(self.weights[i+1].T)
                deltas = error*self.sigmoidDerive(results[i])
            if i == 0:
                self.weights[i] += inputs.T.dot(deltas)
            else:
                self.weights[i] += results[i-1].T.dot(deltas)

    def train(self, inputs, outputs, n):
        for i in range(0, n):
            results = self.forward(inputs)
            self.backward(inputs, results, outputs)


nn = NN([3, 2, 1])

print("Random starting synaptic weights: ")
print(nn.weights)

# The training set, with 4 examples consisting of 3
# input values and 1 output value
training_inputs = np.array([[0,0,1],
                            [1,1,1],
                            [1,0,1],
                            [0,1,1]])

training_outputs = np.array([[0,1,1,0]]).T

# Train the neural network
nn.train(training_inputs, training_outputs, 10000)

print("Synaptic weights after training: ")
print(nn.weights)
print("Output: ")
print(nn.forward(np.array([0,0,1])))