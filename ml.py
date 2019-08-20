import json
import numpy as np

class NN():
    def __init__(self, layersize = [1, 1]):
        """ Initialize the neural network """
        np.random.seed(1) # random seed

        if len(layersize) < 2:
            raise Exception('Invalid Parameters')

        self.weights = [] # our weights
        for i in range(0, len(layersize)-1):
            if layersize[i] < 1 or layersize[i+1] < 1:
                raise Exception('Invalid Parameters')
            self.weights.append(np.random.randn(layersize[i], layersize[i+1])) # initialize them with the numbers of inputs and outputs

    def sigmoid(self, s): # sigmoid function
        return 1/(1+np.exp(-s))

    def sigmoidDerive(self, s): # prime sigmoid function
        return s * (1 - s)

    def save(self, filename):
        """ Save to JSON """
        try:
            data = {}
            data['weights'] = []
            for w in self.weights:
                data['weights'].append(w.tolist())
            with open(filename, 'w') as outfile:
                json.dump(data, outfile)
        except Exception as e:
            print(e)
            return False
        return True

    def load(self, filename):
        """ Load a JSON """
        try:
            with open(filename) as f:
                data = json.load(f)
            self.weights = []
            for w in data['weights']:
                self.weights.append(np.array(w))
        except Exception as e:
            print(e)
            return False
        return True

    def forward(self, inputs):
        """ Forward Propagation
        Give an output for given inputs """
        out = [] # store intermediate outputs for training purpose
        for w in self.weights:
            if len(out) == 0:
                out.append(self.sigmoid(np.dot(inputs.astype(float), w))) # first layer uses inputs
            else:
                out.append(self.sigmoid(np.dot(out[-1].astype(float), w))) # successive layers uses previous layers outputs
        return out

    def backward(self, inputs, outputs, learning = 1.0, decay=0.0, n = 1):
        """ Backward Propagation
        Train the network 
        learning: learning rate
        decay: weight decay"""
        for i in range(0, n): # iterations
            results = self.forward(inputs) # get outputs
            deltas = None
            for i in range(len(results)-1, -1, -1): # for each layer (from last to first)
                if i == len(results)-1: # last layer
                    error = outputs - results[i] # calculate error
                    deltas = error*self.sigmoidDerive(results[i]) # delta output
                else: # other layers
                    error = deltas.dot(self.weights[i+1].T) # calculate error
                    deltas = error*self.sigmoidDerive(results[i]) # delta output
                if i == 0: # first layer
                    self.weights[i] += inputs.T.dot(deltas) * learning # adjust the set of weights
                else: # other layers
                    self.weights[i] += results[i].T.dot(deltas) * learning # adjust the set of weights
                # apply weight decay
                self.weights[i] -= self.weights[i] * decay

# test
nn = NN([2, 3, 1])
if nn.load('test.json'):
    print("Loaded from file")

print("Starting Weights: ")
print(nn.weights)

print("Training...")
"""training_inputs = np.array([[0,0,1],
                            [1,1,1],
                            [1,0,1],
                            [0,1,1]])"""

#training_outputs = np.array([[0],[1],[1],[0]])

training_inputs = np.array(([2, 9], [1, 5], [3, 6]), dtype=float) # input data
training_outputs = np.array(([0.92], [0.86], [0.89]), dtype=float) # output
nn.backward(training_inputs, training_outputs, learning=0.1, decay=0.0001, n=500000)

print("Weights: ")
print(nn.weights)

print("Test Output with inputs [5,10]: ")
print(nn.forward(np.array([5,10]))[-1])


print("Saving...")
nn.save('test.json')