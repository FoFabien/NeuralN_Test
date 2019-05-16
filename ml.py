import random
import json
import math

def KahanSum(l):
    sum = 0.0
    c = 0.0
    for i in l:
        y = i - c
        t = sum + y
        c = (t - sum) - y
        sum = t
    return sum

class NN():
    def __init__(self, inputs = 0, hiddens = [], outputs = 0):
        self.layers = []
        self.training = []
        if inputs < 0 or outputs < 0:
            raise Exception("Negative parameter")
        for h in hiddens:
            if h < 0:
                raise Exception("Negative parameter")
        while len(self.layers) < len(hiddens) + 2:
            self.layers.append([])
            #self.training.append([])
        while len(self.layers[0]) < inputs:
            self.layers[0].append(random.uniform(-1, 1))
            #self.training[0].append(0.0)
        for i in range(0, len(hiddens)):
            while len(self.layers[i+1]) < hiddens[i]:
                self.layers[i+1].append(random.uniform(-1, 1))
        while len(self.layers[-1]) < outputs:
            self.layers[-1].append(random.uniform(-1, 1))

    def save(self, filename):
        try:
            data = {}
            data['layers'] = self.layers
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
            self.layers = data['layers']
        except Exception as e:
            print(e)
            return False
        return True

    def run(self, inputs, full=False):
        if len(inputs) != len(self.layers[0]):
            raise Exception("Invalid input")
        buffer = [inputs]
        for i in range(0, len(self.layers)):
            buffer.append([])
            for j in range(0, len(self.layers[i])):
                sum = []
                for h in range(0, len(buffer[i])):
                    sum.append(buffer[i][h] * self.layers[i][j])
                try:
                    buffer[i+1].append(1.0 / (1.0 + math.exp(-KahanSum(sum))))
                except:
                    buffer[i+1].append(0.0)
        if full: return buffer[1:]
        else: return buffer[-1]

    def train(self, inputs, outputs):
        if len(inputs) != len(outputs):
            raise Exception("Invalid training data")
        trainingdelta = []
        for i in range(0, len(inputs)):
            buffer = self.run(inputs[i], True)
            deltas = [[]]
            for j in range(0, len(buffer[-1])):
                deltas[0].append((buffer[-1][j] - outputs[i][j]) * buffer[-1][j] * (1 - buffer[-1][j]))
            x = len(buffer) - 2
            while x >= 0:
                sum = []
                deltas.insert(0, [])
                for j in range(0, len(deltas[1])):
                    sum.append(deltas[1][j] * self.layers[x+1][j])
                for j in range(0, len(buffer[x])):
                    deltas[0].append(KahanSum(sum) * buffer[x][j] * (1 - buffer[x][j]))
                x -= 1
            trainingdelta.append(deltas)
            """print(i, "buffer", buffer)
            print(i, "delta", deltas)
            print(i, "layers", self.layers)"""
            #for deltas in trainingdelta:
            for j in range(0, len(deltas)):
                for h in range(0, len(deltas[x])):
                    self.layers[j][h] += - 0.5 * deltas[j][h] * buffer[j][h] - 0.0001 * self.layers[j][h]

nn = NN(2, [8, 4, 2], 2)
#nn = NN()
#nn.load("test.json")
for i in range(0, 100000):
    nn.train([[0, 0], [1, 0], [0, 1], [1, 1]], [[0, 1], [0, 1], [0, 1], [1, 0]])
    if i % 2000 == 0:
        print(i, nn.run([0, 1]))
nn.save("test.json")
print([0, 0], nn.run([0, 0]))
print([1, 0], nn.run([1, 0]))
print([0, 1], nn.run([0, 1]))
print([1, 1], nn.run([1, 1]))