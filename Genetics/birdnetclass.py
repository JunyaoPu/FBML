import networkx as nx
import numpy as np

epsilon = 0.1
w_min = -1
w_max = 1

def mutGen():
    return np.random.normal(0, epsilon)

def weightGen():
    return np.random.uniform(-0.001,0.001)

def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

class BirdNet:
    #TODO: Normalize propagation.
    #TODO: Use sigmoid function for activations.

    inputNodeNum    = 2
    hiddenStructure = [100,20,5]
    outputNodeNum   = 2

    tensors = None
    vectors = None




    # Fitness:
    distance = 0

    # Position:
    x = 0
    y = 0

    # Game parameters:
    score = 0
    birdVelY = 0
    birdFlapped = False
    xMidPos = 0

    output = None

    def __init__(self):
        self.tensors = []
        self.vectors = []

        networkStructure = [self.inputNodeNum] + self.hiddenStructure + [self.outputNodeNum]

        for i in range(len(networkStructure) - 1):
            columns = int(networkStructure[i])
            rows    = int(networkStructure[i + 1])
            self.tensors.append(np.random.uniform(w_min,w_max,(rows, columns)))

        for nodes in networkStructure:
            self.vectors.append(np.zeros(shape=(nodes,1)))



    def set_input(self, input1, input2):
        """
            Sets values of input nodes.

            Note: Depends on input nodes having the LOWEST indices in graph #FIXME
        """

        norm = np.sqrt(input1 ** 2 + input2 ** 2)
        if norm > 0:
            input1 = input1 / norm
            input2 = input2 / norm
        else:
            input1 = 0
            input2 = 0

        self.vectors[0][0] = input1
        self.vectors[0][1] = input2



    def process(self):
        """
            Given input values (nodes (0,1)), this updates the rest of the node values. Returns network output.
        """

        for i in range(len(self.vectors)-1):
            self.vectors[i + 1] = sigmoid(self.tensors[i].dot(self.vectors[i]))

        if (self.vectors[-1][0] > self.vectors[-1][1]):
            self.output = 1
        else:
            self.output = -1


    def flush_nodes(self):
        """
            Returns values of all nodes to int(0).
        """
        for vector in self.vectors:
            vector.fill(0)


    def flush_distance(self):

        self.distance = 0
        self.score = 0


    def mutate(self):
        """
            Mutates weights.
        """
        for tensor in self.tensors:
            for i in range(tensor.shape[0]):
                for j in range(tensor.shape[1]):
                    tensor[i,j] += mutGen()


    def fly_up(self):
        self.process()
        self.flush_nodes()

        if (self.output > 0.0):
            return True
        else:
            return False
