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
    # Network structure (can be overridden in __init__)
    inputNodeNum = 3      # dx, dy, velocity
    outputNodeNum = 1     # Single output: > 0.5 = flap

    def __init__(self, hidden_structure=None):
        if hidden_structure is None:
            hidden_structure = [4]

        self.tensors = []
        self.vectors = []

        # Fitness
        self.distance = 0
        self.score = 0

        # Position
        self.x = 0
        self.y = 0

        # Game parameters
        self.birdVelY = 0
        self.birdFlapped = False
        self.xMidPos = 0
        self.output = None

        # Build network: input → hidden layers → output
        networkStructure = [self.inputNodeNum] + hidden_structure + [self.outputNodeNum]

        for i in range(len(networkStructure) - 1):
            columns = int(networkStructure[i])
            rows = int(networkStructure[i + 1])
            self.tensors.append(np.random.uniform(w_min, w_max, (rows, columns)))

        for nodes in networkStructure:
            self.vectors.append(np.zeros(shape=(nodes, 1)))



    def set_input(self, dx, dy, velocity):
        """
            Sets values of input nodes.
        """
        # Normalize all to symmetric -1 to +1 range
        self.vectors[0][0] = (dx - 144) / 144.0   # center around 0: -1 to +1
        self.vectors[0][1] = dy / 200.0           # -1 to +1
        self.vectors[0][2] = velocity / 8.0       # -1 to +1 (exact range)



    def process(self):
        """
            Given input values (nodes (0,1)), this updates the rest of the node values. Returns network output.
        """

        for i in range(len(self.vectors)-1):
            self.vectors[i + 1] = sigmoid(self.tensors[i].dot(self.vectors[i]))

        # Single output: sigmoid gives 0-1, flap if > 0.5
        self.output = float(self.vectors[-1][0, 0])


    def flush_nodes(self):
        """
            Returns values of all nodes to int(0).
        """
        for vector in self.vectors:
            vector.fill(0)


    def flush_distance(self):

        self.distance = 0
        self.score = 0


    def mutate(self, mutation_strength=0.1):
        """
            Mutates ~10% of weights randomly using vectorized numpy operations.
            mutation_strength: std dev of normal distribution for mutations
        """
        for tensor in self.tensors:
            mask = np.random.random(tensor.shape) < 0.1
            num_mutations = mask.sum()
            if num_mutations > 0:
                tensor[mask] += np.random.normal(0, mutation_strength, num_mutations)


    def fly_up(self):
        self.process()
        self.flush_nodes()
        return self.output > 0.5
