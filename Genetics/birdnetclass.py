import networkx as nx
import numpy as np

epsilon = 0.1
w_min = -1
w_max = 1

# Pre-computed reciprocals for faster input normalization (multiplication > division)
INV_144 = 1.0 / 144.0
INV_200 = 1.0 / 200.0
INV_8 = 1.0 / 8.0

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

        # Cache for hot path optimization
        self.num_layers = len(self.vectors)



    def set_input(self, dx, dy, velocity):
        """
            Sets values of input nodes.
        """
        # Cache vectors[0] to avoid repeated LOAD_ATTR + BINARY_SUBSCR
        v0 = self.vectors[0]
        # Normalize all to symmetric -1 to +1 range (using pre-computed reciprocals)
        v0[0] = (dx - 144) * INV_144   # center around 0: -1 to +1
        v0[1] = dy * INV_200           # -1 to +1
        v0[2] = velocity * INV_8       # -1 to +1 (exact range)



    def process(self):
        """
            Given input values (nodes (0,1)), this updates the rest of the node values. Returns network output.
        """
        # Cache self.vectors and self.tensors to avoid repeated LOAD_ATTR in loop
        vectors = self.vectors
        tensors = self.tensors
        for i in range(self.num_layers - 1):
            vectors[i + 1] = sigmoid(tensors[i].dot(vectors[i]))

        # Single output: sigmoid gives 0-1, flap if > 0.5
        # .item() is faster than [0,0] indexing (avoids tuple construction)
        self.output = vectors[-1].item()


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
        # Note: flush_nodes() removed - vectors are overwritten by set_input() next frame
        return self.output > 0.5
