from Genetics.birdnetclass import BirdNet, sigmoid, INV_144, INV_200, INV_8
import numpy as np
import random


def clone_weights(parent, child):
    """Copy parent's weights to child (in-place to avoid allocation)."""
    for i in range(len(parent.tensors)):
        np.copyto(child.tensors[i], parent.tensors[i])
    return child


class Population:

    def __init__(self, bird_num=50, parent_fraction=0.3, mutation_rate=0.1, hidden_structure=None):
        if hidden_structure is None:
            hidden_structure = [4]

        self.bird_num = bird_num
        self.parent_fraction = parent_fraction
        self.mutation_rate = mutation_rate
        self.hidden_structure = hidden_structure

        self.individuals = [BirdNet(hidden_structure=hidden_structure) for _ in range(bird_num)]
        self.best_ever_weights = None
        self.best_ever_distance = 0

    def population_divide(self):
        """Separates population into parents and unfit."""
        parent_num = int(self.parent_fraction * len(self.individuals))
        # Ensure at least 1 parent (needed for breeding)
        parent_num = max(1, parent_num)
        parents = self.individuals[:parent_num]
        unfit = self.individuals[parent_num:]
        return parents, unfit

    def breed(self, parents, unfit):
        """Clone parents into unfit slots, then mutate.
        Higher-ranked parents are more likely to be selected."""
        num_parents = len(parents)
        if num_parents == 0 or len(unfit) == 0:
            return unfit  # Nothing to breed
        # Weights: parent 0 gets num_parents, parent 1 gets num_parents-1, etc.
        weights = [num_parents - i for i in range(num_parents)]

        for child in unfit:
            # Weighted random selection favoring top parents
            parent_idx = random.choices(range(num_parents), weights=weights)[0]
            parent = parents[parent_idx]
            clone_weights(parent, child)
            # Scale mutation: best parent = 0.5x, worst parent = 1.5x
            scale = 0.5 + (parent_idx / max(1, num_parents - 1))
            child.mutate(mutation_strength=self.mutation_rate * scale, mutation_rate=0.1)
        return unfit

    def evolve(self):
        """Advances to next generation, inserting children into population."""
        self.sort()

        # Track best-ever bird
        current_best = self.individuals[0]
        if current_best.distance > self.best_ever_distance:
            self.best_ever_distance = current_best.distance
            self.best_ever_weights = [t.copy() for t in current_best.tensors]
            print(f"New best ever! Distance: {self.best_ever_distance}")

        parents, unfit = self.population_divide()
        children = self.breed(parents, unfit)

        self.individuals = parents + children

        # Elitism: inject best-ever into slot 0 (unmutated)
        if self.best_ever_weights is not None:
            for i, tensor in enumerate(self.best_ever_weights):
                self.individuals[0].tensors[i] = tensor.copy()

        fitness_list = [x.distance for x in self.individuals]
        std_fitness = np.std(fitness_list)
        average_fitness = np.average(fitness_list)
        print("Mean: ", average_fitness)
        print("Sigma:", std_fitness)

        for x in self.individuals:
            x.flush_distance()

    def sort(self):
        distances = np.array([ind.distance for ind in self.individuals])
        sorted_indices = np.argsort(-distances)  # Descending order
        self.individuals = [self.individuals[i] for i in sorted_indices]

    def stack_tensors(self):
        """Stack all birds' weight tensors into 3D arrays for batch inference.
        Call this once per generation after evolve() to enable batch_forward()."""
        if not self.individuals:
            return
        num_layers = len(self.individuals[0].tensors)
        self.stacked_tensors = []
        for layer_idx in range(num_layers):
            # Stack: (num_birds, rows, cols)
            stacked = np.stack([bird.tensors[layer_idx] for bird in self.individuals])
            self.stacked_tensors.append(stacked)

    def batch_forward(self, dx1, dy1, dx2, dy2, vel, alive_mask):
        """Batched forward pass for all birds using einsum.

        Args:
            dx1: numpy array (num_birds,) - distance to first pipe
            dy1: numpy array (num_birds,) - distance to first pipe's gap
            dx2: numpy array (num_birds,) - distance to second pipe
            dy2: numpy array (num_birds,) - distance to second pipe's gap
            vel: numpy array (num_birds,) - bird velocity
            alive_mask: numpy array (num_birds,) bool - which birds are alive

        Returns:
            should_flap: numpy array (num_birds,) bool - which birds should flap
        """
        num_birds = len(self.individuals)

        # Get indices of alive birds
        alive_indices = np.where(alive_mask)[0]
        if len(alive_indices) == 0:
            return np.zeros(num_birds, dtype=bool)

        # Normalize inputs for alive birds only (both pipes visible)
        x = np.column_stack([
            (dx1[alive_indices] - 144) * INV_144,
            dy1[alive_indices] * INV_200,
            (dx2[alive_indices] - 144) * INV_144,
            dy2[alive_indices] * INV_200,
            vel[alive_indices] * INV_8
        ])  # Shape: (n_alive, 5)

        # Forward pass through each layer using einsum
        for layer_idx, stacked_W in enumerate(self.stacked_tensors):
            # Get weights for alive birds only: (n_alive, out_dim, in_dim)
            W = stacked_W[alive_indices]
            # Batched matrix-vector multiply: einsum('nij,nj->ni', W, x)
            x = sigmoid(np.einsum('nij,nj->ni', W, x))

        # x is now (n_alive, 1) - squeeze to (n_alive,)
        outputs = x.squeeze(-1) if x.ndim > 1 else x

        # Build full result array
        should_flap = np.zeros(num_birds, dtype=bool)
        should_flap[alive_indices] = outputs > 0.5

        return should_flap
