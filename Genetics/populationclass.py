from Genetics.birdnetclass import BirdNet
import numpy as np
import random


def clone_weights(parent, child):
    """Copy parent's weights to child."""
    for i in range(len(parent.tensors)):
        child.tensors[i] = parent.tensors[i].copy()
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
        parents = self.individuals[:parent_num]
        unfit = self.individuals[parent_num:]
        return parents, unfit

    def breed(self, parents, unfit):
        """Clone parents into unfit slots, then mutate.
        Higher-ranked parents are more likely to be selected."""
        num_parents = len(parents)
        # Weights: parent 0 gets num_parents, parent 1 gets num_parents-1, etc.
        weights = [num_parents - i for i in range(num_parents)]

        for child in unfit:
            # Weighted random selection favoring top parents
            parent_idx = random.choices(range(num_parents), weights=weights)[0]
            parent = parents[parent_idx]
            clone_weights(parent, child)
            # Scale mutation: best parent = 0.5x, worst parent = 1.5x
            scale = 0.5 + (parent_idx / max(1, num_parents - 1))
            child.mutate(self.mutation_rate * scale)
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

        [x.flush_distance() for x in self.individuals]

    def sort(self):
        self.individuals = sorted(self.individuals, key=lambda x: x.distance, reverse=True)
