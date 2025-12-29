from Genetics.birdnetclass import BirdNet
import numpy as np
#from birdnetclass import BirdNet
import random

parent_fraction = 0.2
mutation_rate   = 0.2
bird_num        = 50

def crossover(father, mother, child):

    for i in range(len(father.tensors)):
        for j in range((father.tensors[i].shape[0])):
            for k in range((father.tensors[i].shape[1])):
                if (random.uniform(0,1) < 0.5):
                    child.tensors[i][j,k] = father.tensors[i][j,k]
                else:
                    child.tensors[i][j,k] = mother.tensors[i][j,k]

    return child


def population_divide(individuals):
    """
        Separates population into parents and unfit.
    """

    parent_num = int(parent_fraction * len(individuals))

    parents = individuals[:parent_num]
    unfit   = individuals[parent_num:]

    return parents, unfit


def breed(parents, unfit):
    """
        Breeding parents, replacing unfit with children to maintain constant population.
        Chooses parents randomly. Can be parent multiple times. Can be parent with self. #FIXME
    """

    for child in unfit:
        np.random.random()

        mother_ID   = random.randint(0, len(parents)-1)
        father_ID   = random.randint(0, len(parents)-1)

        father = parents[father_ID]
        mother = parents[mother_ID]

        child = crossover(father, mother, child)

        if (random.uniform(0, 1) < mutation_rate):
            child.mutate()

    children = unfit

    return children


class Population:

    individuals = []


    def __init__(self):
        self.individuals = [BirdNet() for _ in range(bird_num)]

    def evolve(self):
        """Advances to next generation, inserting children into population."""

        self.sort()

        parents, unfit  = population_divide(self.individuals)
        children        = breed(parents, unfit)



        self.individuals = parents + children

        fitness_list = [x.distance for x in self.individuals]
        std_fitness     = np.std(fitness_list)
        average_fitness = np.average(fitness_list)
        print("Mean: ", average_fitness)
        print("Sigma:", std_fitness)

        [x.flush_distance() for x in self.individuals]


    def sort(self):
        self.individuals = sorted(self.individuals, key=lambda x:x.distance, reverse=True) # Descending order
