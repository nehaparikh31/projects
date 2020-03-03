import numpy
import algo

weights = 8

solution = 9

x = [0.1,0.9, 0.9,0.1, 0.1,0.1, 0.9, 0.9]

num_parent = 2
generations = 10

population_size = (solution,weights) 
new_population = numpy.random.uniform(low=-2.0, high=2.0, size = population_size)

print(new_population)


for gen in range(generations):
    print("Generation : ", gen)

    fitness = algo.population_fitness(x, new_population)

    parent_selection = algo.selection_method(new_population, fitness,num_parent)
    
    crossover = algo.crossover(parent_selection,offspring_size=(population_size[0]-parent_selection.shape[0], weights))

    mutation = algo.mutation(crossover)

    new_population[0:parent_selection.shape[0], :] = parent_selection
    new_population[parent_selection.shape[0]:, :] = mutation

    print("Fittest Result is: ", numpy.max(numpy.sum(new_population*x, axis=1)))



fitness = algo.population_fitness(x, new_population)
best_match = numpy.where(fitness == numpy.max(fitness))

print("Fittest solution is: ", new_population[best_match, :])
print("Best solution for fitness is: ", fitness[best_match])