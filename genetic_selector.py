import numpy as np
import pickle

class Individual:

  def __init__(self, chromossome, fitness = 0):

    self.chromossome = chromossome # ex: '00010001000'
    self.fitness = fitness
  
  def calculateIndivFitness(self, x_train, y_train,
                            x_test, y_test, ModelBuilder,
                            **builderArgs):
        
    assert type(x_train) == np.ndarray, 'Dataset must be numpy arrays types'
    assert type(x_test) == np.ndarray, 'Dataset must be numpy arrays types'
    assert type(y_train) == np.ndarray, 'Dataset must be numpy arrays types'
    assert type(y_test) == np.ndarray, 'Dataset must be numpy arrays types'

    if ModelBuilder.framework == 'tf':
      model = ModelBuilder.modelBuilder(**builderArgs)
      EPOCHS = 5
      BATCH_SIZE = 1024
      print(f'Individual has {x_train[..., self.chromossome].shape[-1]} features. Calculating fitness function...')
      model.fit(x_train[..., self.chromossome], y_train, epochs = EPOCHS,
              batch_size = BATCH_SIZE, verbose = 0)
      testMetrics = model.evaluate(x_test[..., self.chromossome], y_test,
                                 batch_size = BATCH_SIZE)
      self.fitness = testMetrics[1]
      print(f'Fitness: {self.fitness:.4f}')  

      return self.fitness

    elif ModelBuilder.framework == 'sklearn':

      print(f'Individual has {x_train[..., self.chromossome].shape[-1]} features')
      print('Calculating fitness function...')
      model = ModelBuilder.modelBuilder(**builderArgs)
      model.fit(x_train[..., self.chromossome], y_train)
      testMetrics = model.score(x_test[..., self.chromossome], y_test)
      self.fitness = testMetrics
      print(f'Fitness: {self.fitness:.4f}')  

      return self.fitness

  def crossover(self, otherIndividual):

    maxFeatures = len(self.chromossome)
    middleValue = int(maxFeatures/2)

    infLim = int(np.random.randint(low = 0, high = middleValue, size = 1))
    supLim = int(np.random.randint(low = middleValue, high = maxFeatures, size = 1))
    
    tmpChromossome = otherIndividual.chromossome.copy()[infLim:supLim]
    otherIndividual.chromossome[infLim:supLim] = self.chromossome[infLim:supLim]
    self.chromossome[infLim:supLim] = tmpChromossome

  def mutation(self, mutationChance):
    if mutationChance == None:
      mutationChance = 100 - int(1/len(self.chromossome) * 100)
    else:
      mutationChance = 100 - mutationChance

    for genePosition in range(len(self.chromossome)):
      probability = int(np.random.randint(low = 0, high = 100, size = 1))
      if probability >= mutationChance:
        
        if self.chromossome[genePosition] == True:
          self.chromossome[genePosition] = False
        else:
          self.chromossome[genePosition] = True
  
class Population:

  def __init__(self, population = None):
    self.size = 0
    if population == None:
      self.population = []
    else:
      self.population = population
    self.populationFitness = []

  def addIndividual(self, Individual):
    self.population.append(Individual)
    if Individual.fitness != 0:
      self.populationFitness.append(Individual.fitness)
    self.size += 1

  def initializePopulationRandom(self, initialSize, features):
    print('Intializing random population...\n')
    for indivNumber in range(initialSize):
      indiv = Individual(np.random.choice([False, True], size = features.shape[-1]),
                         )
      self.addIndividual(indiv)

  def initializePopulationCustom(self, initialSize, features,
                                customChromossome, mutationChance):
    print('Intializing custom population...\n')
    for indivNumber in range(initialSize):
      indiv = Individual(customChromossome)
      self.addIndividual(indiv)
    self.mutationPopulation(mutationChance)    

  def calculateFitness(self, x_train, y_train, x_test, y_test, ModelBuilder, **otherArgs):
    
    if ModelBuilder.framework == 'tf':
      for individual in self.population:
        fitness = individual.calculateIndivFitness(x_train, y_train,
                                                   x_test, y_test,
                                                   ModelBuilder,
                                                   input_shape = (x_train[..., individual.chromossome].shape[-1], ), **otherArgs)
        self.populationFitness.append(fitness)
        
    elif ModelBuilder.framework == 'sklearn':
      for individual in self.population:
        fitness = individual.calculateIndivFitness(x_train, y_train,
                                                   x_test, y_test,
                                                   ModelBuilder,
                                                   **otherArgs)
        self.populationFitness.append(fitness)
    
    else:
      print(f"Model's framework isn't 'tf' or 'sklearn'")
    
  def selectPopulationRandom(self, size):
    probabilities = ((np.array(self.populationFitness)**2 )
                     / (np.array(self.populationFitness)**2).sum())
    selected = np.random.choice(self.population,
                                size = size,
                                replace = False,
                                p = probabilities)
    bestIndividuals = []
    for individual in selected:
      bestIndividuals.append(Individual(individual.chromossome.copy()))

    return bestIndividuals

  def selectPopulationBest(self, size):

    sorted = np.flip(np.argsort(self.populationFitness))

    selected = np.array(self.population)[sorted[:size]]
    bestIndividuals = []
    for individual in selected:
      bestIndividuals.append(Individual(individual.chromossome.copy()))

    return bestIndividuals

  def replicateParents(self, finalSize):

    sizeMultiplier = int(finalSize/len(self.population) - 1)
    tmp = np.array(self.population)

    for i in range(sizeMultiplier):
      np.random.shuffle(tmp)
      for individual in tmp:
        newIndividual = Individual(individual.chromossome.copy())
        self.addIndividual(newIndividual)
    
  def crossoverPopulation(self):

    for individualPosition in range(0, len(self.population), 2):
      self.population[individualPosition].crossover(self.population[individualPosition + 1])

  def mutationPopulation(self, mutationChance = None):

    for individualPosition in range(0, len(self.population)):
      self.population[individualPosition].mutation(mutationChance)

  def printResults(self, rankSize):

    sorted = np.flip(np.argsort(self.populationFitness))

    print('\nResults for current generation:')
    for individual in np.array(self.population)[sorted[:rankSize]]:
      print(f'Individual with {individual.chromossome.sum()} features and chromossome:')
      print(f'{individual.chromossome}')
      print(f'Fitness: {individual.fitness:.4f}')   

class GeneticAlgorithm:

  def __init__(self):
    self.generations = []
    
  def fit(self, populationSize, selectionSize, resultsSize, numGenerations, 
          x_train, y_train, x_test, y_test, modelBuilderFunction, framework,
          **builderArgs):
    
    model = ModelBuilder(modelBuilderFunction, framework)
    print('Generation number 0')
    
    if self.generations == []:
        
        population = Population()
        population.initializePopulationRandom(populationSize, x_train)

        population.calculateFitness(x_train, y_train,
                                    x_test, y_test,
                                    model, **builderArgs)

        population.printResults(resultsSize)

        self.generations.append(population)        
    
    else:
        
        population = Population(self.Population[-1])
        
    
    for i in range(numGenerations):
      print(f'\nGeneration number {i+1}')
      selected = population.selectPopulationBest(selectionSize)
      population = Population(selected)
      population.replicateParents(populationSize)
      population.crossoverPopulation()
      population.mutationPopulation()
      population.calculateFitness(x_train, y_train,
                                  x_test, y_test,
                                  model, **builderArgs)

      population.printResults(resultsSize) 
      self.generations.append(population)

  def addPopulation(self, Population):
    self.generations.append(Population)

  def saveGenerations(self, pathAndFilename):
    with open(pathAndFilename, 'wb') as output:  
        pickle.dump(self, output, pickle.HIGHEST_PROTOCOL)
        
  def loadGenerations(self, pathAndFileName):
    loaded = pickle.load(open( pathAndFileName, "rb" ))
    self.generations = loaded.generations

class ModelBuilder:
  # Diferentiate between sklearn and tensorflow models

  def __init__(self, modelBuilder, framework):
    self.modelBuilder = modelBuilder
    self.framework = framework