from experiments.crazy_env.tsp_data_collection import Env
from experiments.crazy_env import log3 as Log
import numpy as np
import math
import time
import random, operator
import pandas as pd
import matplotlib.pyplot as plt


def myint(a):
    # return int(np.ceil(a))
    return int(np.floor(a))


class City:
    def __init__(self, x, y, env):
        self.x = x
        self.y = y
        self.env = env

    def distance(self, tocity):
        dx = tocity.x - self.x
        dy = tocity.y - self.y

        if 0 <= self.x + dx < self.env.mapx and 0 <= self.x + dx < self.env.mapy and self.env.mapob[myint(self.x + dx)][
            myint(self.y + dy)] != self.env.OB and \
                self.env.mapob[myint(self.x + (dx / 2))][myint(self.y + (dy / 2))] != self.env.OB and \
                self.env.mapob[myint(self.x + (dx / 3))][myint(self.y + (dy / 3))] != self.env.OB and \
                self.env.mapob[myint(self.x + (2 * dx / 3))][myint(self.y + (2 * dy / 3))] != self.env.OB and \
                self.env.mapob[myint(self.x + (dx / 4))][myint(self.y + (dy / 4))] != self.env.OB and \
                self.env.mapob[myint(self.x + (3 * dx / 4))][myint(self.y + (3 * dy / 4))] != self.env.OB:

            distance = np.sqrt((dx ** 2) + (dy ** 2))
        else:
            distance = 50

        return distance

    def __repr__(self):
        return "(" + str(self.x) + "," + str(self.y) + ")"


class Fitness:
    def __init__(self, route):
        self.route = route
        self.distance = 0
        self.fitness = 0.0

    def routeDistance(self):
        if self.distance == 0:
            pathDistance = 0
            for i in range(0, len(self.route)):
                fromCity = self.route[i]
                toCity = None
                if i + 1 < len(self.route):
                    toCity = self.route[i + 1]
                else:
                    toCity = self.route[0]
                pathDistance += fromCity.distance(toCity)
            self.distance = pathDistance
        return self.distance

    def routeFitness(self):
        if self.fitness == 0:
            self.fitness = 1 / float(self.routeDistance())
        return self.fitness


def createRoute(cityList):
    route = random.sample(cityList, len(cityList))
    return route


def initialPopulation(popSize, cityList):
    population = []

    for i in range(0, popSize):
        population.append(createRoute(cityList))
    return population


def rankRoutes(population):
    fitnessResults = {}
    for i in range(0, len(population)):
        fitnessResults[i] = Fitness(population[i]).routeFitness()
    return sorted(fitnessResults.items(), key=operator.itemgetter(1), reverse=True)


def selection(popRanked, eliteSize):
    selectionResults = []
    df = pd.DataFrame(np.array(popRanked), columns=["Index", "Fitness"])
    df['cum_sum'] = df.Fitness.cumsum()
    df['cum_perc'] = 100 * df.cum_sum / df.Fitness.sum()

    for i in range(0, eliteSize):
        selectionResults.append(popRanked[i][0])
    for i in range(0, len(popRanked) - eliteSize):
        pick = 100 * random.random()
        for i in range(0, len(popRanked)):
            if pick <= df.iat[i, 3]:
                selectionResults.append(popRanked[i][0])
                break
    return selectionResults


def matingPool(population, selectionResults):
    matingpool = []
    for i in range(0, len(selectionResults)):
        index = selectionResults[i]
        matingpool.append(population[index])
    return matingpool


def breed(parent1, parent2):
    child = []
    childP1 = []
    childP2 = []

    geneA = int(random.random() * len(parent1))
    geneB = int(random.random() * len(parent1))

    startGene = min(geneA, geneB)
    endGene = max(geneA, geneB)

    for i in range(startGene, endGene):
        childP1.append(parent1[i])

    childP2 = [item for item in parent2 if item not in childP1]

    child = childP1 + childP2
    return child


def breedPopulation(matingpool, eliteSize):
    children = []
    length = len(matingpool) - eliteSize
    pool = random.sample(matingpool, len(matingpool))

    for i in range(0, eliteSize):  # to carry the best individuals into the next generation
        children.append(matingpool[i])

    for i in range(0, length):
        child = breed(pool[i], pool[len(matingpool) - i - 1])
        children.append(child)
    return children


def mutate(individual, mutationRate):
    for swapped in range(len(individual)):
        if (random.random() < mutationRate):
            swapWith = int(random.random() * len(individual))

            city1 = individual[swapped]
            city2 = individual[swapWith]

            individual[swapped] = city2
            individual[swapWith] = city1
    return individual


def mutatePopulation(population, mutationRate):
    mutatedPop = []

    for ind in range(0, len(population)):
        mutatedInd = mutate(population[ind], mutationRate)
        mutatedPop.append(mutatedInd)
    return mutatedPop


def nextGeneration(currentGen, eliteSize, mutationRate):
    popRanked = rankRoutes(currentGen)
    selectionResults = selection(popRanked, eliteSize)
    matingpool = matingPool(currentGen, selectionResults)
    children = breedPopulation(matingpool, eliteSize)
    nextGeneration = mutatePopulation(children, mutationRate)
    return nextGeneration


def geneticAlgorithm(population, popSize, eliteSize, mutationRate, generations):
    pop = initialPopulation(popSize, population)
    print("Initial distance: " + str(1 / rankRoutes(pop)[0][1]))

    for i in range(0, generations):
        pop = nextGeneration(pop, eliteSize, mutationRate)

    print("Final distance: " + str(1 / rankRoutes(pop)[0][1]))
    bestRouteIndex = rankRoutes(pop)[0][0]
    bestRoute = pop[bestRouteIndex]
    return bestRoute


def geneticAlgorithmPlot(population, popSize, eliteSize, mutationRate, generations, env_log, reg_n):
    log_path = env_log.full_path
    pop = initialPopulation(popSize, population)
    print("Initial distance: " + str(1 / rankRoutes(pop)[0][1]))

    progress = []
    progress.append(1 / rankRoutes(pop)[0][1])

    for i in range(0, generations):
        pop = nextGeneration(pop, eliteSize, mutationRate)
        progress.append(1 / rankRoutes(pop)[0][1])
        end = False
        if i % 10 == 0:
            plt.plot(progress)
            plt.ylabel('Distance')
            plt.xlabel('Generation')
            plt.savefig(log_path + '/Distance_generation_%d.png' % (reg_n))
            plt.close()
            if i > 50:
                test_coverage = progress[i - 50:i]
                list_var = np.var(test_coverage)
                print("%d th var: %f" % (i, list_var))
            else:
                list_var = 1e5
                print(i)

            if list_var < 1e-5:
                end = True
                break

        if end is True:
            break

    print("Final distance: " + str(1 / rankRoutes(pop)[0][1]))
    bestRouteIndex = rankRoutes(pop)[0][0]
    bestRoute = pop[bestRouteIndex]
    return bestRoute


def train(num_uav):
    log = Log.Log()
    env = Env(log)
    print("training %d PoIs..." % (len(env.datas)))
    start = time.clock()

    for n in range(num_uav):
        cityList = []

        for i in range(0, len(env.datas)):
            # 随机测试
            # cityList.append(City(x=random.random() * 16, y=random.random() * 16))
            datax = env.datas[i][0]
            datay = env.datas[i][1]
            ab_reg = float(env.mapx) / num_uav
            if ab_reg * n <= datax <= ab_reg * (n + 1):
                cityList.append(City(x=datax, y=datay, env=env))

        print("\nthe %dth region: %d PoI" % (n, len(cityList)))
        # geneticAlgorithm(population=cityList, popSize=100, eliteSize=20, mutationRate=0.01, generations=500)

        bestRoute = geneticAlgorithmPlot(population=cityList, popSize=300, eliteSize=50, mutationRate=0.01,
                                         generations=3000,
                                         env_log=log,
                                         reg_n=n)

        bestRoutelist = []
        for poi in bestRoute:
            bestRoutelist.append([poi.x, poi.y])

        bestRouteDataFrame = pd.DataFrame(np.array(bestRoutelist), columns=["x", "y"])
        bestRouteDataFrame.to_csv(log.full_path + '/saved_route_uav%d.csv' % n)

    training_time = time.clock() - start
    print("\n\nTraining time: ", training_time)


def __cusume_energy(env, uav, value, distance):
    # distance-0.1, alpha-1.0
    if (env.factor * distance + env.alpha * value < env.energy[uav]):
        env.energy[uav] -= (env.factor * distance + env.alpha * value)
        env.use_energy[uav] += (env.factor * distance + env.alpha * value)
    else:
        env.use_energy[uav] += env.energy[uav]
        distance = env.energy[uav] / env.factor
        env.energy[uav] = 0

    return env


def test(num_uav, model_path):
    print("testing...")
    log = Log.Log()
    env = Env(log)
    _ = env.reset()

    for n in range(num_uav):
        df = pd.read_csv("%s/saved_route_uav%d.csv" % (model_path, n))
        print("the %dth region: %d PoI" % (n, df.shape[0]))
        step = 0
        i = 0

        while step < 500:
            new_positions = [df.loc[i, 'x'], df.loc[i, 'y']]

            # charge
            _pos = np.repeat([new_positions], [env.fills.shape[0]], axis=0)  # just repeat(On)  NB!
            _minus = env.fills - _pos
            _power = np.power(_minus, 2)
            _dis = np.sum(_power, axis=1)
            for index, dis in enumerate(_dis):
                # sensing Fill Station(crange=1.1)
                if np.sqrt(dis) <= env.crange:
                    # uodate poi data
                    if env.fills_energy_remain[index] > 0:
                        # TODO:加油站的信息更新
                        if env.fspeed * env.maxenergy <= env.fills_energy_remain[index]:
                            if env.energy[n] + env.fspeed * env.maxenergy <= env.maxenergy:
                                env.fill_energy[n] += env.fspeed * env.maxenergy
                                env.fills_energy_remain[index] -= env.fspeed * env.maxenergy
                                env.energy[n] += env.fspeed * env.maxenergy
                            else:
                                env.fill_energy[n] += env.maxenergy - env.energy[n]
                                env.fills_energy_remain[index] -= (env.maxenergy - env.energy[n])
                                env.energy[n] = env.maxenergy
                        else:
                            if env.energy[n] + env.fills_energy_remain[index] <= env.maxenergy:
                                env.fill_energy[n] += env.fills_energy_remain[index]
                                env.energy[n] += env.fills_energy_remain[index]
                                env.fills_energy_remain[index] = 0
                            else:
                                env.fill_energy[n] += env.maxenergy - env.energy[n]
                                env.fills_energy_remain[index] -= (env.maxenergy - env.energy[n])
                                env.energy[n] = env.maxenergy
                    break

            # collect
            data = 0
            _pos = np.repeat([new_positions], [env.datas.shape[0]], axis=0)
            _minus = env.datas - _pos
            _power = np.power(_minus, 2)
            _dis = np.sum(_power, axis=1)
            for index, dis in enumerate(_dis):
                # sensing PoI(crange=1.1)
                if np.sqrt(dis) <= env.crange:
                    # uodate poi data
                    if env.mapmatrix[index] > 0:
                        tmp_data = env._mapmatrix[index] * env.cspeed
                        if env.energy[n] >= tmp_data * env.alpha:
                            data += tmp_data
                            env.mapmatrix[index] -= tmp_data
                            if env.mapmatrix[index] < 0:
                                env.mapmatrix[index] = 0.
                        else:
                            data += env.energy[n]
                            env.mapmatrix[index] -= env.energy[n]
                            if env.mapmatrix[index] < 0:
                                env.mapmatrix[index] = 0.
                            break

            value = data if env.energy[n] >= data * env.alpha else env.energy[n]
            env.collection[n] += value
            env = __cusume_energy(env, n, value, 0.)  # collect

            if i == df.shape[0] - 1:
                # env.energy[n]=env.maxenergy  # 不加！
                ii = 0
            else:
                ii = i + 1

            distance = np.sqrt(((df.loc[ii, 'x'] - df.loc[i, 'x']) ** 2) + ((df.loc[ii, 'y'] - df.loc[i, 'y']) ** 2))

            if distance <= env.maxdistance:
                env = __cusume_energy(env, n, 0, distance)  # move

                # 撞墙
                dx = df.loc[ii, 'x'] - df.loc[i, 'x']
                dy = df.loc[ii, 'y'] - df.loc[i, 'y']
                if 0 <= df.loc[ii, 'x'] < env.mapx and 0 <= df.loc[ii, 'y'] < env.mapy and \
                        env.mapob[myint(df.loc[ii, 'x'])][
                            myint(df.loc[ii, 'y'])] != env.OB and \
                        env.mapob[myint(df.loc[i, 'x'] + (dx / 2))][myint(df.loc[i, 'y'] + (dy / 2))] != env.OB and \
                        env.mapob[myint(df.loc[i, 'x'] + (dx / 3))][myint(df.loc[i, 'y'] + (dy / 3))] != env.OB and \
                        env.mapob[myint(df.loc[i, 'x'] + (2 * dx / 3))][
                            myint(df.loc[i, 'y'] + (2 * dy / 3))] != env.OB and \
                        env.mapob[myint(df.loc[i, 'x'] + (dx / 4))][myint(df.loc[i, 'y'] + (dy / 4))] != env.OB and \
                        env.mapob[myint(df.loc[i, 'x'] + (3 * dx / 4))][myint(df.loc[i, 'y'] + (3 * dy / 4))] != env.OB:
                    i = ii
            else:
                env = __cusume_energy(env, n, 0, env.maxdistance)  # move
                newx = df.loc[i, 'x'] + (df.loc[ii, 'x'] - df.loc[i, 'x']) * (env.maxdistance / distance)
                newy = df.loc[i, 'y'] + (df.loc[ii, 'y'] - df.loc[i, 'y']) * (env.maxdistance / distance)

                dx = newx - df.loc[i, 'x']
                dy = newy - df.loc[i, 'y']
                if 0 <= newx < env.mapx and 0 <= newy < env.mapy and \
                        env.mapob[myint(newx)][myint(newy)] != env.OB and \
                        env.mapob[myint(df.loc[i, 'x'] + (dx / 2))][myint(df.loc[i, 'y'] + (dy / 2))] != env.OB and \
                        env.mapob[myint(df.loc[i, 'x'] + (dx / 3))][myint(df.loc[i, 'y'] + (dy / 3))] != env.OB and \
                        env.mapob[myint(df.loc[i, 'x'] + (2 * dx / 3))][
                            myint(df.loc[i, 'y'] + (2 * dy / 3))] != env.OB and \
                        env.mapob[myint(df.loc[i, 'x'] + (dx / 4))][myint(df.loc[i, 'y'] + (dy / 4))] != env.OB and \
                        env.mapob[myint(df.loc[i, 'x'] + (3 * dx / 4))][myint(df.loc[i, 'y'] + (3 * dy / 4))] != env.OB:
                    df.loc[i, 'x'] = newx
                    df.loc[i, 'y'] = newy
            step += 1

    print('efficiency: %.3f' % env.efficiency)
    print('data_collection_ratio: %.3f' % (1.0 - env.leftrewards))
    print('fairness: %.3f' % env.collection_fairness)
    print('normal fairness: %.3f' % env.normal_collection_fairness)
    print('energy_consumption: %.3f' % (np.sum(env.normal_use_energy)))
    print('fill:', env.fills_energy_remain)


if __name__ == '__main__':
    num_uav = 5
    # train(num_uav=num_uav)
    test(num_uav=num_uav, model_path='/home/dzp1997/PycharmProjects/maddpg-czy-DZP/experiments/2019/06-29/uav5')
