import random
import numpy as np
import functools

import utils

K = 10 #number of piles
POP_SIZE = 500 # population size
MAX_GEN = 1000 # maximum number of generations
CX_PROB = 0.8 # crossover probability
MUT_PROB = 0.2 # mutation probability
MAX_MIN_MUT_PROB = 0.8
MUT_FLIP_PROB = 0.1 # probability of chaninging value during mutation
REPEATS = 10 # number of runs of algorithm (should be at least 10)
OUT_DIR = 'partition' # output directory for logs
EXP_ID = 'informed_fit_cross' # the ID of this experiment (used to create log names)

# reads the input set of values of objects
def read_weights(filename):
    with open(filename) as f:
        return list(map(int, f.readlines()))

# computes the bin weights
# - bins are the indices of bins into which the object belongs
def bin_weights(weights, bins):
    bw = [0]*K
    for w, b in zip(weights, bins):
        bw[b] += w
    return bw

def nPointCrossover(ind1, ind2, n: int = 2):
    lastInd = 0
    res1, res2 = ind1[:], ind2[:]
    for index in range(n):
        split = random.randrange(lastInd, (len(ind1) - n + (index+1)))
        if index % 2 == 0:
            res1[lastInd:split] = ind1[lastInd:split]
            res1[split:] = ind2[split:]
            res2[lastInd:split] = ind2[lastInd:split]
            res2[split:] = ind1[split:]
        else:
            res2[lastInd:split] = ind1[lastInd:split]
            res2[split:] = ind2[split:]
            res1[lastInd:split] = ind2[lastInd:split]
            res1[split:] = ind1[split:]
        lastInd = split + 1
    return res1, res2

def swappingMutation(p1, mut_prob: float = 0.5):
    if random.random() < mut_prob:
        ind1 = random.randrange(0, len(p1)-3)
        ind2 = random.randrange(ind1, len(p1)-2)
        ind3 = random.randrange(ind2, len(p1)-1)
        ind4 = random.randrange(ind3, len(p1))
        seg1 = p1[0:ind1]
        seg2 = p1[ind1:ind2]
        seg3 = p1[ind2:ind3]
        seg4 = p1[ind3:ind4]
        seg5 = p1[ind4:]
        return seg1 + seg4 + seg3 + seg2 + seg5
    else:
        return p1[:]

# the fitness function
def fitness(ind, weights):
    bw = bin_weights(weights, ind)
    return utils.FitObjPair(fitness=1/(max(bw) - min(bw) + 1), 
                            objective=max(bw) - min(bw))

def fitness_with_avg(ind, weights):
    bw = bin_weights(weights, ind)
    return utils.FitObjPair(fitness=1/(compare_to_avg(bw) + 1), 
                            objective=max(bw) - min(bw))

def fitness_with_total_diff(ind, weights):
    bw = bin_weights(weights, ind)
    return utils.FitObjPair(fitness=1/(sum_total_diffs(bw) + 1), 
                            objective=max(bw) - min(bw))

def compare_to_avg(bin_weight):
    avg = np.average(bin_weight)
    return sum(abs(bin_weight-avg))

def sum_total_diffs(bin_weight):
    res = 0
    for i in bin_weight:
        for j in bin_weight:
            res += abs(i-j)
    return res

# creates the individual
def create_ind(ind_len):
    return [random.randrange(0, K) for _ in range(ind_len)]

# creates the population using the create individual function
def create_pop(pop_size, create_individual):
    return [create_individual() for _ in range(pop_size)]

# the roulette wheel selection
def roulette_wheel_selection(pop, fits, k):
    return random.choices(pop, fits, k=k)

# tournament selection
def tour_selection(pop, fits, k):
    selected = []
    for _ in range(k):
        i1, i2 = random.randrange(0, len(pop)), random.randrange(0, len(pop))
        if fits[i1] > fits[i2]:
            selected.append(pop[i1])
        else: 
            selected.append(pop[i2])
    return selected

# implements the one-point crossover of two individuals
def one_pt_cross(p1, p2):
    point = random.randrange(1, len(p1))
    o1 = p1[:point] + p2[point:]
    o2 = p2[:point] + p1[point:]
    return o1, o2

# implements the "bit-flip" mutation of one individual
def flip_mutate(p, prob, upper):
    return [random.randrange(0, upper) if random.random() < prob else i for i in p]

def flip_mutate_informed(p, prob, upper, weights):
    new =  [random.randrange(0, upper) if random.random() < prob else i for i in p]
    if fitness_with_total_diff(p, weights)[0] > fitness_with_total_diff(new, weights)[0]:
        return p
    else:
        return new

# implements the "bit-flip" mutation of one individual
def flip_to_lowest_mutate(p, prob, weights):
    bw = bin_weights(weights, p)
    sought, target = 0, 0
    if random.random() < prob:
        sought = np.array(bw).argmax()
    else:
        sought = random.randrange(0, len(bw))
    if random.random() < prob:
        target = np.array(bw).argmin()
    else:
        target = random.randrange(0, len(bw))
    
    soughtMin = np.Infinity
    soughtIndex = -1
    for i in range(len(p)):
        if p[i] == sought and weights[i] < soughtMin:
            soughtMin = weights[i]
            soughtIndex = i

    p[soughtIndex] = target
    return p

# applies a list of genetic operators (functions with 1 argument - population) 
# to the population
def mate(pop, operators):
    for o in operators:
        pop = o(pop)
    return pop

# applies the cross function (implementing the crossover of two individuals)
# to the whole population (with probability cx_prob)
def crossover(pop, cross, cx_prob):
    off = []
    for p1, p2 in zip(pop[0::2], pop[1::2]):
        if random.random() < cx_prob:
            o1, o2 = cross(p1, p2)
        else:
            o1, o2 = p1[:], p2[:]
        off.append(o1)
        off.append(o2)
    return off

def informed_crossover(pop, cross, cx_prob, weights):
    off = []
    for p1, p2 in zip(pop[0::2], pop[1::2]):
        if random.random() < cx_prob:
            o1, o2 = cross(p1, p2)
        else:
            o1, o2 = p1[:], p2[:]

        if fitness_with_total_diff(p1, weights)[0] > fitness_with_total_diff(o1, weights):
            off.append(p1)
        else:
            off.append(o1)

        if fitness_with_total_diff(p2, weights)[0] > fitness_with_total_diff(o2, weights):
            off.append(p2)
        else:
            off.append(o2)

    return off

# applies the mutate function (implementing the mutation of a single individual)
# to the whole population with probability mut_prob)
def mutation(pop, mutate, mut_prob):
    return [mutate(p) if random.random() < mut_prob else p[:] for p in pop]

# implements the evolutionary algorithm
# arguments:
#   pop_size  - the initial population
#   max_gen   - maximum number of generation
#   fitness   - fitness function (takes individual as argument and returns 
#               FitObjPair)
#   operators - list of genetic operators (functions with one arguments - 
#               population; returning a population)
#   mate_sel  - mating selection (funtion with three arguments - population, 
#               fitness values, number of individuals to select; returning the 
#               selected population)
#   map_fn    - function to use to map fitness evaluation over the whole 
#               population (default `map`)
#   log       - a utils.Log structure to log the evolution run
def evolutionary_algorithm(pop, max_gen, fitness, operators, mate_sel, *, map_fn=map, log=None):
    evals = 0
    for G in range(max_gen):
        fits_objs = list(map_fn(fitness, pop))
        evals += len(pop)
        if log:
            log.add_gen(fits_objs, evals)
        fits = [f.fitness for f in fits_objs]
        objs = [f.objective for f in fits_objs]
        best = sorted(pop, key=fitness)[-10:]
        mating_pool = mate_sel(pop, fits, POP_SIZE)
        offspring = mate(mating_pool, operators)
        pop = offspring[:]
        pop[-10:] = best

    return pop

if __name__ == '__main__':
    # read the weights from input
    # run the algorithm `REPEATS` times and remember the best solutions from 
    # last generations
    CX_PROB = 0.85 # crossover probability
    MUT_PROB = 0.8 # mutation probability
    MUT_FLIP_PROB = 0.03 # probability of chaninging value during mutation
    for i in range(1):
        #MUT_FLIP_PROB += 0.1
        best_inds = []
        # use `functool.partial` to create fix some arguments of the functions 
        # and create functions with required signatures
        weights = read_weights('inputs/partition-easy.txt')
        cr_ind = functools.partial(create_ind, ind_len=len(weights))
        fit = functools.partial(fitness, weights=weights)
        fit_avg = functools.partial(fitness_with_avg, weights=weights)
        fit_total = functools.partial(fitness_with_total_diff, weights=weights)
        xover = functools.partial(crossover, cross=one_pt_cross, cx_prob=CX_PROB)
        informed_xover = functools.partial(crossover, cross=one_pt_cross, cx_prob=CX_PROB, weights=weights)
        mut = functools.partial(mutation, mut_prob=MUT_PROB, 
                                mutate=functools.partial(flip_mutate, prob=MUT_FLIP_PROB, upper=K))
        informed_mut = functools.partial(mutation, mut_prob=MUT_PROB, 
                                mutate=functools.partial(flip_to_lowest_mutate, prob=MAX_MIN_MUT_PROB, weights=weights))
        fit_mut = functools.partial(mutation, mut_prob=MUT_PROB, 
                                mutate=functools.partial(flip_mutate_informed, prob=MUT_FLIP_PROB, upper=K, weights=weights))
        # we can use multiprocessing to evaluate fitness in parallel
        import multiprocessing
        pool = multiprocessing.Pool()

        import matplotlib.pyplot as plt
        for run in range(REPEATS):
            # initialize the log structure
            log = utils.Log(OUT_DIR, EXP_ID, run, 
                            write_immediately=True, print_frequency=5)
            # create population
            pop = create_pop(POP_SIZE, cr_ind)
            # run evolution - notice we use the pool.map as the map_fn
            pop = evolutionary_algorithm(pop, MAX_GEN, fit_total, [xover, fit_mut], tour_selection, map_fn=pool.map, log=log)
            # remember the best individual from last generation, save it to file
            bi = max(pop, key=fit)
            best_inds.append(bi)

            #with open(f'{OUT_DIR}/{EXP_ID}_{run}.best', 'w') as f:
            #    for w, b in zip(weights, bi):
            #        f.write(f'{w} {b}\n')
            
            # if we used write_immediately = False, we would need to save the 
            # files now
            # log.write_files()

        # print an overview of the best individuals from each run
        diffs = []
        for i, bi in enumerate(best_inds):
            diffs.append(fit(bi).objective)
            print(f'Run {i}: difference = {fit(bi).objective}, bin weights = {bin_weights(weights, bi)}')
        print(f'Average difference = {np.average(diffs)}')
        print(f'For MUT_FLIP_PROB = {MUT_FLIP_PROB}')
        # write summary logs for the whole experiment
        utils.summarize_experiment(OUT_DIR, EXP_ID)

        # read the summary log and plot the experiment
        evals, lower, mean, upper = utils.get_plot_data(OUT_DIR, EXP_ID)
        plt.figure(figsize=(12, 8))
        utils.plot_experiment(evals, lower, mean, upper, legend_name = 'Default settings')
        plt.legend()
        #plt.show()

    # you can also plot mutiple experiments at the same time using 
    # utils.plot_experiments, e.g. if you have two experiments 'default' and 
    # 'tuned' both in the 'partition' directory, you can call
    # utils.plot_experiments('partition', ['default', 'tuned'], 
    #                        rename_dict={'default': 'Default setting'})
    # the rename_dict can be used to make reasonable entries in the legend - 
    # experiments that are not in the dict use their id (in this case, the 
    # legend entries would be 'Default settings' and 'tuned') 