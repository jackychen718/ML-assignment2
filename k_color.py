import mlrose_hiive as mlrose
import numpy as np
import matplotlib.pyplot as plt

def generate_map(n_block,epsilon):
    orig_map=np.random.rand(n_block,n_block)
    sparse_map=orig_map<epsilon
    result_map=np.zeros((n_block,n_block))
    for i in range(n_block):
        for j in range(i+1,n_block):
            if sparse_map[i][j]:
                result_map[i][j]=1
                #result_map[j][i]=1
    return result_map



def homo_string_fn(state):
    global count
    count+=1
    fitness=0
    for n in range(len(state)):
        temp_fit=np.sum(state==n)
        if temp_fit>fitness:
            fitness=temp_fit
    return fitness


def read_map(map_name):
    handle=open(map_name,'r')
    content=handle.read()
    handle.close()
    rows=content.split('\n')
    neighbor_map=np.array([[i for i in r.split(',')] for r in rows if r],dtype=int)
    return neighbor_map

neighbor_map=read_map('map1.csv')
count=0
def k_color_fit(state):
    global count
    count+=1
    fitness_value=0
    for i in range(len(state)):
        for j in range(i+1,len(state)):
            if neighbor_map[i][j]==1 and state[i]==state[j]:
                fitness_value-=1
    return fitness_value
                

def method_compare():
    global count
    count=0
    fitness_obj=mlrose.CustomFitness(k_color_fit)
    opt=mlrose.DiscreteOpt(50,fitness_obj,maximize=True,max_val=8)
    best_state_climb,best_fitness_climb,fitness_curve_climb=mlrose.random_hill_climb(opt,curve=True)
    print('---------------------random hill climb-------------------------')
    print('hill climbing best state for k-color problem:',best_state_climb)
    print('hill climbing best fitness for k-color problem:',best_fitness_climb)
    print('hill climbing fitting curve for k-color problem:',fitness_curve_climb)
    print('number of fitness call used:',count)
    count=0
    print('-------------------simulated annealing-------------------------')
    best_state_ann,best_fitness_ann,fitness_curve_ann=mlrose.simulated_annealing(opt,schedule=mlrose.ExpDecay(),curve=True)
    print('simulated annealing best state for k-color problem:',best_state_ann)
    print('simulated annealing best fitness for k-color problem:',best_fitness_ann)
    print('simulated annealing fitting curve for k-color problem:',fitness_curve_ann)
    print('number of fitness call used:',count)
    count=0
    best_state_ga,best_fitness_ga,fitness_curve_ga=mlrose.genetic_alg(opt,pop_size=500, mutation_prob=0.2,curve=True)
    print('---------------------genetic alg----------------------------')
    print('genetic algorithm best state for k-color problem:',best_state_ga)
    print('genetic algorithm best fitnees for k-color problem:',best_fitness_ga)
    print('genetic algorithm fitness curve for k-color problem:',fitness_curve_ga)
    print('number of fitness call used:',count)
    count=0
    best_state_mimic,best_fitness_mimic,fitness_curve_mimic=mlrose.mimic(opt,pop_size=500,curve=True)
    print('------------------------mimic-------------------------------')
    print('mimic best state for k-color problem:',best_state_mimic)
    print('mimic best fitness value for k-color problem:',best_fitness_mimic)
    print('mimic curve for k-color problem:',fitness_curve_mimic)
    print('number of fitness call used:',count)
    count=0
    plt.figure(figsize=(12,10))
    plt.subplot(221)
    plt.plot(fitness_curve_climb)
    plt.ylabel('fitness')
    plt.xlabel('num_iter')
    plt.ylim(-30,0)
    plt.title('random hill climb')
    plt.subplot(222)
    plt.plot(fitness_curve_ann)
    plt.ylabel('fitness')
    plt.xlabel('num_iter')
    plt.ylim(-30,0)
    plt.title('simulated annealing')
    plt.subplot(223)
    plt.plot(fitness_curve_ga)
    plt.ylim(-30,0)
    plt.ylabel('fitness')
    plt.xlabel('num_iter')
    plt.title('genetic algorithm')
    plt.subplot(224)
    plt.plot(fitness_curve_mimic)
    plt.title('mimic')
    plt.ylabel('fitness')
    plt.xlabel('num_iter')
    plt.ylim(-30,0)
    plt.show()
    

def compare_multi_round_k_color():
    global count
    count=0
    fitness_obj=mlrose.CustomFitness(k_color_fit)
    opt=mlrose.DiscreteOpt(50,fitness_obj,maximize=True,max_val=8)
    fitness_list_rhc=[]
    fitness_list_ann=[]
    fitness_list_genetic=[]
    fitness_list_mimic=[]
    num_sample_rhc=[]
    num_sample_ann=[]
    num_sample_genetic=[]
    num_sample_mimic=[]
    for i in range(20):
        best_state_climb,best_fitness_climb,fitness_curve_climb=mlrose.random_hill_climb(opt,curve=True)
        fitness_list_rhc.append(best_fitness_climb)
        num_sample_rhc.append(count)
        count=0
        best_state_ann,best_fitness_ann,fitness_curve_ann=mlrose.simulated_annealing(opt,schedule=mlrose.ExpDecay(),curve=True)
        fitness_list_ann.append(best_fitness_ann)
        num_sample_ann.append(count)
        count=0
        best_state_ga,best_fitness_ga,fitness_curve_ga=mlrose.genetic_alg(opt,pop_size=500, mutation_prob=0.5,curve=True)
        fitness_list_genetic.append(best_fitness_ga)
        num_sample_genetic.append(count)
        count=0
        best_state_mimic,best_fitness_mimic,fitness_curve_mimic=mlrose.mimic(opt,pop_size=500,curve=True)
        fitness_list_mimic.append(best_fitness_mimic)
        num_sample_mimic.append(count)
        count=0
    plt.figure(figsize=(10,6))
    plt.subplot(121)
    plt.plot(fitness_list_rhc,label='rhc')
    plt.plot(fitness_list_ann,label='ann')
    plt.plot(fitness_list_genetic,label='ga')
    plt.plot(fitness_list_mimic,label='mimic')
    plt.xlabel('rounds')
    plt.ylabel('finess value')
    plt.title('fitness value comparision')
    plt.legend(loc='lower right')
    plt.subplot(122)
    plt.plot(num_sample_rhc,label='rhc')
    plt.plot(num_sample_ann,label='ann')
    plt.plot(num_sample_genetic,label='ga')
    plt.plot(num_sample_mimic,label='mimic')
    plt.xlabel('rounds')
    plt.ylabel('fitness calls')
    plt.title('fitness call number comparision')
    plt.legend(loc='upper right')
    plt.show()


if __name__=='__main__':
    method_compare()
    compare_multi_round_k_color()