import numpy as np
import matplotlib.pyplot as plt
import mlrose_hiive as mlrose

'''
f(x)=  sin(pi*x)  0<=x<=1
       0.5*|sin(2pi*x)| 1<=x<2
       -3*(x-3)**2+3    2<=x<4
       4*x - 16         4<=x<5
       -4*x+24          5<=x<6
       3|sin(pi*x)|     6<=x<8
       -2*(x-9)**2+2    8<=x<=10
'''
count=0
def n_peak_fit(state):
    global count
    count+=1
    '''
    elif state_value<5:
        value=4*state_value-16
    elif state_value<6:
        value=-4*state_value+24
    '''
    value=None
    state_value=state[0]*1.e-3
    if state_value<1:
        value=np.sin(np.pi*state_value)
    elif state_value<2:
        value=abs(np.sin(2*np.pi*state_value))*0.5
    elif state_value<4:
        value=-3*(state_value-3)**2+3
    elif state_value<5:
        value=5*state_value - 20
    elif state_value<6:
        value=-5*state_value + 30
    elif state_value<8:
        value=3*abs(np.sin(np.pi*state_value))
    else:
        value=-2*(state_value-9)**2+2
    return value

def plot_function():
    x_list=np.arange(10000)
    func_value=[]
    for x in x_list:
        func_value.append(n_peak_fit([x]))
    plt.plot(x_list,func_value)
    plt.show()

def method_compare():
    global count
    count=0
    fitness_obj=mlrose.CustomFitness(n_peak_fit)
    opt=mlrose.DiscreteOpt(1,fitness_obj,maximize=True,max_val=10000)
    
    best_state_climb,best_fitness_climb,fitness_curve_climb=mlrose.random_hill_climb(opt,curve=True)
    print('---------------------random hill climb-------------------------')
    print('hill climbing best state for n-peak problem:',best_state_climb)
    print('hill climbing best fitness for n-peak problem:',best_fitness_climb)
    print('hill climbing fitting curve for n-peak problem:',fitness_curve_climb)
    print('number of fitness call used:',count)
    count=0
    print('-------------------simulated annealing-------------------------')
    best_state_ann,best_fitness_ann,fitness_curve_ann=mlrose.simulated_annealing(opt,schedule=mlrose.ExpDecay(),curve=True)
    print('simulated annealing best state for n-peak problem:',best_state_ann)
    print('simulated annealing best fitness for n-peak problem:',best_fitness_ann)
    print('simulated annealing fitting curve for n-peak problem:',fitness_curve_ann)
    print('number of fitness call used:',count)
    count=0
    best_state_ga,best_fitness_ga,fitness_curve_ga=mlrose.genetic_alg(opt,pop_size=20, mutation_prob=0.5,curve=True)
    print('---------------------genetic alg----------------------------')
    print('genetic algorithm best state for n-peak problem:',best_state_ga)
    print('genetic algorithm best fitnees for n-peak problem:',best_fitness_ga)
    print('genetic algorithm fitness curve for n-peak problem:',fitness_curve_ga)
    print('number of fitness call used:',count)
    count=0
    best_state_mimic,best_fitness_mimic,fitness_curve_mimic=mlrose.mimic(opt,pop_size=20,curve=True)
    print('------------------------mimic-------------------------------')
    print('mimic best state for n-peak problem:',best_state_mimic)
    print('mimic best fitness value for n-peak problem:',best_fitness_mimic)
    print('mimic curve for n-peak problem:',fitness_curve_mimic)
    print('number of fitness calls used:',count)
    count=0
    plt.figure(figsize=(10,10))
    plt.subplot(221)
    plt.plot(fitness_curve_climb)
    plt.ylabel('fitness')
    plt.xlabel('num_iter')
    plt.ylim(0,5)
    plt.title('random hill climb')
    plt.subplot(222)
    plt.plot(fitness_curve_ann)
    plt.ylabel('fitness')
    plt.xlabel('num_iter')
    plt.ylim(0,5)
    plt.title('simulated annealing')
    plt.subplot(223)
    plt.plot(fitness_curve_ga)
    plt.ylim(0,5)
    plt.ylabel('fitness')
    plt.xlabel('num_iter')
    plt.title('genetic algorithm')
    plt.subplot(224)
    plt.plot(fitness_curve_mimic)
    plt.ylim(0,5)
    plt.title('mimic')
    plt.ylabel('fitness')
    plt.xlabel('num_iter')
    plt.show()
    
def compare_gen_mimic():
    fitness_obj=mlrose.CustomFitness(n_peak_fit)
    opt=mlrose.DiscreteOpt(1,fitness_obj,maximize=True,max_val=10000)
    global count
    count=0
    iter_num_counter_ga=[]
    fitness_list_ga=[]
    iter_num_counter_mimic=[]
    fitness_list_mimic=[]
    for i in range(20):
        best_state_ga,best_fitness_ga,fitness_curve_ga=mlrose.genetic_alg(opt,pop_size=20, mutation_prob=0.5,curve=True)
        iter_num_counter_ga.append(count)
        fitness_list_ga.append(best_fitness_ga)
        count=0
        best_state_mimic,best_fitness_mimic,fitness_curve_mimic=mlrose.mimic(opt,pop_size=20,curve=True)
        iter_num_counter_mimic.append(count)
        fitness_list_mimic.append(best_fitness_mimic)
        count=0
    plt.figure(figsize=(8,6))
    plt.subplot(121)
    plt.plot(fitness_list_ga,label='ga')
    plt.plot(fitness_list_mimic,label='mimic')
    plt.xlabel('rounds')
    plt.ylabel('finess value')
    plt.title('fitness value comparision')
    plt.legend(loc='lower right')
    plt.subplot(122)
    plt.plot(iter_num_counter_ga,label='ga')
    plt.plot(iter_num_counter_mimic,label='mimic')
    plt.xlabel('rounds')
    plt.ylabel('fitness call no.')
    plt.title('fitness call number comparision')
    plt.legend(loc='upper right')
    plt.show()
    
    
if __name__=='__main__':
    plot_function()
    method_compare()
    compare_gen_mimic()