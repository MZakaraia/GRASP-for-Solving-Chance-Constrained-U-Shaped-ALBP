import random as rn
from Multi_Manned_with_U_Shaped_functions import prob_data, SolutionClass
import time
import numpy as np
import math as ma

def GRASP(problem = None, ct = None, alpha = None, swapping_rate = 0.2, n = 20, m = 30, mutation_fn = 'FS',varsize = 4):
    prob = prob_data(problem)
    problem_data = prob[0]
    tasks = [x for x in problem_data]
    sol_structure = rn.sample(tasks,len(tasks))
    total_area = prob[1]
    best_solution_eval = float('inf')
    # Mutation function selection
    if mutation_fn == 'FS':
        sw_fn = lambda solution, sw_rate, m: solution.forward_mutation(sw_rate, m)
    elif mutation_fn == 'BS':
        sw_fn = lambda solution, sw_rate, m: solution.Backward_mutation(sw_rate, m)
    else:
        sw_fn = lambda solution, sw_rate, m: solution.bi_directional_mutation(sw_rate, m)
    # Calculate lower bound
    LB = 0
    for i in prob[0]:
        LB = LB + prob[0][i]['Processing time']
    LB = ma.ceil(LB / ct)
    # Apply modified Carraway method to generate variance for each task
    for i in prob[0]:
        prob[0][i]['Variance'] = rn.random() * ((prob[0][i]['Processing time']/varsize) ** 2)
        ct_test = prob[0][i]['Processing time'] + 1.96 * np.sqrt(prob[0][i]['Variance'])
        if ct_test > ct:
            prob[0][i]['Variance'] = ((ct - prob[0][i]['Processing time']) / 1.96) ** 2
    start = time.time()
    for j in range(n):	
        sol_structure = rn.sample(tasks,len(tasks))	
        sol = SolutionClass(ct, total_area, alpha, problem_data, sol_structure,1)
        # print(j)
        # sol.forward_mutation(swapping_rate, m)
        sw_fn(sol, swapping_rate, m)
        sol.mutated_solutions.append(sol.solution)
        # Generate descendants
        k = 0		
        for i in sol.mutated_solutions:		
            if i[1] < best_solution_eval:
                best_solution = i
                # best_struct = sol.mutated_structures[k]
                best_solution_eval = i[1]
                end = time.time() - start
                # print(i[2], i[3])
            k += 1
    return best_solution, end, LB
