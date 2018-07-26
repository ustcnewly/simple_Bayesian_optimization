import os
import time
import numpy as np
from collections import OrderedDict
from bayes_opt import BayesianOptimization
from jedi.evaluate import param

results = np.loadtxt('input_partial_results.txt')
param_names = ['alpha', 'beta']
param_ranges = [pow(10, np.arange(-5.0, 6.0, 1.0)), pow(10, np.arange(-5.0, 6.0, 1.0))]
nparam = len(param_names)
assert(nparam == results.shape[1]-1)

max_iter = 10

# generate all parameter candidates
prev_params = [[]]
for i, param_name in enumerate(param_names):
    all_params = []
    for prev_param in prev_params:        
        for new_param in param_ranges[i]:
            combo_param = prev_param+[new_param]
            all_params.append(combo_param)
    prev_params = all_params

acc_dict = dict()
for iline in range(results.shape[0]):    
    acc_dict[np.array(results[iline, :-1]).tostring()] = results[iline][-1]

def target(**inargs):
    ordered_values = [inargs[param_name] for param_name in param_names]
    acc =  acc_dict[np.array(ordered_values).tostring()]
    return acc
    
init_dict = OrderedDict()
for i, param_name in enumerate(param_names):
    init_dict[param_name] = (min(param_ranges[i]), max(param_ranges[i]))
     
bo = BayesianOptimization(target,   init_dict, verbose=0)

done_params = np.reshape(results[:,:-1], (results.shape[0], nparam))
param_dict = OrderedDict()
for i, param_name in enumerate(param_names):
    param_dict[param_name] = done_params[:, i]    
bo.explore(param_dict)

##################################################
# main loop
for iter in range(max_iter):
    #you can tune the gp parameters and bo parameters
    #when acq='ucb', set kappa within [10^-3, 10^-2, ..., 10^3]
    #when acq='poi' or 'ei', set xi within [10^-3, 10^-2, ..., 10^3]
    gp_params = {'kernel': None, 'alpha': 1e-5}
    bo.maximize(init_points=0, n_iter=0, acq='poi', xi=0.01,  **gp_params)
    utility = bo.util.utility(all_params, bo.gp, 0)
    sort_indices = np.argsort(utility)
    sort_indices = sort_indices[:: -1]
    
    for tmp_index in sort_indices:
        next_params = all_params[tmp_index]
        if not acc_dict.has_key(np.array(next_params).tostring()):            
            break

    print next_params
    # evalue the model trained with next_params
    param_str = ' '.join(['%f'%next_param for next_param in next_params])
    cmd  = 'python simulate_func.py %s' %param_str
    os.system(cmd)
    
    result = np.loadtxt('new_result.txt')
    print 'Iter %d: %f' %(iter, result)
    acc_dict[np.array(next_params).tostring()] =  result
     
    # modify input_partial_results.txt
    fid = open('input_partial_results.txt', 'a')
    fid.write('%s %f\n' %(param_str, result))
    fid.close()
    
    # prepare for generating the next point
    for i, param_name in enumerate(param_names):
        param_dict[param_name] = [next_params[i]]    
    bo.explore(param_dict, eager=True)
    
