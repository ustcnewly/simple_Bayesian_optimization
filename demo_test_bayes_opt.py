import random
import numpy as np
from collections import OrderedDict
from bayes_opt import BayesianOptimization

results = np.loadtxt('input_complete_results.txt')
param_names = ['alpha', 'beta']
nparam = len(param_names)
assert(nparam == results.shape[1]-1)
params = np.reshape(results[:,:-1], (results.shape[0], nparam))
max_acc = np.max(results[:, -1]) 

acc_dict = dict()
for iline in range(results.shape[0]):    
    acc_dict[np.array(results[iline, :-1]).tostring()] = results[iline][-1]

def target(**inargs):
    ordered_values = [inargs[param_name] for param_name in param_names]
    return  acc_dict[np.array(ordered_values).tostring()]

init_dict = OrderedDict()
for i, param_name in enumerate(param_names):
    init_dict[param_name] = (min(params[:,i]), max(params[:,i]))
     
init_sample_num = 2
bo = BayesianOptimization(target,   init_dict, verbose=0)
 
fit_indices = set()
sample_combo = random.sample(list(enumerate(params)), init_sample_num)
sample_indices, sample_params = zip(*sample_combo)
fit_indices = fit_indices.union(sample_indices)

param_dict = OrderedDict()
for i, param_name in enumerate(param_names):
    param_dict[param_name] = [sample[i] for sample in sample_params]    
bo.explore(param_dict)

for iround in range(results.shape[0]-init_sample_num):
    #you can tune the gp parameters and bo parameters as follows
    #gp_params = {'kernel': None, 'alpha': 1e-5}
    #bo.maximize(init_points=0, n_iter=0, acq='cub', kappa=5,  **gp_params)
    bo.maximize(init_points=0, n_iter=0)
    utility = bo.util.utility(params, bo.gp, 0) 

    sort_indices = np.argsort(utility)
    sort_indices = sort_indices[:: -1]
    # remove the explored parameters
    sort_indices = [x for x in sort_indices if x not in fit_indices]
    add_index = sort_indices[0]
    fit_indices.add(add_index)
    add_param = params[add_index]
    
    acc = acc_dict[np.array(add_param).tostring()]
    if acc>max_acc*0.9:
        print 'Take %d rounds to find the accuracy %f' %(iround, acc)
        break  
    
    for i, param_name in enumerate(param_names):
        param_dict[param_name] = [add_param[i]]    
    bo.explore(param_dict, eager=True)

 
