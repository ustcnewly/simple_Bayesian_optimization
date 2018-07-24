import numpy as np
from collections import OrderedDict
from bayes_opt import BayesianOptimization

results = np.loadtxt('input_partial_results.txt')
param_names = ['alpha', 'beta']

param_ranges = [pow(10, np.arange(-5.0, 6.0, 1.0)), pow(10, np.arange(-5.0, 6.0, 1.0))]
# the number of recommended parameters
noutput = 10
nparam = len(param_names)
assert(nparam == results.shape[1]-1)

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
    return  acc_dict[np.array(ordered_values).tostring()]
    
init_dict = OrderedDict()
for i, param_name in enumerate(param_names):
    init_dict[param_name] = (min(param_ranges[i]), max(param_ranges[i]))
     
bo = BayesianOptimization(target,   init_dict, verbose=0)

done_params = np.reshape(results[:,:-1], (results.shape[0], nparam))
param_dict = OrderedDict()
for i, param_name in enumerate(param_names):
    param_dict[param_name] = done_params[:, i]    
bo.explore(param_dict)

#you can tune the gp parameters and bo parameters as follows
#gp_params = {'kernel': None, 'alpha': 1e-5}
#bo.maximize(init_points=0, n_iter=0, acq='cub', kappa=5,  **gp_params)
bo.maximize(init_points=0, n_iter=0)
utility = bo.util.utility(all_params, bo.gp, 0)

sort_indices = np.argsort(utility)
sort_indices = sort_indices[:: -1]

fid = open('output_params.txt', 'w')
icount = 0
for tmp_index in sort_indices:
    tmp_param = all_params[tmp_index]
    if not acc_dict.has_key(np.array(tmp_param).tostring()):
        tmp_param = [str(param) for param in tmp_param]
        fid.write(' '.join(tmp_param)+'\n')        
        icount = icount+1
        if icount>=noutput:
            break
fid.close()
print 'Save %d recommended parameters to output_params.txt' %noutput

     
