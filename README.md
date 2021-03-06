This is a simple Bayesian optimization tool for hyper-parameter tuning based on Gaussian process. The code is modified based on https://github.com/fmfn/BayesianOptimization. You can also refer to this [blog](https://ustcnewly.github.io/2018/07/18/paper_note/Simple%20Bayesian%20Optimization/) for more details.

**Requirements:**

* numpy: pip install numpy

* bayes_opt: pip install bayesian-optimization

**Preparation:**

* modify 'parameter_names', 'parameter_ranges', and input/output file paths in the demo file.

* prepare parameter-performance sheet like 'input_complete_results.txt' and 'input_partial_results.txt', in which each line contains one group of parameter values and performance. The last column is the performance and other columns are parameter values.

**Run:**

* 'demo_test_bayes_opt.py' is used to simulate how fast bayes optimization can locate the optimal parameter given all the results under all the parameter settings. 

* 'demo_test_bayes_opt.py' is used to recommend the next parameter to try, given a history of parameter-performance pairs.

* 'demo_run_bayes_opt.py' is used to automatically tune the hyper-parameters given the initial results under at least 2 parameter settings. Replace the function in simulate_func.py with your own, in which the input are hyper-parameters and the output is performance. 

**Note:**

* you can try tuning the hyper parameters for Gaussian process and Bayesian optimization in the function 'bo.maximize(init_points=0, n_iter=0)'.

* you can try different initial hyper-parameters.