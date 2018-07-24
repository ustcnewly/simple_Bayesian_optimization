import sys
import numpy as np

def main():
    alpha = float(sys.argv[1])
    beta = float(sys.argv[2])
    params = [alpha, beta]
    
    ###################################
    # implement your own function here
    ###################################
    results = np.loadtxt('input_complete_results.txt')
    acc_dict = dict()
    for iline in range(results.shape[0]):    
        acc_dict[np.array(results[iline, :-1]).tostring()] = results[iline][-1]    
    acc = acc_dict[np.array(params).tostring()]
    ###################################
    
    fid = open('new_result.txt', 'w')
    fid.write('%f'%acc)
    fid.close()
    
if __name__ == "__main__":
    main()