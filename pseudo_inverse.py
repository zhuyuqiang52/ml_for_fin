import numpy as np
from scipy import linalg

def mp_pseudoinv(mx_array: np.array):
    U, s, V = linalg.svd(mx_array, full_matrices=False)
    s_inv = 1/s
    s_inv[np.isinf(s_inv)] = 0
    s_inv = np.diag(s_inv)
    pseudo_inv = np.matmul(np.matmul(V.T,s_inv),U.T)
    return pseudo_inv

def main():
    # test
    N_int = 1000
    count_int = 0
    for i in range(N_int):
        #random size
        row_int = np.random.randint(5,100)
        #random diagnoal matrix
        diag_ele_array = np.random.randint(low = 3,high = 1000,size = row_int)
        diag_array = np.diag(diag_ele_array)
        diag_inv_array = mp_pseudoinv(diag_array)
        mul_array = np.matmul(diag_array,diag_inv_array)
        if np.sum(mul_array)-row_int<1e-6:
            count_int+=1
        else:
            print(' wrong')
    print(f'accuracy rate: {count_int/N_int}')
    pass

if __name__ == '__main__':
    main()

