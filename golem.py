import torch
import math

# I guess everything uses the pytorch datastructure of a Tensor? (basically a matrix)

# B = weighted adjacency matrix
# x = set of variables being represented by the DAG
# !!! assuming both B is a square of d by d size, and that x is a 2D matrix with dimensions d by x, where there are d variables and n cases of each (basically data points)
def L2(B, x): 

    # first half: 
    d = x.size(dim = 0) # d : number of variables in x
    n = x.size(dim = 1); # n : number of data points per variable in x

    doubleSum = 0 # initlaizing sum

    # !!! this assumes that the sum results in a scalar, not sure if that's right or not
    for i in range (d): # outer sum
        for k in range(n): # inner sum

            xki = x[i, k] # x^k_i : first part which takes the element at index [k, i] of x

            bi = B[:, i] # B_i : ith column of B
            bti = torch.transpose(bi, 0, 1) # B^T_i : ith column of B transposed (rotated 90 degrees)

            xk = x[k, :] # x^k : kth column of x                                                                       

            btixk = torch.mul(bti, xk) # B^T_i * x^k : matrix multiplication of the two previous parts
            xki_btixk = xki - btixk # x^k_i - B^T_i * x^k : entire thing except for the square
            xki_btixk_squared = xki_btixk ** 2.0 # squared
            doubleSum = doubleSum + xki_btixk_squared # adds the inner stuff to them sum

    logDoubleSum = math.log(doubleSum) # log(sum stuff) : log of the sum stuff

    # second half:
    I = torch.eye(d) # I : creating identity matrix of the same dimention as B
    IB = torch.sub(I, B)  # I - B : subtracting B from the identity matrix
    logDetIB = torch.logdet(IB) # log|det(I - B)| : I think this is a scalar?

    L2Result = logDoubleSum * d / 2.0 - logDetIB # final result of the L2 function

    return(L2Result)

