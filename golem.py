import torch

# I guess everything uses the pytorch datastructure of a Tensor? (basically a matrix)

# B = weighted adjacency matrix
# x = set of variables being represented by the DAG
# !!! assuming both B is a square of d by d size, and that x is a 2D matrix with dimensions d by x, where there are d variables and n cases of each (basically data points)
def golemify(B, x): 

    # first half: 
    firstHalf = torch.tensor()

    d = x.size(dim = 0) # d : number of variables in x
    n = x.size(dim = 1); # n : number of data points per variable in x

    sum = 0 # initlaizing sum

    # !!! this assumes that the sum results in a scalar, not sure if that's right or not
    for i in range (d): # outer sum
        for k in range(n): # inner sum

            innerThingA = x[i, k] # x^k_i : first part which takes the element at index [k, i] of x

            innerThingB = B[:, i] # B_i : ith column of B
            innerThingB = torch.transpose(innerThingB, 0, 1) # B^T_i : ith column of B transposed (rotated 90 degrees)

            innerThingC = x[k, :] # x^k : kth column of x                                                                       

            innerThing = torch.mul(innerThingB, innerThingC) # B^T_i * x^k : matrix multiplication of the two previous parts
            innerThing = innerThingA - innerThingB # x^k_i - B^T_i * x^k : entire thing except for the square
            innerThing = innerThing ** 2.0 # squared
            sum = torch.add(sum, innerThing) # adds the inner stuff to them sum

    logSum = torch.log(sum) # log(sum stuff) : log of the sum stuff

    firstHalf = torch.mul(logSum, d/2.0) # d*log(sum stuff)/2 : first half of the equation, pretty sure if this is a scalar

    # second half:

    I = torch.eye(d) # I : creating identity matrix of the same dimention as B
    inner = torch.sub(I, B)  # I - B : subtracting B from the identity matrix
    secondHalf = torch.logdet(inner) # log|det(I - B)| : I think this is a scalar?

    return(torch.sub(firstHalf, secondHalf))

