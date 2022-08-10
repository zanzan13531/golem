import torch

# I guess everything uses the pytorch datastructure of a Tensor? (basically a matrix)

# B = weighted adjacency matrix
# x = set of variables being represented by the DAG
# !!! assuming both B and x are square and of the same dimensions?
def golemify(B, x): 

    # first half: 
    firstHalf = torch.tensor()

    n=1; # what does n equal?
    d = x.size(dim = 0) # d : size of the set of X

    sum = torch.tensor() # initlaizing sum

    for i in range (d): # outer sum
        for k in range(n): # inner sum
            sum = torch.add(sum, 0) # replace the 0 with the actual sum contents, no idea if this is a matrix of a scalar

    logSum = torch.log(sum) # log(sum stuff) : log of the sum stuff

    firstHalf = torch.mul(sum, d/2.0) # d*log(sum stuff)/2 : first half of the equation, not sure if this is a scalar

    # second half:

    I = torch.eye(d) # I : creating identity matrix of the same dimention as B
    inner = torch.sub(I, B)  # I - B : subtracting B from the identity matrix
    secondHalf = torch.logdet(inner) # log|det(I - B)| : I think this is a scalar?

    return(torch.sub(firstHalf, secondHalf))

