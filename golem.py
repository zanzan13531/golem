import torch
import torchvision
import torchvision.transforms as transforms
import math


    
def netFunction():
    print("placeholder")

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

def h(B): #characterization of DAGness function
    return (torch.trace(torch.matrix_exp(torch.mul(B, B))) - B.size(dim = 0)) # tr(e^(B o B)) - d : trace of the matrix exponential of the hadamard product of B and itself

def scoreFunction2(B, x):
    return (L2(B, x) + torch.norm(B) + h(B))


learning_rate = 0.1


transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
])

# 1
train_dataset = torchvision.datasets.CIFAR10(root='../data/', train=True, download=True, transform=transform_train)
test_dataset = torchvision.datasets.CIFAR10(root='../data/', train=False, download=True, transform=transform_test)

# 2
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=4)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=512, shuffle=False, num_workers=4)


# 3
net = netFunction() #should be net function
net = torch.nn.DataParallel(net)

# 4
optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate)


def train(epoch):
    print(f'\n[ Train epoch: {epoch} ]')
    net.train()
    train_loss = 0
    correct = 0
    # total = 0
    total = len(train_dataset)
    # 5
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        #  6
        optimizer.zero_grad()

        # 7
        benign_outputs = net(inputs)

        # 8 
        loss = scoreFunction2(benign_outputs, targets)

        # 9
        loss.backward() # calculate gradients

        # 10
        optimizer.step() # perform gradient descent to minimize the loss functions


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

