import torch
import torchvision
import torchvision.transforms as transforms

from torch.utils.data import DataLoader


class golem():
    def __init__(self, *, lambda1=1, lambda2=1, learningRate=0.001):
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.learningRate = learningRate
        self.batchSize = 512

        self.net = None

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = 'cpu'


    class Net(torch.nn.Module):
        def __init__(self, input_dim):
            super().__init__()
            self.L = torch.nn.Linear(input_dim, input_dim, bias = False)

        def forward(self, x):
            x = self.L(x)
            return x
    

    # B = weighted adjacency matrix
    # x = set of variables being represented by the DAG
    # !!! assuming both B is a square of d by d size, and that x is a 2D matrix with dimensions d by x, where there are d variables and n cases of each (basically data points)
    def L1(self, B, x): 

        # first half: 
        d = x.size(dim = 1) # d : number of variables in x
        n = x.size(dim = 0); # n : number of data points per variable in x


        doubleSum = 0 # initlaizing sum

        # !!! this assumes that the sum results in a scalar, not sure if that's right or not
        for i in range (d): # outer sum

            for k in range(n): # inner sum

                xki = x[k, i] # x^k_i : first part which takes the element at index [k, i] of x

                bi = B[:, i] # B_i : ith column of B
                #bti = torch.transpose(bi, 0, 1) # B^T_i : ith column of B transposed (rotated 90 degrees)

                #xk = x[:, k] # x^k : kth column of x   
                xk = x[k, :]                                                                      

                btixk = torch.mul(bi, xk) # B^T_i * x^k : matrix multiplication of the two previous parts
                xki_btixk = xki - btixk # x^k_i - B^T_i * x^k : entire thing except for the square
                xki_btixk_squared = xki_btixk ** 2.0 # squared
                doubleSum = doubleSum + torch.log(xki_btixk_squared) # takes the log of the inner sum and adds it to the total sum

        # second half:
        I = torch.eye(d) # I : creating identity matrix of the same dimention as B
        IB = torch.sub(I, B)  # I - B : subtracting B from the identity matrix
        IB = torch.abs(IB)
        logDetIB = torch.logdet(IB) # log|det(I - B)| : I think this is a scalar?

        L1Result = doubleSum / 2.0 - logDetIB # final result of the L2 function

        return(L1Result)


    # B = weighted adjacency matrix
    # x = set of variables being represented by the DAG
    # !!! assuming both B is a square of d by d size, and that x is a 2D matrix with dimensions d by x, where there are d variables and n cases of each (basically data points)
    def L2(self, B, x): 

        # first half: 
        d = x.size(dim = 1) # d : number of variables in x (dimension of b)
        n = x.size(dim = 0); # n : number of data points per variable in x

        doubleSum = 0 # initlaizing sum

        # !!! this assumes that the sum results in a scalar, not sure if that's right or not
        for i in range (d): # outer sum
            for k in range(n): # inner sum

                xki = x[k, i] # x^k_i : first part which takes the element at index [k, i] of x

                bi = B[:, i] # B_i : ith column of B
                #bti = torch.transpose(bi, 0, 1) # B^T_i : ith column of B transposed (rotated 90 degrees)

                #xk = x[:, k] # x^k : kth column of x   
                xk = x[k, :]                                        

                btixk = torch.mul(bi, xk) # B^T_i * x^k : matrix multiplication of the two previous parts
                xki_btixk = xki - btixk # x^k_i - B^T_i * x^k : entire thing except for the square
                xki_btixk_squared = xki_btixk ** 2.0 # squared
                doubleSum = doubleSum + xki_btixk_squared # adds the inner stuff to them sum

        logDoubleSum = torch.log(doubleSum) # log(sum stuff) : log of the sum stuff

        # second half:
        I = torch.eye(d) # I : creating identity matrix of the same dimention as B
        IB = torch.sub(I, B)  # I - B : subtracting B from the identity matrix
        IB = torch.abs(IB)
        logDetIB = torch.logdet(IB) # log|det(I - B)| : I think this is a scalar?

        L2Result = logDoubleSum * d / 2.0 - logDetIB # final result of the L2 function

        return(L2Result)


    def h(self, B): #characterization of DAGness function
        return (torch.trace(torch.matrix_exp(torch.mul(B, B))) - B.size(dim = 0)) # tr(e^(B o B)) - d : trace of the matrix exponential of the hadamard product of B and itself minus d

    def scoreFunction1(self, B, x):
        return (self.L1(B, x) + self.lambda1 * B.norm(1) + self.lambda2 * self.h(B)) # S2(B, x) = L2(B, x) + lambda1 * ||B||_1 + lambda2 * h(B) : first score function

    def scoreFunction2(self, B, x):
        return (self.L2(B, x) + self.lambda1 * B.norm(1) + self.lambda2 * self.h(B)) # S2(B, x) = L2(B, x) + lambda1 * ||B||_1 + lambda2 * h(B) : second score function

    def train(self, dataset, epochs=1, scoreFunction=None): #should be batches by x (# of batches by d by n)
        self.net = self.Net(dataset.size(dim=1))
        self.net = self.net.to(self.device)

        if (dataset.dim() == 2) :
            dataset = dataset[None, :]

        #next(net.parameter())

        optimizer = torch.optim.SGD(self.net.parameters(), lr=self.learningRate)

        self.net.train()

        for x in range(epochs):
            for i, data in enumerate(dataset): #data is the same thing as x
                data = data.to(self.device)
                optimizer.zero_grad()

                #output = net(data)

                B = self.net.L.weight

                #print(B)
            
                score = None
                if (scoreFunction == 1):
                    score = self.scoreFunction1(B, data).sum()
                else:
                    score = self.scoreFunction2(B, data).sum()

                score.backward()
    
                optimizer.step()
            
                print(f'current batch score for epoch {x} batch {i} is {score}')

    def getModel(self):
        return(self.net.L.weight)

    def train2(self, dataset, epochs=1, scoreFunction=None): #should be batches by x (# of batches by d by n)
        self.net = self.Net(dataset.size(dim=1))
        self.net = self.net.to(self.device)

        dataloader = DataLoader(dataset=dataset, batch_size=self.batchSize)

        #next(net.parameter())

        optimizer = torch.optim.SGD(self.net.parameters(), lr=self.learningRate)

        self.net.train()

        for x in range(epochs):
            for i, data in enumerate(dataloader): #data is the same thing as x
                data = data.to(self.device)
                optimizer.zero_grad()

                #output = net(data)

                B = self.net.L.weight

                #print(B)
            
                score = None
                if (scoreFunction == 1):
                    score = self.scoreFunction1(B, data).sum()
                else:
                    score = self.scoreFunction2(B, data).sum()

                score.backward()
    
                optimizer.step()
            
                print(f'current batch score for epoch {x} batch {i} is {score}')