import torch
import torchvision
import torchvision.transforms as transforms



class golem():
    def __init__(self, *, lambda1=1, lambda2=1, learningRate=0.1):
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.learningRate = learningRate

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'


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
    def L1(B, x): 

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
                doubleSum = doubleSum + torch.log(xki_btixk_squared) # takes the log of the inner sum and adds it to the total sum

        # second half:
        I = torch.eye(d) # I : creating identity matrix of the same dimention as B
        IB = torch.sub(I, B)  # I - B : subtracting B from the identity matrix
        logDetIB = torch.logdet(IB) # log|det(I - B)| : I think this is a scalar?

        L1Result = doubleSum / 2.0 - logDetIB # final result of the L2 function

        return(L1Result)


    # B = weighted adjacency matrix
    # x = set of variables being represented by the DAG
    # !!! assuming both B is a square of d by d size, and that x is a 2D matrix with dimensions d by x, where there are d variables and n cases of each (basically data points)
    def L2(B, x): 

        # first half: 
        d = x.size(dim = 0) # d : number of variables in x (dimension of b)
        n = x.size(dim = 1); # n : number of data points per variable in x

        print(d)
        print(n)

        doubleSum = 0 # initlaizing sum

        # !!! this assumes that the sum results in a scalar, not sure if that's right or not
        for i in range (d): # outer sum
            for k in range(n): # inner sum

                xki = x[i, k] # x^k_i : first part which takes the element at index [k, i] of x

                bi = B[:, i] # B_i : ith column of B
                #bti = torch.transpose(bi, 0, 1) # B^T_i : ith column of B transposed (rotated 90 degrees)

                xk = x[k, :] # x^k : kth column of x                                                                       

                btixk = torch.mul(bi, xk) # B^T_i * x^k : matrix multiplication of the two previous parts
                xki_btixk = xki - btixk # x^k_i - B^T_i * x^k : entire thing except for the square
                xki_btixk_squared = xki_btixk ** 2.0 # squared
                doubleSum = doubleSum + xki_btixk_squared # adds the inner stuff to them sum

        logDoubleSum = torch.log(doubleSum) # log(sum stuff) : log of the sum stuff

        # second half:
        I = torch.eye(d) # I : creating identity matrix of the same dimention as B
        IB = torch.sub(I, B)  # I - B : subtracting B from the identity matrix
        logDetIB = torch.logdet(IB) # log|det(I - B)| : I think this is a scalar?

        L2Result = logDoubleSum * d / 2.0 - logDetIB # final result of the L2 function

        return(L2Result)


    def h(B): #characterization of DAGness function
        return (torch.trace(torch.matrix_exp(torch.mul(B, B))) - B.size(dim = 0)) # tr(e^(B o B)) - d : trace of the matrix exponential of the hadamard product of B and itself minus d

    def scoreFunction1(self, B, x):
        return (self.L1(B, x) + self.lambda1 * torch.norm(B) + self.lambda2 * self.h(B)) # S2(B, x) = L2(B, x) + lambda1 * ||B||_1 + lambda2 * h(B) : first score function

    def scoreFunction2(self, B, x):
        return (self.L2(B, x) + self.lambda1 * torch.norm(B) + self.ambda2 * self.h(B)) # S2(B, x) = L2(B, x) + lambda1 * ||B||_1 + lambda2 * h(B) : second score function

    def train(self, dataset): #should be batches by x (# of batches by d by n)
        net = self.Net(dataset.size(dim=1))
        net = net.to(self.device)

        #next(net.parameter())

        optimizer = torch.optim.SGD(net.parameters(), lr=self.learningRate)

        net.train()
        for batch_idx, (target, data) in enumerate(dataset):
            data, target = data.to(self.device), target.to(self.device)
            optimizer.zero_grad()

            #output = net(data)
            #loss = scoreFunction2(output, target)

            B = net.L.weight
            
            score = self.scoreFunction2(B, target)
            score.backward()
            optimizer.step()
            
            print(f'current batch score for batch {batch_idx} is {score}')


    """
    def train(epoch):
        print(f'\n[ Train epoch: {epoch} ]')
        #net.train()
        train_loss = 0
        correct = 0
        total = len(train_dataset)

        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()

            benign_outputs = net(inputs)

            loss = scoreFunction2(benign_outputs, targets)
            loss.backward() # calculate gradients

            optimizer.step() # perform gradient descent to minimize the loss functions



            train_loss += loss.item()
            _, predicted = benign_outputs.max(1)

            correct += predicted.eq(targets).sum().item()

        print('\nTotal benign train accuarcy:', 100. * correct / total)
        print('Total benign train loss:', train_loss)
    """

"""
    def train2(epoch):
        net.train()
        for batch_idx, (target, data) in enumerate(train_loader):
            data = new 
            print(data)
            print(target)
            print(data.size())
            print(target.size())
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            #output = net(data)
            #loss = scoreFunction2(output, target)
            B = net.L.weight
            print(B)
            loss = scoreFunction2(B, target)
            loss.backward()
            optimizer.step()
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
"""

    """
    def train3(epoch):

        for e in range(epoch):
            for i, (data, X)  in enumerate(train_loader):        
                data, X = data.to(device), X.to(device)

                optimizer.zero_grad()

                X_out = net(data) # x_out = x B

                score = 0.5 * torch.log((X_out - X).pow(2).sum(dim=0)).sum() # first part of L_1
                weight = net.L.weight
                score -= torch.det((torch.eye(net.L.weight.size(dim=0)) - weight).abs()) # second part of L_1
                score += lambda1 * weight.norm(1) + lambda2 * h(weight)
            
                score.backward() # calculate gradient
            
                optimizer.step() # doing gradient descent
    """

    """
    def test2(epoch):
        print(f'\n[ Test epoch: {epoch} ]')
        net.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
            
                output = net(data)
                test_loss += scoreFunction2(output, target, reduction='sum').item()  # sum up batch loss
                pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
                correct += pred.eq(target.view_as(pred)).sum().item()

        test_loss /= len(test_loader.dataset)

        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, len(test_loader.dataset),
            100. * correct / len(test_loader.dataset)))
    """
