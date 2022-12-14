import Golem
import numpy as np
import torch

npSample = np.loadtxt("sample.txt")

tensorSample = torch.from_numpy(npSample)


testGolem = Golem.golem()
dataset = tensorSample

testGolem.train(dataset, 20, 1)

print(testGolem.getModel())
print(testGolem.getModel().size())

testGolem.train(dataset, 20, 2)

print(testGolem.getModel())
print(testGolem.getModel().size())

npResult = testGolem.getModel().detach().numpy()
np.savetxt("result.txt", npResult)