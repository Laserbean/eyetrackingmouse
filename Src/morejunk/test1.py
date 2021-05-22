
import torch.nn as nn
import torch

class test(torch.nn.Module):
    def __init__(self):
        test.fish = 3; 

    def forward(self,x):
        print(x)
        
        
test1 = test() 
test1(3)