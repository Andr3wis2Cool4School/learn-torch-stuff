import torch 
import torch.nn as nn



# 13 features

# 13 in 
# 100 out
# 20 in 
# out

class SalaryNet(nn.Module):
    def __init__(self, in_size, h1_size, h2_size, out_size):
        super(SalaryNet, self).__init__()
        self.h1 = nn.Linear(in_size, h1_size)
        self.relu = nn.ReLU()
        self.h2 = nn.Linear(h1_size, h2_size)
        self.out = nn.Linear(h2_size, out_size)

    def forward(self, x):
        h1_relu = self.relu(self.h1(x))
        h2_relu = self.relu(self.h2(h1_relu))
        predict = self.out(h2_relu)
        return predict
