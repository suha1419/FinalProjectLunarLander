import torch
from torch import nn
import torch.nn.functional as F



class DQN(nn.Module):

    def __init__(self, state_dim, action_dim, hidden_dim=[256], seed=42):
        super(DQN, self).__init__()

        torch.manual_seed(seed)

        # Create layer of each dimensions
        layers = []
        input_dim = state_dim

        for hidden in hidden_dim:
            layers.append(nn.Linear(input_dim,hidden))
            layers.append(nn.ReLU())
            input_dim = hidden

        layers.append(nn.Linear(input_dim, action_dim))
        self.model = nn.Sequential(*layers)
    
    def forward(self,x):
        return self.model(x)
