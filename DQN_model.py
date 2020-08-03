import torch
import torch.nn as nn
import torch.nn.functional as F

class DQNetwork(nn.Module):

    def __init__(self, state_size, action_size, seed):

        super(DQNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)

        '''
        Five Fully Connected Layers.
        '''
        fc1_units = 1024
        fc2_units = 512
        fc3_units = 256
        fc4_units = 128

        self.fc1 = nn.Linear(state_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.fc3 = nn.Linear(fc2_units, fc3_units)
        self.fc4 = nn.Linear(fc3_units, fc4_units)
        self.fc5 = nn.Linear(fc4_units, action_size)

    def forward(self, state):
        """Build a network that maps state -> action values."""

        x = F.relu(self.fc1(state))
        
        x = F.relu(self.fc2(x))
        
        x = F.relu(self.fc3(x))
        
        x = F.relu(self.fc4(x))
        
        x = self.fc5(x)

        return x
