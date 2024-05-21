import torch.nn as nn
import warnings
import pandas as pd    
import torch    
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", message="Initializing zero-element tensors is a no-op")



class LeNet5(nn.Module):
    def __init__(self,representation_dim: int, final_dim: int):
        super(LeNet5, self).__init__()

        self.representation_dim = representation_dim 
        self.final_dim = final_dim
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5)
        self.fc1 = nn.Linear(16 * 4 * 4, representation_dim)
        self.fc2 = nn.Linear(representation_dim, representation_dim)
        self.fc3 = nn.Linear(representation_dim, final_dim)

    def create_representation(self, x):

        if isinstance(x,pd.DataFrame):
            x = torch.tensor(x.values).float()
        x = x.view(x.shape[0], 1, 28, 28)
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 16 * 4 * 4)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))

        return x   

    def final_layer(self,x):

        x = torch.relu(self.fc3(x))

        return x
 
    def forward(self,x):

        x = self.create_representation(x)
        x = self.final_layer(x)
        return x


    def save(self,model_filepath):

        checkpoint = {
        'representation_dim': self.representation_dim,
        'final_dim' : self.final_dim,
        'state_dict': self.state_dict()
        }   
        torch.save(checkpoint,str(model_filepath))


    @classmethod
    def load(cls,model_filepath):

        checkpoint = torch.load(model_filepath)
        hyperparameters = {key: checkpoint[key] for key in set(checkpoint.keys()) - set(["state_dict"])}
        model = cls(**hyperparameters)
        model.load_state_dict(checkpoint["state_dict"])
        model.eval()

        return model









