import torch.nn as nn
import torch   
from hdbscan.hdbscan_ import HDBSCAN
import torch.nn.functional as F


def off_diagonal(x):
        # return a flattened view of the off-diagonal elements of a square matrix
        n, m = x.shape
        assert n == m
        return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()



class VICRegSingle(nn.Module):
    def __init__(self):
        super(VICRegSingle, self).__init__()

    def forward(self, x):

        std_x = torch.sqrt(x.var(dim=0) + 0.0001)
        x = (x - x.mean(dim=0))/std_x
        cov_x = (x.T @ x) / (x.shape[0] - 1)
        cov_loss = off_diagonal(cov_x).pow_(2).sum().div(x.shape[1])

        return cov_loss


class VICRegCluster(nn.Module):
    def __init__(self):
        super(VICRegCluster, self).__init__()  
        self.vicreg = VICRegSingle()

    def forward(self,x):

        if x.shape[0]>1 and not torch.isnan(x).any():
            labels_learned = HDBSCAN().fit_predict(x.cpu().detach().numpy())
            vicreg_loss = torch.tensor(0.).to(x.device)
            for label in set(labels_learned):
                this_cluster = x[labels_learned==label,:]
                if this_cluster.shape[0]>1:
                    vicreg_loss+=self.vicreg(this_cluster)
            return vicreg_loss
        else:
            return torch.tensor(0.).to(x.device)


class VICReg(nn.Module):
    """hat tip to JP"""

    def __init__(self,sim_coeff=25.,std_coeff=25.,cov_coeff=1.):
        super(VICReg, self).__init__()

        self.sim_coeff = sim_coeff
        self.std_coeff = std_coeff
        self.cov_coeff = cov_coeff

    def forward(self, x, y):

        repr_loss = F.mse_loss(x, y)

        x = x - x.mean(dim=0)
        y = y - y.mean(dim=0)

        std_x = torch.sqrt(x.var(dim=0) + 0.0001)
        std_y = torch.sqrt(y.var(dim=0) + 0.0001)
        std_loss = torch.mean(F.relu(1 - std_x)) / 2 + torch.mean(F.relu(1 - std_y)) / 2

        cov_x = (x.T @ x) / (x.shape[0] - 1)
        cov_y = (y.T @ y) / (y.shape[0] - 1)
        cov_loss = off_diagonal(cov_x).pow_(2).sum().div(
            x.shape[1]
        ) + off_diagonal(cov_y).pow_(2).sum().div(y.shape[1])

        loss = (
            self.sim_coeff * repr_loss
            + self.std_coeff * std_loss
            + self.cov_coeff * cov_loss
        )
        return loss







