import torch.nn as nn
import torch 
import torch.nn.functional as F




class C3Loss(nn.Module):
    def __init__(self,zeta,device=None):
        super().__init__()

        self.zeta = zeta
        self.device = device if device else 'cpu'


    def __call__(self,z_i, z_j):

        batch_size = z_i.shape[0]
        z = torch.cat((z_i, z_j), dim=0)
        z = F.normalize(z)
        multiply = torch.matmul(z, z.T)

        a = torch.ones([batch_size])
        mask = 2 * (torch.diag(a, -batch_size) + torch.diag(a, batch_size) + torch.eye(2 * batch_size))
        mask.to(self.device)

        exp_mul = torch.exp(multiply)
        numerator = torch.sum(torch.where((multiply + mask) > self.zeta, exp_mul, torch.zeros(multiply.shape).to(self.device)), dim=1)
        den = torch.sum(exp_mul, dim=1)

        return -torch.sum(torch.log(torch.div(numerator, den))) / batch_size