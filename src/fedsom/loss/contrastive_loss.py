import torch
import torch.nn.functional as F
from scipy.spatial.distance import cdist
import time 
import numpy as np    



class ContrastiveAttention(object):

    def __init__(self,categorical,metric,device=None):

        self.categorical = categorical
        self.metric = metric 
        self.device = device if device else 'cpu'


    def compute_dist_attention(self,batch,batch_aug):

        # used to kill all entries in the contrastive objective denominator that are close
        # we don't want positive samples in the contrastive denominator
        # returns a matrix such that each value is false if that value is below the 
        # 10th percentile for the entire batch

        distances = torch.tensor(self.metric(batch.values,batch_aug.values)).to(self.device)
        return (distances>torch.sort(distances)[0][int(0.1*len(distances))])


    def __call__(self,batch,batch_aug,batch_raw,batch_raw_aug,temp):

        batch_raw,batch_raw_aug = self.categorical(batch_raw),self.categorical(batch_raw_aug)
        dist_attention = self.compute_dist_attention(batch_raw,batch_raw_aug)

        # print(dist_attention)
        # input('DONE')
  
        batch = F.normalize(batch, dim=-1)
        batch_aug = F.normalize(batch_aug, dim=-1)

        similarity = torch.exp(torch.div(torch.matmul(batch, batch_aug.T),temp))
        numerator = torch.diag(similarity) 
        diag_mask = torch.eye(similarity.size(0),dtype=torch.bool).to(self.device)

        denominators = torch.sum(similarity*dist_attention,dim=1) - torch.sum(similarity*dist_attention*diag_mask,dim=1)

        return -torch.sum(torch.log(numerator/denominators))/batch.shape[0]



def contrastive_loss(batch, batch_aug, temp=0.5):
    """Ref: https://arxiv.org/pdf/2106.01342.pdf
    Numerically stable version in which logs are applied as soon as possible"""

    # normalize for cosine sim
    batch = F.normalize(batch, dim=-1)
    batch_aug = F.normalize(batch_aug, dim=-1)

    # compute similarity matrix consisting of all combos of dot products
    similarity = torch.exp(torch.div(torch.matmul(batch, batch_aug.T), temp))

    numerator = torch.diag(similarity) 
    diag_mask = torch.eye(similarity.size(0),dtype=torch.bool).to(batch.device)
    denominators = torch.sum(similarity,dim=1) - torch.sum(similarity*diag_mask,dim=1)

    return -torch.sum(torch.log(numerator/denominators))/batch.shape[0]





if __name__ == "__main__":
    torch.manual_seed(0)

    batch = torch.randn(14, 10)
    batch_aug = torch.randn(14, 10)

    temp = 2.0
    loss = contrastive_loss(batch, batch_aug, temp=temp)
    print(loss.item())
