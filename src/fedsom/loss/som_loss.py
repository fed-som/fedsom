import torch     
import torch.nn.functional as F
from torch.nn import CosineEmbeddingLoss as CEL




class SOMLoss(object):

    def __init__(self,som,cosine=True):

        self.som = som   
        self.cosine = cosine  
        self.cel = cel = CEL() 

    def __call__(self,v,v_pert):

        v_clusters = self.som.get_labels(v)
        v_pert_clusters = self.som.get_labels(v_pert)
        idx = torch.tensor(v_clusters==v_pert_clusters,dtype=torch.bool)

        if self.cosine:
            idx[idx==0] = -1
            return self.cel(v,v_pert,idx) 
        else:
            if idx.sum()==len(idx):
                return torch.tensor(0.0)
            return torch.mean((v[~idx,:]-v_pert[~idx,:])**2)


def som_loss(v,v_pert,cosine=True):

    cel = CEL()
    if cosine==True:
        return cel(F.normalize(v),F.normalize(v_pert),torch.ones(v.shape[0]))
    elif cosine==False:
        return cel(v,v_pert,torch.ones(v.shape[0]))
    else:
        return torch.mean((v-v_pert)**2)

