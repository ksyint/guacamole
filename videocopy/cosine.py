import torch
import torch.nn.functional as F


def cosine_loss(query,refer):
   
    
    cos_sim = F.cosine_similarity(query, refer,dim=1)
    return 1-cos_sim
    




