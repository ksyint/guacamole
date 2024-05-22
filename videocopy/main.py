
import numpy as np
from isc_feature_extractor import create_model
import pandas as pd  
import numpy as np 
import torch
from torch.utils.data import DataLoader
from cosine import cosine_loss
from isc2.isc_feature_extractor.model import ISCNet
import timm
from PIL import Image
from torchvision import transforms
backbone = timm.create_model("timm/tf_efficientnetv2_m.in21k_ft_in1k", features_only=True)
model=ISCNet(backbone)

model=torch.nn.DataParallel(model)
model=model.cuda()
T1=transforms.ToTensor()

class pair_dataset(torch.utils.data.Dataset):

    def __init__(self, file):
        
        self.file=file
        

    def __len__(self):
        
        csv=pd.read_csv(self.file)
        
        return len(csv.iloc[:,0])

    def __getitem__(self, idx):
        
        csv2=pd.read_csv(self.file)
        query=csv2.iloc[idx,0]
        refer=csv2.iloc[idx,1]
        query=np.load(f"/home/ksyint1111/isc/eff256d/{query}.npy")
        refer=np.load(f"/home/ksyint1111/isc/eff256d/{refer}.npy")
        query=Image.fromarray(query)
        refer=Image.fromarray(refer)
        query=query.resize((256,256),Image.BICUBIC)
        refer=refer.resize((256,256),Image.BICUBIC)
        query=T1(query)
        refer=T1(refer)
        query = torch.cat((query,query,query), dim=0)
        refer = torch.cat((refer,refer,refer), dim=0)

        
        return query,refer
    
train_dataset=pair_dataset("/home/ksyint1111/isc/VCSL/data/pair_file_train.csv")
val_dataset=pair_dataset("/home/ksyint1111/isc/VCSL/data/pair_file_val.csv")


train_dataloader = DataLoader(train_dataset,batch_size =1,shuffle = True)
#val_dataloader = DataLoader(val_dataset,batch_size =1,shuffle = True)


optimizer=torch.optim.Adam(model.parameters(),lr=0.001)



model.train()
for epoch in range(1000):
    
    for query,refer in train_dataloader:
        try:
            query=query.cuda()
            refer=refer.cuda()
        except:
            continue
    
        query_feature=model(query)
        refer_feature=model(refer)
        loss=cosine_loss(query_feature,refer_feature)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(loss.item())
        torch.save(model.state_dict(),"current_model_isc.pth")
        

        
    
    
        
    
    




        
    
        
        
    
    
    


