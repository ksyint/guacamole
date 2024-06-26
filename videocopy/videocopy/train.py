#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import torch
from torch.utils.data import Dataset, DataLoader
from exps.exp import Exp
from transvcl.utils import postprocess
import numpy as np
import os
import json
import pandas as pd
from collections import defaultdict
import argparse
from loguru import logger
import torch.nn as nn

criterion=nn.MSELoss()
criterion=criterion.cuda()

def feat_paddding(feat: torch.Tensor, axis: int, new_size: int, fill_value: int = 0):
    pad_shape = list(feat.shape)
    pad_shape[axis] = max(0, new_size - pad_shape[axis])
    feat_pad = torch.Tensor(*pad_shape).fill_(fill_value)
    return torch.cat([feat, feat_pad], dim=axis)

def load_features_list(feat1, feat2, file_name):
    feat_length = 1200
    feat1_list, feat2_list = [], []
    i, j = -1, -1
    for i in range(len(feat1) // feat_length):
        feat1_list.append(feat1[i * feat_length: (i + 1) * feat_length])
    for j in range(len(feat2) // feat_length):
        feat2_list.append(feat2[j * feat_length: (j + 1) * feat_length])
    if len(feat1) > (i + 1) * feat_length:
        feat1_list.append(feat1[(i + 1) * feat_length:])
    if len(feat2) > (j + 1) * feat_length:
        feat2_list.append(feat2[(j + 1) * feat_length:])
    batch_list = []
    for i in range(len(feat1_list)):
        for j in range(len(feat2_list)):
            mask1, mask2 = np.zeros(feat_length, dtype=bool), np.zeros(feat_length, dtype=bool)
            mask1[:len(feat1_list[i])] = True
            mask2[:len(feat2_list[j])] = True

            feat1_padding = feat_paddding(torch.tensor(feat1_list[i]), 0, feat_length)
            feat2_padding = feat_paddding(torch.tensor(feat2_list[j]), 0, feat_length)

            img_info = [torch.tensor([len(feat1_list[i])]), torch.tensor([len(feat2_list[j])])]

            file_name_idx = file_name + "_" + str(i) + "_" + str(j)

            batch_list.append((feat1_padding, feat2_padding, torch.from_numpy(mask1), torch.from_numpy(mask2), img_info, file_name_idx))

    return batch_list

class SimFeatDataset(Dataset):
    def __init__(self, batch_list, **kwargs):
        self.batch_list = batch_list

    def __getitem__(self, item):
        return self.batch_list[item]

    def __len__(self):
        return len(self.batch_list)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--model-file", type=str, default=None, help="TransVCL model file")
    parser.add_argument("--feat-dir", type=str, default=None, help="video feature dir")
    parser.add_argument("--feat-length", type=int, default=1200, help="feature length for TransVCL input")
    parser.add_argument("--test-file", type=str, default=None, help="test pair list of query and reference videos")
    parser.add_argument("--conf-thre", type=float, default=0.1, help="conf threshold of copied segments")
    parser.add_argument("--nms-thre", type=float, default=0.3, help="nms threshold of copied segments")
    parser.add_argument("--img-size", type=int, default=640, help="length for copied localization module")
    parser.add_argument("--load-batch", type=int, default=8192, help="batch size of loading features to CPU")
    parser.add_argument("--inference-batch", type=int, default=1024, help="batch size of TransVCL inference")
    parser.add_argument("--save-file", type=str, default=None, help="save json file of results")
    parser.add_argument("--device", type=int, default=None, help="GPU device")
    import gc 
    gc.collect()
    torch.cuda.empty_cache()
    args = parser.parse_args()
    if args.device is not None:
        device = "cuda:" + str(args.device)
    else:
        os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
        os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    feat_dir, test_file = args.feat_dir, args.test_file
    num_classes, confthre, nmsthre = 1, args.conf_thre, args.nms_thre
    img_size, feat_max_length = (args.img_size, args.img_size), args.feat_length
    with open("data/VCSL/label_file.json") as label:
            label2=json.load(label)
                
    label3=list(label2)

    df = pd.read_csv(test_file)
    process_list = [f"{q}-{r}" for q, r in zip(df.query_id.values, df.reference_id.values)]
    process_list = [file.split(".")[0] for file in process_list]

    result = defaultdict(list)

    exp = Exp()
    model = exp.get_model()

    
    ckpt = torch.load(args.model_file, map_location="cpu")
    model.load_state_dict(ckpt["model"])
    model=model.cuda()
    
    #print(len(process_list))

    batch_feat_list = []
    for idx, process_img in enumerate(process_list):
        #print(idx)
        #batch_feat_list = []
        feat1_name, feat2_name = process_img.split("-")[0], process_img.split("-")[1]
        feat1_name, feat2_name = feat_dir + feat1_name + ".npy", feat_dir + feat2_name + ".npy"
        feat1, feat2 = np.load(feat1_name), np.load(feat2_name)
        batch_feat_list+=load_features_list(feat1, feat2, process_img)
        #print(batch_feat_list)
    
    #print(len(batch_feat_list))
    
    dataset = SimFeatDataset(batch_feat_list)
    dataloader_kwargs = {"batch_size": 1, "num_workers": 0}
    loader = DataLoader(dataset, **dataloader_kwargs)
    model.train()
    optimizer=torch.optim.Adam(model.parameters(),lr=0.1)

    for epoch in range(10):
        print(f"Epoch : {epoch+1}")
        for idx, batch_data in enumerate(loader):
        
            feat1, feat2, mask1, mask2, img_info, file_name = batch_data
            feat1, feat2, mask1, mask2 = feat1.cuda(), feat2.cuda(), mask1.cuda(), mask2.cuda()
            model_outputs = model(feat1, feat2, mask1, mask2, file_name, img_info)
            outputs = postprocess(
                model_outputs[1], num_classes, confthre,
                nmsthre, class_agnostic=True
            )
         
            
            bboxes = outputs[0][:, :5].cpu()
            scale1, scale2 = img_info[0].item() / 640, img_info[1].item() / 640
            bboxes[:, 0:4:2] *= scale2
            bboxes[:, 1:4:2] *= scale1
            prediction=bboxes[:, (1, 0, 3, 2, 4)].tolist()
            prediction=torch.tensor(prediction[0][:-1],requires_grad=True)
            prediction=prediction.cuda()
            
            
            gt=torch.tensor(label2[label3[idx]][0][0:]).cuda()
            
            loss=criterion(prediction.to(torch.float32),gt.to(torch.float32))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print(loss)
            
            
            
                    
                        
                            
                       

        






