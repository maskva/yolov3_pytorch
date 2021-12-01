from time import sleep

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch import nn, optim
from tqdm import tqdm

from util.Augmentation import SSDAugmentation
import VOCDataset
from model.yolo import Yolo

# with open('data/test.txt') as f:
#     val_lines = f.readlines()
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model=Yolo(n_classes=20,image_size=416,anchors=None,nms_thresh=0.45).to(device)
# dataset = VOCDataset.VOCDataset(val_lines, transform=SSDAugmentation([416,416]), train=True)
# dataloader = DataLoader(dataset=dataset,
#                         batch_size=4,
#                         shuffle=True,
#                         num_workers=0,
#                         collate_fn=VOCDataset.collate_fn
#                         )
# with torch.no_grad():
#     for e in range(0, 1):
#         print(len(dataset)//4)
#         for it,(images, targets) in enumerate(dataloader):
#             # print(images)
#             print(type(images),type(targets))
#
#             print(type(targets[0]))
#             print(targets[0])
#
#             print(images)
#             preds = model(images.to(device))
#             break




# p_mask=[[1,0,1],
#         [1,1,0],
#         [0,1,0]]
# cls=[[[1,11],[2,22],[3,33]],
#      [[4,44],[5,55],[6,66]],
#      [[7,77],[8,88],[9,99]]]
# p_mask=torch.tensor(p_mask,dtype=torch.int)
# cls=torch.tensor(cls,dtype=torch.int)
# m=p_mask==1
#
# print(cls[...,1])
# print(m)
# print(cls[m])


# for epoch in range(0,100):
#     with tqdm(total=10, desc=f'Epoch {epoch + 1}/100', postfix=dict,mininterval=0.3) as pbar:
#         for iteration in range(0, 10):
#             sleep(1)
#             pbar.set_postfix(**{'loss': 10 / (iteration + 1)})
#             pbar.update(1)


# darknet_params = model.darknet.parameters()
# other_params = [i for i in model.parameters()if i not in darknet_params]
# optimizer = optim.SGD(
#     [
#         {"params": darknet_params, 'initial_lr': 0.1, 'lr': 0.2},
#         {'params': other_params, 'initial_lr': 0.3, 'lr': 0.4}
#     ],
#     momentum=0.9,
#     weight_decay=0.001
# )
#
# print(optimizer.param_groups[0]['initial_lr'])
# print(optimizer.param_groups[1]['initial_lr'])

anchors=[
        [[116, 90], [156, 198], [373, 326]],
        [[30, 61], [62, 45], [59, 119]],
        [[10, 13], [16, 30], [33, 23]],
    ]

print(torch.tensor(anchors).shape)
print(torch.tensor(anchors).view(-1,2))


