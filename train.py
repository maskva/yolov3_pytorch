import argparse

import numpy as np
import torch.types
from torch import optim
from torch.utils.data import DataLoader

from VOCDataset import VOCDataset ,collate_fn
from util.Augmentation import SSDAugmentation
from model.yolo import Yolo
from model.loss import YoloLoss
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser(description='YOLO Detection')
    parser.add_argument('--start', default=0,type=int,help='start epoch')
    #parser.add_argument('--weight',default='data/yolo_weights.pth',help='weight file')
    parser.add_argument('--yolo_path', help='yolo_path')
    parser.add_argument('--darknet_path',default='data/darknet_weights.pth', help='darknet_path')
    parser.add_argument('--momentum', help='momentum',type=float,default=0.9)
    parser.add_argument('--weight_decay', help='weight_decay',type=float,default=4e-5)
    parser.add_argument('--lr', help='lr', type=float, default=0.01)
    parser.add_argument('--backbone_lr', help='backbone_lr', type=float, default=1e-3)
    parser.add_argument('--lr_step_size', help='lr_step_size', type=int, default=15)
    parser.add_argument('--batch_size', help='batch_size', type=int, default=4)
    parser.add_argument('--start_epoch', help='start_epoch', type=int, default=0)
    parser.add_argument('--max_epoch', help='max_epoch', type=int, default=60)
    parser.add_argument('--save_frequency', help='save_frequency', type=int, default=1)
    parser.add_argument('--save_dir', help='save_dir', default='data')
    parser.add_argument('--log_file', help='log_file',  default='log')


    return parser.parse_args()

def main():
    args = parse_args()
    device=torch.device('cuda'if torch.cuda.is_available()else 'cpu')
    anchors=[
        [116, 90], [156, 198], [373, 326],
        [30, 61], [62, 45], [59, 119],
        [10, 13], [16, 30], [33, 23],
    ]
    anchor_masks=[[0,1,2],[3,4,5],[6,7,8]]
    num_classes=20

    #读取数据，创建dataset和dataloader
    with open("data/2007_train.txt") as f:
        train_lines = f.readlines()
    with open("data/2012_train.txt") as f:
        train_lines.extend(f.readlines())
    with open('data/2007_val.txt') as f:
        val_lines = f.readlines()
    with open('data/2012_val.txt') as f:
        val_lines.extend(f.readlines())
    num_train   = len(train_lines)
    num_val     = len(val_lines)
    dataset=VOCDataset(train_lines,transform=SSDAugmentation(),train=True)
    dataloader=DataLoader(dataset=dataset,
                          batch_size=args.batch_size,
                          shuffle=True,
                          num_workers=6,
                          collate_fn=collate_fn
                          )

    #创建模型，载入权重
    model=Yolo(n_classes=20,image_size=416,anchors=None,nms_thresh=0.45).to(device)
    if args.yolo_path:
        model.load_state_dict(torch.load(args.yolo_path,map_location=device))
        print('🧪 成功载入 Yolo 模型：' + args.yolo_path)
    elif args.darknet_path:
        model.darknet.load_state_dict(torch.load(args.darknet_path,map_location=device))
        print('🧪 成功载入 Darknet53 模型：' + args.darknet_path)
    else:
        raise ValueError("必须指定预训练的 Darknet53 模型文件路径")

    # 损失函数
    yololoss=YoloLoss(anchors,anchor_masks,num_classes,overlap_thresh=0.5)

    # 优化器
    darknet_params = model.darknet.parameters()
    other_params = [i for i in model.parameters()
                    if i not in darknet_params]
    optimizer = optim.SGD(
        [
            {"params": darknet_params, 'initial_lr': args.backbone_lr, 'lr': args.backbone_lr},
            {'params': other_params, 'initial_lr': args.lr, 'lr': args.lr}
        ],
        momentum=args.momentum,
        weight_decay=args.weight_decay
    )

    lr_schedule = optim.lr_scheduler.StepLR(optimizer, args.lr_step_size, 0.1, last_epoch=args.start_epoch)


    print('🚀 开始训练！')
    model.train()
    for epoch in range(args.start_epoch,args.max_epoch):
        with tqdm(total=num_train / args.batch_size, desc=f'Epoch {epoch + 1}/{args.max_epoch}', postfix=dict,
                  mininterval=0.3) as pbar:
            loss=0
            for iteration,(images, targets) in enumerate(dataloader):
                with torch.no_grad():
                    optimizer.zero_grad()
                    preds=model(images.to(device))

                    loc_loss, conf_loss, cls_loss ,pos_num= yololoss(preds, targets)
                    cur_loss = (loc_loss + conf_loss + cls_loss)/pos_num
                    loss += cur_loss

                    cur_loss.backward()
                    optimizer.step()

                    # 学习率退火
                    lr_schedule.step()

                    # 定期保存模型
                    if epoch > args.start_epoch and (epoch + 1 - args.start_epoch) % args.save_frequency == 0:
                        model.eval()
                        path = args.save_dir / f'Yolov3_{epoch + 1}.pth'
                        torch.save(model.state_dict(), path)


                    pbar.set_postfix(**{'loss': loss / (iteration + 1),
                                        'darknet_params_lr': optimizer.param_groups[0]['lr'],
                                        'other_params':optimizer.param_groups[1]['lr']})
                    pbar.update(1)

if __name__=='__main__':
    main()