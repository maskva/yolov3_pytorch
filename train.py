import argparse

from torch.utils.data import DataLoader

from VOCDataset import VOCDataset
from model import yolo
from model.yolo import Yolo


def parse_args():
    parser = argparse.ArgumentParser(description='YOLO Detection')
    parser.add_argument('--start', default=0,type=int,help='start epoch')
    parser.add_argument('--weight',default='data/yolo_weights.pth',help='weight file')
    parser.add_argument('--yolo_path', help='yolo_path')
    parser.add_argument('--darknet_path', help='darknet_path')
    return parser.parse_args()

def main():
    args = parse_args()

    with open("2007_train.txt") as f:
        train_lines = f.readlines()
    with open("2012_train.txt") as f:
        train_lines.extend(f.readlines())
    with open('2007_val.txt') as f:
        val_lines   = f.readlines()
    with open('2012_val.txt') as f:
        val_lines.extend(f.readlines())

    num_train   = len(train_lines)
    num_val     = len(val_lines)
    dataset=VOCDataset(train_lines,transform=None,train=True)
    dataloader=DataLoader(dataset=dataset,
                          batch_size=16,
                          shuffle=True,
                          num_workers=8)
    model=Yolo(n_classes=20,image_size=416,anchors=None,nms_thresh=0.45)
    if args.yolo_path:
        model.load(yolo_path)
        print('🧪 成功载入 Yolo 模型：' + yolo_path)
    elif args.darknet_path:
        model.darknet.load(darknet_path)
        print('🧪 成功载入 Darknet53 模型：' + darknet_path)
    else:
        raise ValueError("必须指定预训练的 Darknet53 模型文件路径")

