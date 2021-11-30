import cv2 as cv
import numpy as np
import torch
from torch.utils.data.dataset import Dataset
class VOCDataset(Dataset):
    def __init__(self,annotation_lines,
                 transform=None,
                 train=True):
        self.annotations_lines=annotation_lines
        self.transform = transform
        self.train=train
    def __getitem__(self, index):
        """ 获取样本
        Parameters
        ----------
        index: int
            下标
        Returns
        -------
        image: Tensor of shape `(3, H, W)`
            增强后的图像数据
        target: `np.ndarray` of shape `(n_objects, 5)`
            标签数据，第一列为类别标签，剩下四列为边界框坐标 `(cx, cy, w, h)`
        """

        # 读入图片和标签数据
        line=self.annotations_lines[index].split()
        image = cv.cvtColor(cv.imread(line[0]), cv.COLOR_BGR2RGB)
        target = np.array([np.array(list(map(int, box.split(',')))) for box in line[1:]])
        boxes, labels = target[:, :4], target[:, 4]
        boxes=Box_Transform(boxes,image.shape[0],image.shape[1])

        if self.transform is not None:
            image, boxs, labels = self.transform(image, boxes,labels )
            target = np.hstack((labels[:, np.newaxis], boxes))

        #H,W,Cto C,H,W
        image=torch.from_numpy(image).permute(2,0,1)

        return image,target

    def __len__(self):
        return len(self.annotations_lines)

def Box_Transform(boxes,h,w):
    boxes = boxes.astype(np.float)
    boxes[:,2:] -=boxes[:,:2]
    boxes[:,:2]+=boxes[:,2:]/2

    boxes[:,[0,2]]/=w
    boxes[:,[1,3]]/=h

    return  boxes




def collate_fn(batch):
    """ 整理 dataloader 取出的数据
        Parameters
        ----------
        batch: list of shape `(N, 2)`
            一批数据，列表中的每一个元组包括两个元素：
            * image: Tensor of shape `(3, H, W)`
            * target: `~np.ndarray` of shape `(n_objects, 5)`
        Returns
        -------
        image: Tensor of shape `(N, 3, H, W)`
            图像
        target: List[Tensor]
            标签
        """
    images = []
    targets = []
    for img, target in batch:
        images.append(img.to(torch.float32))
        targets.append(torch.Tensor(target))
    return torch.stack(images, 0), targets


