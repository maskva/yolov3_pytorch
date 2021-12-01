import math

import numpy as np
import torch
from torch import nn


class YoloLoss(nn.Module):
    def __init__(self,anchors,anchor_masks,n_classes,overlap_thresh=0.45,lambda_obj=1, lambda_noobj=0.5, lambda_cls=1,lambda_box=2.5,img_w=416,img_h=416):
        super().__init__()
        self.n_classes=n_classes
        self.overlap_thresh=overlap_thresh
        #self.image_size=image_size
        self.anchors=anchors
        self.anchor_masks=anchor_masks
        self.mse_loss = nn.MSELoss(reduction='none')
        self.bce_loss = nn.BCELoss(reduction='none')
        self.lambda_obj=lambda_obj
        self.lambda_noobj=lambda_noobj
        self.lambda_cls=lambda_cls
        self.lambda_box=lambda_box
        self.img_w=img_w
        self.img_h=img_h
    def forward(self,preds,targets):
        """
        Parameters
        ----------
        preds: Tuple[Tensor]
            Yolo 神经网络输出的各个特征图，每个特征图的维度为 `(N, (n_classes+5)*n_anchors, H, W)`

        targets: List[Tensor]
            标签数据，每个标签张量的维度为 `(N, n_objects, 5)`，最后一维的第一个元素为类别，剩下为边界框 `(cx, cy, w, h)`

        Returns
        -------
        loc_loss: Tensor
            定位损失

        conf_loss: Tensor
            置信度损失

        cls_loss: Tensor
            分类损失
        """
        loc_loss = 0
        conf_loss = 0
        cls_loss = 0
        pos_num=0

        for l in range(len(preds)):
            N, _, featuremap_h, featuremap_w = preds[l].shape
            n_anchors = len(self.anchor_masks[l])

            # 调整特征图尺寸，方便索引
            pred = preds[l].view(N, n_anchors, self.n_classes + 5,featuremap_h, featuremap_w).permute(0, 1, 3, 4, 2).contiguous()

            # 获取特征图最后一个维度的每一部分
            x = pred[..., 0].sigmoid()
            y = pred[..., 1].sigmoid()
            w = pred[..., 2]
            h = pred[..., 3]
            conf = pred[..., 4].sigmoid()
            cls = pred[..., 5:].sigmoid()

            # 匹配边界框
            p_mask, n_mask, t, scale = match_box(self.anchors, targets,self.anchor_masks[l],featuremap_h, featuremap_w, self.n_classes,self.overlap_thresh,self.img_h,self.img_w )
            pos_num += torch.sum(p_mask)

            p_mask = p_mask.to(pred.device)
            n_mask = n_mask.to(pred.device)
            t = t.to(pred.device)
            scale = scale.to(pred.device)

            # 定位损失
            x_loss = torch.sum(self.bce_loss(x, t[..., 0]) * p_mask * scale)
            y_loss = torch.sum(self.bce_loss(y, t[..., 1]) * p_mask * scale)
            w_loss = torch.sum(self.mse_loss(w, t[..., 2]) * p_mask * scale)
            h_loss = torch.sum(self.mse_loss(h, t[..., 3]) * p_mask * scale)
            loc_loss += (x_loss + y_loss + w_loss + h_loss) * self.lambda_box

            # 置信度损失(其实这里也可以考虑把p_mask,n_mask提取出来)
            conf_loss += torch.sum(self.bce_loss(conf * p_mask, p_mask) * self.lambda_obj + \
                                   self.bce_loss(conf * n_mask, 0 * n_mask) * self.lambda_noobj)

            # 分类损失
            m = p_mask == 1  # 感觉这种过滤方法和前面的乘以一个p_mask(要先扩充一个维度)有什么区别，
            cls_loss += torch.sum(self.bce_loss(cls[m], t[..., 5:][m]) * self.lambda_cls)

        return loc_loss, conf_loss, cls_loss, pos_num




        # for anchors, pred in zip(self.all_anchors, preds):
        #     N, _, featuremap_h, featuremap_w = pred.shape
        #     n_anchors = len(anchors)
        #
        #     # 调整特征图尺寸，方便索引
        #     pred = pred.view(N, n_anchors, self.n_classes + 5,
        #                      featuremap_h, featuremap_w).permute(0, 1, 3, 4, 2).contiguous()
        #
        #     # 获取特征图最后一个维度的每一部分
        #     x = pred[..., 0].sigmoid()
        #     y = pred[..., 1].sigmoid()
        #     w = pred[..., 2]
        #     h = pred[..., 3]
        #     conf = pred[..., 4].sigmoid()
        #     cls = pred[..., 5:].sigmoid()
        #
        #     # 匹配边界框
        #     step_h = self.image_size / featuremap_h
        #     step_w = self.image_size / featuremap_w
        #     anchors = [[i / step_w, j / step_h] for i, j in anchors]
        #     #p_mask, n_mask, t, scale = match(anchors, targets, featuremap_h, featuremap_w, self.n_classes, self.overlap_thresh)
        #     p_mask, n_mask, t, scale=match_box(anchors,targets,featuremap_h,featuremap_w,self.n_classes,)
        #     pos_num+=torch.sum(p_mask)
        #
        #     p_mask = p_mask.to(pred.device)
        #     n_mask = n_mask.to(pred.device)
        #     t = t.to(pred.device)
        #     scale = scale.to(pred.device)
        #
        #     # 定位损失
        #     x_loss = torch.sum(self.bce_loss(x, t[..., 0]) * p_mask * scale)
        #     y_loss = torch.sum(self.bce_loss(y, t[..., 1]) * p_mask * scale)
        #     w_loss = torch.sum(self.mse_loss(w, t[..., 2]) * p_mask * scale)
        #     h_loss = torch.sum(self.mse_loss(h, t[..., 3]) * p_mask * scale)
        #     loc_loss += (x_loss + y_loss + w_loss + h_loss) * self.lambda_box
        #
        #     # 置信度损失(其实这里也可以考虑把p_mask,n_mask提取出来)
        #     conf_loss += torch.sum(self.bce_loss(conf * p_mask, p_mask) * self.lambda_obj + \
        #                  self.bce_loss(conf * n_mask, 0 * n_mask) * self.lambda_noobj)
        #
        #     # 分类损失
        #     m = p_mask == 1#感觉这种过滤方法和前面的乘以一个p_mask(要先扩充一个维度)有什么区别，
        #     cls_loss += torch.sum(self.bce_loss(cls[m], t[..., 5:][m]) * self.lambda_cls)
        #
        # return loc_loss, conf_loss, cls_loss,pos_num


def match_box(anchors,targets,anchor_mask,h,w,n_classes,overlap_thresh=0.5,img_w=416,img_h=416):
    N = len(targets)
    n_anchors = len(anchor_mask)
    # 初始化返回值
    p_mask = torch.zeros(N, n_anchors, h, w)
    n_mask = torch.ones(N, n_anchors, h, w)
    t = torch.zeros(N, n_anchors, h, w, n_classes+5)
    scale = torch.zeros(N, n_anchors, h, w)
    # 匹配先验框和边界框
    anchors = np.hstack((np.zeros((n_anchors, 2)), np.array(anchors)))
    for i in range(len(targets)):
        target = targets[i]  # shape:(n_objects, 5)
        # 迭代每一个 ground truth box
        for j in range(target.size(0)):
            # 获取标签数据
            gw = target[j, 3] * img_w
            gh = target[j, 4] * img_h

            # 计算边界框和先验框的交并比
            box = np.array([0, 0,gw, gh])
            iou = jaccard_overlap_numpy(box, anchors)
            index = np.argmax(iou)
            if index not in anchor_mask:
                continue
            index=anchor_mask.index(index)

            # 获取边界框中心所处的单元格的坐标
            cx = target[j, 1] * w
            cy = target[j, 2] * h
            gj, gi = int(cx), int(cy)

            # 标记出正例和反例
            p_mask[i, index, gi, gj] = 1
            # 正例除外，与 ground truth 的交并比都小于阈值则为负例
            n_mask[i, index, gi, gj] = 0
            # 大于阈值的则为非负例（正例或者忽视样本)
            iou=iou[anchor_mask[i]]
            n_mask[i, iou >= overlap_thresh, gi, gj] = 0
            # 说明：这里要设置p_mask,n_maske，是为了方便划分样本，只有如下三种情况
            # p_mask=1,m_mask=0则为正例，n_mask=1,p_mask=0则为负例，n_mask=p_mask=0为忽视样本

            # 计算标签值
            t[i, index, gi, gj, 0] = cx-gj
            t[i, index, gi, gj, 1] = cy-gi
            t[i, index, gi, gj, 2] = math.log(gw/anchors[index, 2]+1e-16)
            t[i, index, gi, gj, 3] = math.log(gh/anchors[index, 3]+1e-16)
            t[i, index, gi, gj, 4] = 1
            t[i, index, gi, gj, 5+int(target[j, 0])] = 1

            # 缩放值，用于惩罚小方框的定位
            scale[i, index, gi, gj] = 2 - target[j, 3] * target[j, 4]

        return p_mask, n_mask, t, scale


# def match(anchors: list, targets, h, w, n_classes, overlap_thresh=0.5):
#     """ 匹配先验框和边界框真值
#     Parameters
#     ----------
#     anchors: list of shape `(n_anchors, 2)`
#         根据特征图的大小进行过缩放的先验框
#     targets: List[Tensor]
#         标签，每个元素的最后一个维度的第一个元素为类别，剩下四个为 `(cx, cy, w, h)`
#     h: int
#         特征图的高度
#     w: int
#         特征图的宽度
#     n_classes: int
#         类别数
#     overlap_thresh: float
#         IOU 阈值
#     Returns
#     -------
#     p_mask: Tensor of shape `(N, n_anchors, H, W)`
#         正例遮罩
#     n_mask: Tensor of shape `(N, n_anchors, H, W)`
#         反例遮罩
#     t: Tensor of shape `(N, n_anchors, H, W, n_classes+5)`
#         标签
#     scale: Tensor of shape `(N, n_anchors, h, w)`
#         缩放值，用于惩罚小方框的定位
#     """
#     N = len(targets)
#     n_anchors = len(anchors)
#
#     # 初始化返回值
#     p_mask = torch.zeros(N, n_anchors, h, w)
#     n_mask = torch.ones(N, n_anchors, h, w)
#     t = torch.zeros(N, n_anchors, h, w, n_classes+5)
#     scale = torch.zeros(N, n_anchors, h, w)
#
#     # 匹配先验框和边界框
#     anchors = np.hstack((np.zeros((n_anchors, 2)), np.array(anchors)))
#     for i in range(N):
#         target = targets[i]  # shape:(n_objects, 5)
#
#         # 迭代每一个 ground truth box
#         for j in range(target.size(0)):
#             # 获取标签数据
#             cx, gw = target[j, [1, 3]]*w
#             cy, gh = target[j, [2, 4]]*h
#
#             # 获取边界框中心所处的单元格的坐标
#             gj, gi = int(cx), int(cy)
#
#             # 计算边界框和先验框的交并比
#             bbox = np.array([0, 0, gw, gh])
#             iou = jaccard_overlap_numpy(bbox, anchors)
#
#             # 标记出正例和反例
#             index = np.argmax(iou)
#             p_mask[i, index, gi, gj] = 1
#             # 正例除外，与 ground truth 的交并比都小于阈值则为负例
#             n_mask[i, index, gi, gj] = 0
#             #大于阈值的则为非负例（正例或者忽视样本)
#             n_mask[i, iou >= overlap_thresh, gi, gj] = 0
#             #说明：这里要设置p_mask,n_maske，是为了方便划分样本，只有如下三种情况
#             #p_mask=1,m_mask=0则为正例，n_mask=1,p_mask=0则为负例，n_mask=p_mask=0为忽视样本
#
#             # 计算标签值
#             t[i, index, gi, gj, 0] = cx-gj
#             t[i, index, gi, gj, 1] = cy-gi
#             t[i, index, gi, gj, 2] = math.log(gw/anchors[index, 2]+1e-16)
#             t[i, index, gi, gj, 3] = math.log(gh/anchors[index, 3]+1e-16)
#             t[i, index, gi, gj, 4] = 1
#             t[i, index, gi, gj, 5+int(target[j, 0])] = 1
#
#             # 缩放值，用于惩罚小方框的定位
#             scale[i, index, gi, gj] = 2-target[j, 3]*target[j, 4]
#
#     return p_mask, n_mask, t, scale

def jaccard_overlap_numpy(box: np.ndarray, boxes: np.ndarray):
    """ 计算一个边界框和多个边界框的交并比

    Parameters
    ----------
    box: `~np.ndarray` of shape `(4, )`
        边界框

    boxes: `~np.ndarray` of shape `(n, 4)`
        其他边界框

    Returns
    -------
    iou: `~np.ndarray` of shape `(n, )`
        交并比
    """
    # 计算交集
    xy_max = np.minimum(boxes[:, 2:], box[2:])#相交框右下角坐标
    xy_min = np.maximum(boxes[:, :2], box[:2])#相交框左上角坐标
    inter = np.clip(xy_max-xy_min, a_min=0, a_max=np.inf)#计算相交框的宽高，不相交时，xy_max-xy_min会出现负数
    inter = inter[:, 0]*inter[:, 1]

    # 计算并集
    area_boxes = (boxes[:, 2]-boxes[:, 0])*(boxes[:, 3]-boxes[:, 1])
    area_box = (box[2]-box[0])*(box[3]-box[1])

    # 计算 iou
    iou = inter/(area_box+area_boxes-inter)  # type: np.ndarray
    return iou