# YOLOv5 🚀 by Ultralytics, AGPL-3.0 license
"""
Loss functions
"""

import torch
import torch.nn as nn

from utils.metrics import bbox_iou
from utils.torch_utils import de_parallel


def smooth_BCE(eps=0.1):  # https://github.com/ultralytics/yolov3/issues/238#issuecomment-598028441
    # return positive, negative label smoothing BCE targets
    return 1.0 - 0.5 * eps, 0.5 * eps


class BCEBlurWithLogitsLoss(nn.Module):
    # BCEwithLogitLoss() with reduced missing label effects.
    def __init__(self, alpha=0.05):
        super().__init__()
        self.loss_fcn = nn.BCEWithLogitsLoss(reduction='none')  # must be nn.BCEWithLogitsLoss()
        self.alpha = alpha

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)
        pred = torch.sigmoid(pred)  # prob from logits
        dx = pred - true  # reduce only missing label effects
        # dx = (pred - true).abs()  # reduce missing label and false label effects
        alpha_factor = 1 - torch.exp((dx - 1) / (self.alpha + 1e-4))
        loss *= alpha_factor
        return loss.mean()


class FocalLoss(nn.Module):
    # Wraps focal loss around existing loss_fcn(), i.e. criteria = FocalLoss(nn.BCEWithLogitsLoss(), gamma=1.5)
    def __init__(self, loss_fcn, gamma=1.5, alpha=0.25):
        super().__init__()
        self.loss_fcn = loss_fcn  # must be nn.BCEWithLogitsLoss()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = loss_fcn.reduction
        self.loss_fcn.reduction = 'none'  # required to apply FL to each element

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)
        # p_t = torch.exp(-loss)
        # loss *= self.alpha * (1.000001 - p_t) ** self.gamma  # non-zero power for gradient stability

        # TF implementation https://github.com/tensorflow/addons/blob/v0.7.1/tensorflow_addons/losses/focal_loss.py
        pred_prob = torch.sigmoid(pred)  # prob from logits
        p_t = true * pred_prob + (1 - true) * (1 - pred_prob)
        alpha_factor = true * self.alpha + (1 - true) * (1 - self.alpha)
        modulating_factor = (1.0 - p_t) ** self.gamma
        loss *= alpha_factor * modulating_factor

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss


class QFocalLoss(nn.Module):
    # Wraps Quality focal loss around existing loss_fcn(), i.e. criteria = FocalLoss(nn.BCEWithLogitsLoss(), gamma=1.5)
    def __init__(self, loss_fcn, gamma=1.5, alpha=0.25):
        super().__init__()
        self.loss_fcn = loss_fcn  # must be nn.BCEWithLogitsLoss()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = loss_fcn.reduction
        self.loss_fcn.reduction = 'none'  # required to apply FL to each element

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)

        pred_prob = torch.sigmoid(pred)  # prob from logits
        alpha_factor = true * self.alpha + (1 - true) * (1 - self.alpha)
        modulating_factor = torch.abs(true - pred_prob) ** self.gamma
        loss *= alpha_factor * modulating_factor

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss


class ComputeLoss:
    sort_obj_iou = False

    # Compute losses
    def __init__(self, model, autobalance=False):
        """

        Args:
            model:
            autobalance:
        """
        device = next(model.parameters()).device  # get model device
        h = model.hyp  # hyperparameters

        # Define criteria
        BCEcls = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h['cls_pw']], device=device))
        BCEobj = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h['obj_pw']], device=device))

        # Class label smoothing https://arxiv.org/pdf/1902.04103.pdf eqn 3
        self.cp, self.cn = smooth_BCE(eps=h.get('label_smoothing', 0.0))  # positive, negative BCE targets

        # Focal loss
        g = h['fl_gamma']  # focal loss gamma
        if g > 0:
            BCEcls, BCEobj = FocalLoss(BCEcls, g), FocalLoss(BCEobj, g)

        m = de_parallel(model).model[-1]  # Detect() module, type(m) = <class 'models.yolo.Detect'>
        self.balance = {3: [4.0, 1.0, 0.4]}.get(m.nl, [4.0, 1.0, 0.25, 0.06, 0.02])  # P3-P7
        self.ssi = list(m.stride).index(16) if autobalance else 0  # stride 16 index
        self.BCEcls, self.BCEobj, self.gr, self.hyp, self.autobalance = BCEcls, BCEobj, 1.0, h, autobalance
        self.na = m.na  # number of anchors
        self.nc = m.nc  # number of classes
        self.nl = m.nl  # number of layers
        self.anchors = m.anchors
        self.device = device

    def __call__(self, p, targets):  # predictions, targets
        """

        Args:
            p:
            targets:

        Returns:

        """
        # p.shape = [torch.Size([1, 3, 80, 80, 85]), torch.Size([1, 3, 40, 40, 85]), torch.Size([1, 3, 20, 20, 85])]
        # type(targets) = <class 'torch.Tensor'>, targets.shape = torch.Size([28, 6])
        lcls = torch.zeros(1, device=self.device)  # class loss
        lbox = torch.zeros(1, device=self.device)  # box loss
        lobj = torch.zeros(1, device=self.device)  # object loss
        tcls, tbox, indices, anchors = self.build_targets(p, targets)  # targets

        # Losses
        for i, pi in enumerate(p):  # layer index, layer predictions
            b, a, gj, gi = indices[i]  # image, anchor, gridy, gridx
            tobj = torch.zeros(pi.shape[:4], dtype=pi.dtype, device=self.device)  # target obj

            n = b.shape[0]  # number of targets
            if n:
                # pxy, pwh, _, pcls = pi[b, a, gj, gi].tensor_split((2, 4, 5), dim=1)  # faster, requires torch 1.8.0
                pxy, pwh, _, pcls = pi[b, a, gj, gi].split((2, 2, 1, self.nc), 1)  # target-subset of predictions

                # Regression
                pxy = pxy.sigmoid() * 2 - 0.5
                pwh = (pwh.sigmoid() * 2) ** 2 * anchors[i]
                pbox = torch.cat((pxy, pwh), 1)  # predicted box
                iou = bbox_iou(pbox, tbox[i], CIoU=True).squeeze()  # iou(prediction, target)
                lbox += (1.0 - iou).mean()  # iou loss

                # Objectness
                iou = iou.detach().clamp(0).type(tobj.dtype)
                if self.sort_obj_iou:
                    j = iou.argsort()
                    b, a, gj, gi, iou = b[j], a[j], gj[j], gi[j], iou[j]
                if self.gr < 1:
                    iou = (1.0 - self.gr) + self.gr * iou
                tobj[b, a, gj, gi] = iou  # iou ratio

                # Classification
                if self.nc > 1:  # cls loss (only if multiple classes)
                    t = torch.full_like(pcls, self.cn, device=self.device)  # targets
                    t[range(n), tcls[i]] = self.cp
                    lcls += self.BCEcls(pcls, t)  # BCE

                # Append targets to text file
                # with open('targets.txt', 'a') as file:
                #     [file.write('%11.5g ' * 4 % tuple(x) + '\n') for x in torch.cat((txy[i], twh[i]), 1)]

            obji = self.BCEobj(pi[..., 4], tobj)
            # 每个 feature map 的置信度损失权重不同  要乘以相应的权重系数 self.balance[i]
            # 一般来说，检测小物体的难度大一点，所以会增加大特征图的损失系数，让模型更加侧重小物体的检测
            lobj += obji * self.balance[i]  # obj loss
            if self.autobalance:
                # 自动更新各个 feature map 的置信度损失系数
                self.balance[i] = self.balance[i] * 0.9999 + 0.0001 / obji.detach().item()

        if self.autobalance:
            self.balance = [x / self.balance[self.ssi] for x in self.balance]

        # 根据超参中的损失权重参数 对各个损失进行平衡  防止总损失被某个损失主导
        lbox *= self.hyp['box']
        lobj *= self.hyp['obj']
        lcls *= self.hyp['cls']
        bs = tobj.shape[0]  # batch size
        # print(f"\n\nhyp['box'] = {self.hyp['box']}, hyp['obj'] = {self.hyp['obj']}, hyp['cls'] = {self.hyp['cls']}, bs = {bs}")

        return (lbox + lobj + lcls) * bs, torch.cat((lbox, lobj, lcls)).detach()

    def build_targets(self, p, targets):
        """
        Build targets for compute_loss(), input targets(image,class,x,y,w,h)
        Args:
            p: 只用来获取每个detect head的输出尺寸，anchors, shape = self.anchors[i], p[i].shape
            targets: targets.shape = [nt, 6]

        Returns:

        """
        # na = 3, nt = 28
        na, nt = self.na, targets.shape[0]  # number of anchors, targets
        tcls, tbox, indices, anch = [], [], [], []
        # gain.shape = [7]
        gain = torch.ones(7, device=self.device)  # normalized to gridspace gain
        # ai用来标记下每个target属于哪个anchor，ai.shape = (na, nt), 第一行nt个0，第二行nt个1，...第na-1行nt个na-1
        # torch.arange(na, device=self.device).float() = tensor([0., 1., 2.])
        # torch.arange(na, device=self.device).float().view(na, 1) = tensor([[0.],
        #                                                                    [1.],
        #                                                                    [2.]])
        # ai = tensor([[0., 0., 0., ..., 0.],
        #              [1., 1., 1., ..., 1.],
        #              [2., 2., 2., ..., 2.]])
        ai = torch.arange(na, device=self.device).float().view(na, 1).repeat(1, nt)  # same as .repeat_interleave(nt)
        # targets.repeat(na, 1, 1).shape = [na, nt, 6]
        # ai[..., None].shape = [na, nt, 1]
        # targets.shape = [na, nt, 7]
        # 7: [img_idx, class_idx, x, y, w, h, anchor_idx]
        # targets的内容为：
        # targets[0, :] = [img_idx, class_idx, x, y, w, h, 0]
        # targets[1, :] = [img_idx, class_idx, x, y, w, h, 1]
        # ... ...
        # targets[na-1, :] = [img_idx, class_idx, x, y, w, h, na-1]
        # 每一行的7个信息可以理解为：该target属于哪个image，是哪个class，bbox位置是xywh，对应于na个base_anchor的哪一个
        targets = torch.cat((targets.repeat(na, 1, 1), ai[..., None]), 2)  # append anchor indices

        g = 0.5  # bias
        off = torch.tensor(
            [
                [0, 0],
                [1, 0],
                [0, 1],
                [-1, 0],
                [0, -1],  # j,k,l,m
                # [1, 1], [1, -1], [-1, 1], [-1, -1],  # jk,jm,lk,lm
            ],
            device=self.device).float() * g  # offsets

        for i in range(self.nl):
            # p[i].shape = [[1, 3, 80, 80, 85], [1, 3, 40, 40, 85], [1, 3, 20, 20, 85]]
            # anchors[i].shape = [3, 2], 该head有3个base_anchor，每个anchor对应wh两个参数
            # 注意，model创建时，已经根据每个detect head对应的stride（例如32），将anchor的wh映射到对应的尺度上了（w /= stride）
            anchors, shape = self.anchors[i], p[i].shape
            # gain[2:6] save the scale of each feature map
            # gain[2:6] = tensor([80., 80., 80., 80.]) / tensor([40., 40., 40., 40.]) / tensor([20., 20., 20., 20.])
            gain[2:6] = torch.tensor(shape)[[3, 2, 3, 2]]  # xyxy gain

            # Match targets to anchors
            # targets的xywh是在1x1的图上（范围是0~1），这里将其映射到feature map大小（例如80x80）的图上（xywh都乘以80）
            # t.shape = [na, nt, 7]
            t = targets * gain
            if nt:  # 如果有目标则开始匹配
                # Matches
                # 计算每个gt_box与当前层的三个base_anchor的宽高比(gt_w/anchor_w  gt_h/anchor_h)
                # r.shape = [3, 28, 2], r表示第i个(共3个)base_anchor与第j个(共28个)gt_box的宽高比(2个)
                r = t[..., 4:6] / anchors[:, None]  # wh ratio
                # torch.max(r, 1 / r): r和1/r分别代表gt/anchor和anchor/gt，意思是，无论gt和anchor谁比较大
                # 只要相互的比值超过了阈值，gt就会被过滤掉，不参与计算（尺度差别太大导致超出了这一层anchor的检测能力）
                # j.shape = [3, 28]，表示28个gt_box与3个base_anchor的长宽比筛选结果，True/False
                j = torch.max(r, 1 / r).max(2)[0] < self.hyp['anchor_t']  # compare
                # yolov3 v4的筛选方法: wh_iou  GT与anchor的wh_iou超过一定的阈值就是正样本
                # j = wh_iou(anchors, t[:, 4:6]) > model.hyp['iou_t']  # iou(3,n)=wh_iou(anchors(3,2), gwh(n,2))
                
                # 根据筛选条件j, 过滤gt
                # t = [3, 28, 7], j = [3, 28] -> t[j] = [num_positive, 7], num_positive是j里面True的总数
                t = t[j]  # filter
                # 此时，知道了当前obj的坐标, obj属于哪张图片, base_anchor的idx, 也就得到了当前obj由base_anchor
                # 中的哪一个负责预测

                # Offsets
                # 对筛选出的num_positive个target，判断其上下左右四个相邻cell是否也参与预测该target
                # grid xy, target中心点坐标（该坐标是相对当前feature map左上角的坐标, 例如在80x80中的xy坐标）
                # gxy.shape = [num_positive, 2]
                gxy = t[:, 2:4]
                # gain.shape = torch.Size([7])
                # gain[[2, 3]].shape = torch.Size([2])
                # gxy.shape = torch.Size([26, 2])
                # gxi.shape = torch.Size([26, 2]), gxi是gxy的对角位置坐标
                gxi = gain[[2, 3]] - gxy  # inverse
                # gxy % 1 < g 与 gxi % 1 < g，保证j,k,l,m中一定有2个是True，2个是False
                # j.shape = k.shape = l.shape = m.shape = [num_positive]
                # 分别表示对于num_positive个target，每个target的4个方向的cell是否被选择
                # j - 左，k - 上，l - 右，m - 下
                j, k = ((gxy % 1 < g) & (gxy > 1)).T  # gxy % 1表示取xy小数部分
                l, m = ((gxi % 1 < g) & (gxi > 1)).T  # gxy > 1表示xy都不在边缘cell里（边缘cell没有4个相邻cell）
                # j.shape = torch.Size([5, num_positive])，5表示中心cell+上下左右4个cell
                # j整体表示，对于num_positive个target中的每一个，其上下左右中5个cell，哪些cell被选出参与预测
                # torch.ones_like(j)表示每个target的中心cell都被选择
                j = torch.stack((torch.ones_like(j), j, k, l, m))
                # 对于num_positive个target，每个target都考虑由5个cell对其预测，因此每个cell都分配一份该target的gt，如下所示:
                # t.repeat((5, 1, 1)) = torch.Size([5, num_positive, 7])
                # 5个cell中，中心cell是始终参与预测的，上下左右4个cell，有且只有2个参与预测，因此每个target最终由3个cell参与预测
                # j标记了5个cell中，哪3个参与预测，如下所示：
                # t.repeat((5, 1, 1))[j] = torch.Size([num_cells, 7]), num_cells = num_positive * 3
                t = t.repeat((5, 1, 1))[j]
                # torch.zeros_like(gxy)[None].shape = torch.Size([1, num_positive, 2])
                # off[:, None].shape = torch.Size([5, 1, 2])
                # (torch.zeros_like(gxy)[None] + off[:, None]).shape = torch.Size([5, num_positive, 2])
                # offsets.shape = torch.Size([num_cells, 2]), num_cells = num_positive * 3
                offsets = (torch.zeros_like(gxy)[None] + off[:, None])[j]
            else:
                t = targets[0]
                offsets = 0

            # Define
            # bc.shape = [num_cells, 2]
            # gxy.shape = [num_cells, 2], float
            # gwh.shape  = [num_cells, 2]
            # a.shape = [num_cells, 1]
            bc, gxy, gwh, a = t.chunk(4, 1)  # (image, class), grid xy, grid wh, anchor_idx
            a, (b, c) = a.long().view(-1), bc.long().T  # anchors_idx, image_idx, class_idx
            gij = (gxy - offsets).long()  # gij is int, gij.shape = torch.Size([num_cells, 2])
            gi, gj = gij.T  # grid indices, gi.shape = gj.shape = [num_cells]

            # Append
            indices.append((b, a, gj.clamp_(0, shape[2] - 1), gi.clamp_(0, shape[3] - 1)))  # image, anchor, grid
            tbox.append(torch.cat((gxy - gij, gwh), 1))  # box
            anch.append(anchors[a])  # anchors
            tcls.append(c)  # class

        return tcls, tbox, indices, anch
