import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch import Tensor
from typing import List, Tuple

def compute_iou(box1: Tensor, box2: Tensor) -> Tensor:
    """Compute the intersection over union of two set of boxes, each box is [x1,y1,x2,y2].
    Args:
      box1: (tensor) bounding boxes, sized [N,4].
      box2: (tensor) bounding boxes, sized [M,4].
    Return:
      (tensor) iou, sized [N,M].
    """
    N = box1.size(0)
    M = box2.size(0)

    lt = torch.max(
        box1[:, :2].unsqueeze(1).expand(N, M, 2),  # [N,2] -> [N,1,2] -> [N,M,2]
        box2[:, :2].unsqueeze(0).expand(N, M, 2),  # [M,2] -> [1,M,2] -> [N,M,2]
    )
    rb = torch.min(
        box1[:, 2:].unsqueeze(1).expand(N, M, 2),  # [N,2] -> [N,1,2] -> [N,M,2]
        box2[:, 2:].unsqueeze(0).expand(N, M, 2),  # [M,2] -> [1,M,2] -> [N,M,2]
    )

    wh = rb - lt  # [N,M,2]
    wh[wh < 0] = 0  # clip at 0
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    area1 = (box1[:, 2] - box1[:, 0]) * (box1[:, 3] - box1[:, 1])  # [N,]
    area2 = (box2[:, 2] - box2[:, 0]) * (box2[:, 3] - box2[:, 1])  # [M,]
    area1 = area1.unsqueeze(1).expand_as(inter)  # [N,] -> [N,1] -> [N,M]
    area2 = area2.unsqueeze(0).expand_as(inter)  # [M,] -> [1,M] -> [N,M]

    iou = inter / (area1 + area2 - inter)
    return iou


class YoloLoss(nn.Module):
    def __init__(self, S, B, l_coord, l_noobj):
        super().__init__()
        self.S = S
        self.B = B
        self.l_coord = l_coord
        self.l_noobj = l_noobj


    def xywh2xyxy(self, boxes):
        """
        Parameters:
        boxes: (N,4) representing by x,y,w,h

        Returns:
        boxes: (N,4) representing by x1,y1,x2,y2

        if for a Box b the coordinates are represented by [x, y, w, h] then
        x1, y1 = x/S - 0.5*w, y/S - 0.5*h ; x2,y2 = x/S + 0.5*w, y/S + 0.5*h
        Note: Over here initially x, y are the center of the box and w,h are width and height.
        """
        x1 = boxes[:, 0] / self.S - boxes[:, 2] / 2
        y1 = boxes[:, 1] / self.S - boxes[:, 3] / 2
        x2 = boxes[:, 0] / self.S + boxes[:, 2] / 2
        y2 = boxes[:, 1] / self.S + boxes[:, 3] / 2

        return torch.stack((x1, y1, x2, y2), dim=1)


    def find_best_iou_boxes(self, pred_box_list: List[Tensor], box_target: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Parameters:
        box_pred_list : [(tensor) size (-1, 5) ...]
        box_target : (tensor) size (-1, 4)

        Returns:
        best_iou: (tensor) size (-1, 1)
        best_boxes : (tensor) size (-1, 5), containing the boxes which give the best iou among the two (self.B) predictions

        Hints:
        1) Find the iou's of each of the 2 bounding boxes of each grid cell of each image.
        2) For finding iou's use the compute_iou function
        3) use xywh2xyxy to convert bbox format if necessary,
        Note: Over here initially x, y are the center of the box and w,h are width and height.
        We perform this transformation to convert the correct coordinates into bounding box coordinates.
        """
        N = box_target.size(0)
        box_target = self.xywh2xyxy(box_target)

        best_ious = -torch.ones(N).to('cuda')
        best_boxes = torch.zeros(N, 5).to('cuda')

        for b in range(len(pred_box_list)):
            iou = compute_iou(self.xywh2xyxy(pred_box_list[b][:, :4]), box_target) # (N, N)
            # get diagonal of iou matrix
            iou = iou.diag() # (N, )

            # boolean mask of the best iou
            best_idx = (iou > best_ious).nonzero().squeeze() # (N, )
            best_ious[best_idx] = iou[best_idx]
            best_boxes[best_idx, :] = pred_box_list[b][best_idx, :]

        best_ious = best_ious.unsqueeze(1).detach()

        return best_ious, best_boxes


    def get_class_prediction_loss(self, classes_pred: Tensor, classes_target: Tensor, has_object_map: Tensor) -> Tensor:
        """
        Parameters:
        classes_pred : (tensor) size (batch_size, S, S, 20)
        classes_target : (tensor) size (batch_size, S, S, 20)
        has_object_map: (tensor) size (batch_size, S, S)

        Returns:
        class_loss : scalar
        """
        classes_sum = has_object_map * (classes_pred - classes_target).pow(2).sum(dim=-1) # (batch_size, S, S)

        return classes_sum.sum()


    def get_no_object_loss(self, pred_boxes_list: List[Tensor], has_object_map: Tensor):
        """
        Parameters:
        pred_boxes_list: (list) [(tensor) size (N, S, S, 5)  for B pred_boxes]
        has_object_map: (tensor) size (N, S, S)

        Returns:
        loss : scalar

        Hints:
        1) Only compute loss for cell which doesn't contain object
        2) compute loss for all predictions in the pred_boxes_list list
        3) You can assume the ground truth confidence of non-object cells is 0
        """
        no_obj_loss = 0
        has_no_obj_map = ~has_object_map

        for pred_boxes in pred_boxes_list:
            no_obj_loss += (has_no_obj_map * pred_boxes[:, :, :, 4].squeeze().pow(2)).sum()

        no_obj_loss *= self.l_noobj

        return no_obj_loss


    def get_contain_conf_loss(self, box_pred_conf: Tensor, box_target_conf: Tensor):
        """
        Parameters:
        box_pred_conf : (tensor) size (-1,1)
        box_target_conf: (tensor) size (-1,1)

        Returns:
        contain_loss : scalar

        Hints:
        The box_target_conf should be treated as ground truth, i.e., no gradient

        """
        return (box_pred_conf - box_target_conf).pow(2).sum()


    def get_regression_loss(self, box_pred_response: Tensor, box_target_response: Tensor):
        """
        Parameters:
        box_pred_response : (tensor) size (-1, 4)
        box_target_response : (tensor) size (-1, 4)
        Note : -1 corresponds to ravels the tensor into the dimension specified
        See : https://pytorch.org/docs/stable/tensors.html#torch.Tensor.view_as

        Returns:
        reg_loss : scalar
        """
        box_pred_response[:, 2:] = box_pred_response[:, 2:].sqrt()
        box_target_response[:, :2] = box_target_response[:, :2].sqrt()

        return self.l_coord * (box_pred_response - box_target_response).pow(2).sum()


    def forward(self, pred_tensor: Tensor, target_boxes: Tensor, target_cls: Tensor, has_object_map: Tensor):
        """
        pred_tensor: (tensor) size(N,S,S,Bx5+20=30) N:batch_size
                      where B - number of bounding boxes this grid cell is a part of = 2
                            5 - number of bounding box values corresponding to [x, y, w, h, c]
                                where x - x_coord, y - y_coord, w - width, h - height, c - confidence of having an object
                            20 - number of classes

        target_boxes: (tensor) size (N, S, S, 4): the ground truth bounding boxes
        target_cls: (tensor) size (N, S, S, 20): the ground truth class
        has_object_map: (tensor, bool) size (N, S, S): the ground truth for whether each cell contains an object (True/False)

        Returns:
        loss_dict (dict): with key value stored for total_loss, reg_loss, containing_obj_loss, no_obj_loss and cls_loss
        """
        N = pred_tensor.size(0)
        total_loss = 0.0

        # split the pred tensor from an entity to separate tensors:
        # -- pred_boxes_list: a list containing all bbox prediction (list) [(tensor) size (N, S, S, 5)  for B pred_boxes]
        # -- pred_cls (containing all classification prediction)
        pred_boxes_list = []
        pred_cls = pred_tensor[:, :, :, self.B * 5:]

        for b in range(self.B):
            start, end = b * 5, (b + 1) * 5
            pred_boxes_list.append(pred_tensor[:, :, :, start : end])

        # compcute classification loss
        cls_loss = self.get_class_prediction_loss(pred_cls, target_cls, has_object_map)

        # compute no-object loss
        no_obj_loss = self.get_no_object_loss(pred_boxes_list, has_object_map)

        # Re-shape boxes in pred_boxes_list and target_boxes to meet the following desires
        # 1) only keep having-object cells
        # 2) vectorize all dimensions except for the last one for faster computation
        has_object_idx = torch.nonzero(has_object_map.view(-1)).squeeze() # (-1)

        for i in range(len(pred_boxes_list)):
            # reshape the boxes to meet the desire
            pred_boxes_list[i] = pred_boxes_list[i].contiguous().view(-1, 5) # (-1, 5)
            # keep only the cells which have objects
            pred_boxes_list[i] = pred_boxes_list[i][has_object_idx]

        target_boxes = target_boxes.view(-1, 4)[has_object_idx] # (-1, 4)
        target_cls = target_cls.view(-1, 20)[has_object_idx] # (-1, 20)

        # find the best boxes among the 2 (or self.B) predicted boxes and the corresponding iou
        best_ious, best_boxes = self.find_best_iou_boxes(pred_boxes_list, target_boxes)

        # compute regression loss between the found best bbox and GT bbox for all the cell containing objects
        reg_loss = self.get_regression_loss(best_boxes[:, :4], target_boxes)

        # compute contain_object_loss
        contain_obj_loss = self.get_contain_conf_loss(best_boxes[:, 4], best_ious)

        # compute final loss
        total_loss += cls_loss + no_obj_loss + contain_obj_loss + reg_loss

        # construct return loss_dict
        loss_dict = dict(
            total_loss = total_loss,
            reg_loss = reg_loss,
            containing_obj_loss = contain_obj_loss,
            no_obj_loss = no_obj_loss,
            cls_loss = cls_loss,
        )

        return loss_dict
