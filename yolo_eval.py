from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
from util.bbox import generate_all_anchors, xywh2xxyy, box_transform_inv, xxyy2xywh
from util.bbox import box_ious
import time
import config as cfg


def yolo_filter_boxes(boxes_pred, conf_pred, classes_pred, confidence_threshold=0.6):
    cls_max_conf, cls_max_id = torch.max(classes_pred, dim=-1, keepdim=True)
    cls_conf = conf_pred * cls_max_conf

    pos_inds = (cls_conf > confidence_threshold).view(-1)

    filtered_boxes = boxes_pred[pos_inds, :]

    filtered_conf = conf_pred[pos_inds, :]

    filtered_cls_max_conf = cls_max_conf[pos_inds, :]

    filtered_cls_max_id = cls_max_id[pos_inds, :]

    return filtered_boxes, filtered_conf, filtered_cls_max_conf, filtered_cls_max_id.float()


def yolo_nms(boxes, scores, threshold):
    score_sort_index = torch.sort(scores, dim=0, descending=True)[1]

    keep = []

    while score_sort_index.numel() > 0:

        i = score_sort_index[0]
        keep.append(i)

        if score_sort_index.numel() == 1:
            break

        cur_box = boxes[score_sort_index[0], :].view(-1, 4)
        res_box = boxes[score_sort_index[1:], :].view(-1, 4)

        ious = box_ious(cur_box, res_box).view(-1)

        inds = torch.nonzero(ious < threshold).squeeze()

        score_sort_index = score_sort_index[inds + 1].view(-1)

    return torch.LongTensor(keep)


def generate_prediction_boxes(deltas_pred):
    H = int(cfg.test_input_size[0] / cfg.strides)
    W = int(cfg.test_input_size[1] / cfg.strides)

    anchors = torch.FloatTensor(cfg.anchors)
    all_anchors_xywh = generate_all_anchors(anchors, H, W)

    all_anchors_xywh = deltas_pred.new(
        *all_anchors_xywh.size()).copy_(all_anchors_xywh)

    boxes_pred = box_transform_inv(all_anchors_xywh, deltas_pred)

    return boxes_pred


def scale_boxes(boxes, im_info):
    h = im_info['height']
    w = im_info['width']

    input_h, input_w = cfg.test_input_size
    scale_h, scale_w = input_h / h, input_w / w

    boxes *= cfg.strides

    boxes[:, 0::2] /= scale_w
    boxes[:, 1::2] /= scale_h

    boxes = xywh2xxyy(boxes)

    boxes[:, 0::2].clamp_(0, w-1)
    boxes[:, 1::2].clamp_(0, h-1)

    return boxes


def yolo_eval(yolo_output, im_info, conf_threshold=0.6, nms_threshold=0.4):
    deltas = yolo_output[0].cpu()
    conf = yolo_output[1].cpu()
    classes = yolo_output[2].cpu()

    num_classes = classes.size(1)

    boxes = generate_prediction_boxes(deltas)

    if cfg.debug:
        print('check box: ', boxes.view(13*13, 5, 4).permute(1,
              0, 2).contiguous().view(-1, 4)[0:10, :])
        print('check conf: ', conf.view(
            13*13, 5).permute(1, 0).contiguous().view(-1)[:10])

    boxes, conf, cls_max_conf, cls_max_id = yolo_filter_boxes(
        boxes, conf, classes, conf_threshold)

    if boxes.size(0) == 0:
        return []

    boxes = scale_boxes(boxes, im_info)

    if cfg.debug:
        all_boxes = torch.cat([boxes, conf, cls_max_conf, cls_max_id], dim=1)
        print('check all boxes: ', all_boxes)
        print('check all boxes len: ', len(all_boxes))

    detections = []

    cls_max_id = cls_max_id.view(-1)

    for cls in range(num_classes):
        cls_mask = cls_max_id == cls
        inds = torch.nonzero(cls_mask).squeeze()

        if inds.numel() == 0:
            continue

        boxes_pred_class = boxes[inds, :].view(-1, 4)
        conf_pred_class = conf[inds, :].view(-1, 1)
        cls_max_conf_class = cls_max_conf[inds].view(-1, 1)
        classes_class = cls_max_id[inds].view(-1, 1)

        nms_keep = yolo_nms(
            boxes_pred_class, conf_pred_class.view(-1), nms_threshold)

        boxes_pred_class_keep = boxes_pred_class[nms_keep, :]
        conf_pred_class_keep = conf_pred_class[nms_keep, :]
        cls_max_conf_class_keep = cls_max_conf_class.view(-1, 1)[nms_keep, :]
        classes_class_keep = classes_class.view(-1, 1)[nms_keep, :]

        seq = [boxes_pred_class_keep, conf_pred_class_keep,
               cls_max_conf_class_keep, classes_class_keep.float()]

        detections_cls = torch.cat(seq, dim=-1)
        detections.append(detections_cls)

    return torch.cat(detections, dim=0)
