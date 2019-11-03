import numpy as np
import cv2
import random
import copy
from parse_data_dict import load_obj

def anchor_gen(ratios, scales,resize_image=(224,224),featureMap_size=(14,14),anchor_stride=1):
    rpn_stride=int(resize_image[0]/featureMap_size[0])
    ratios, scales = np.meshgrid(ratios, scales)
    ratios, scales = ratios.flatten(), scales.flatten()

    width = scales / np.sqrt(ratios)
    height = scales * np.sqrt(ratios)

    shift_x = np.arange(0, featureMap_size[0], anchor_stride) * rpn_stride
    shift_y = np.arange(0, featureMap_size[1], anchor_stride) * rpn_stride
    shift_x, shift_y = np.meshgrid(shift_x, shift_y)
    centerX, anchorX = np.meshgrid(shift_x, width)
    centerY, anchorY = np.meshgrid(shift_y, height)
    boxCenter = np.stack([centerY, centerX], axis=2).reshape(-1, 2)
    boxSize = np.stack([anchorX, anchorY], axis=2).reshape(-1, 2)

    boxes = np.concatenate([boxCenter - 0.5 * boxSize, boxCenter + 0.5 * boxSize], axis=1)
    return boxes

def compute_iou(box, boxes, area, areas):
    y1 = np.maximum(box[0], boxes[:, 0])
    x1 = np.maximum(box[1], boxes[:, 1])
    y2 = np.minimum(box[2], boxes[:, 2])
    x2 = np.minimum(box[3], boxes[:, 3])
    interSec = np.maximum(y2-y1, 0) * np.maximum(x2-x1, 0)
    union = areas[:] + area - interSec
    iou = interSec / union
    return iou

def compute_overlap(boxes1, boxes2):
    areas1 = (boxes1[:,3] - boxes1[:,1]) * (boxes1[:,2] - boxes1[:,0])
    areas2 = (boxes2[:,3] - boxes2[:,1]) * (boxes2[:,2] - boxes2[:,0])
    overlap = np.zeros((boxes1.shape[0], boxes2.shape[0]))
    for i in range(boxes2.shape[0]):
        box = boxes2[i]
        overlap[:,i] = compute_iou(box, boxes1, areas2[i], areas1)
    return overlap

def accord_bbox(img_data,*size_network):
    resized_width,resized_height=size_network
    num_bboxes=len(img_data["bboxes"])
    width=img_data["width"]
    height=img_data["height"]
    gta = np.zeros((num_bboxes, 4))
    for bbox_num, bbox in enumerate(img_data['bboxes']):
        # get the GT box coordinates, and resize to account for image resizing
        gta[bbox_num, 1] = bbox['x1'] * (resized_width / float(width))
        gta[bbox_num, 3] = bbox['x2'] * (resized_width / float(width))
        gta[bbox_num, 0] = bbox['y1'] * (resized_height / float(height))
        gta[bbox_num, 2] = bbox['y2'] * (resized_height / float(height))
    gta_bbox=gta
    return gta_bbox

def build_rpnTarget(boxes, anchors, config):
    rpn_match = np.zeros(anchors.shape[0], dtype=np.int32)
    rpn_bboxes = np.zeros((config.train_rois_num, 4))

    iou = compute_overlap(anchors, boxes)
    maxArg_iou = np.argmax(iou, axis=1)
    max_iou = iou[np.arange(iou.shape[0]), maxArg_iou]
    postive_anchor_idxs = np.where(max_iou > 0.4)[0]
    negative_anchor_idxs = np.where(max_iou < 0.1)[0]

    rpn_match[postive_anchor_idxs] = 1
    rpn_match[negative_anchor_idxs] = -1
    maxIou_anchors = np.argmax(iou, axis=0)
    rpn_match[maxIou_anchors] = 1

    ids = np.where(rpn_match == 1)[0]
    extral = len(ids) - config.train_rois_num // 2
    if extral > 0:
        ids_ = np.random.choice(ids, extral, replace=False)
        rpn_match[ids_] = 0

    ids = np.where(rpn_match == -1)[0]
    extral = len(ids) - (config.train_rois_num - np.where(rpn_match == 1)[0].shape[0])
    if extral > 0:
        ids_ = np.random.choice(ids, extral, replace=False)
        rpn_match[ids_] = 0

    idxs = np.where(rpn_match == 1)[0]
    ix = 0
    for i, a in zip(idxs, anchors[idxs]):
        gt = boxes[maxArg_iou[i]]

        gt_h = gt[2] - gt[0]
        gt_w = gt[3] - gt[1]
        gt_centy = gt[0] + 0.5 * gt_h
        gt_centx = gt[1] + 0.5 * gt_w

        a_h = a[2] - a[0]
        a_w = a[3] - a[1]
        a_centy = a[0] + 0.5 * a_h
        a_centx = a[1] + 0.5 * a_w

        rpn_bboxes[ix] = [(gt_centy - a_centy) / a_h, (gt_centx - a_centx) / a_w,
                          np.log(gt_h / a_h), np.log(gt_w / a_w)]
        rpn_bboxes[ix] /= config.RPN_BBOX_STD_DEV
        ix += 1
    return rpn_match, rpn_bboxes

data=load_obj("test_1")
anchors=anchor_gen(ratios = [0.5, 1, 2],scales = [4, 8, 16])
bacth_size=32
train=[]
valdation=[]
for i in range(len(data)):
    if data[i]["imageset"]=="trian":
        train.append(data[i])
    if data[i]["imageset"]=="test":
        valdation.append(data[i])
def generator_rpn(data,batch_size):
    length=len(data)
    for  i in range(length//batch_size-1):
        img_collection=data[i*batch_size:(i+1)*batch_size]
        input_data = []
        batch_rpn_match = []
        batch_rpn_bboxes = []
        for j in img_collection:
            img=cv2.imread(j["filepath"])
            boxes=accord_bbox(j, (224,224))
            rpn_match,rpn_bbox=build_rpnTarget(boxes, anchors, config)
            input_data.append(img)
            batch_rpn_match.append(batch_rpn_match)
            batch_rpn_bboxes.append(batch_rpn_bboxes)
        yield np.stack(input_data),np.stack(batch_rpn_match),np.stack(batch_rpn_bboxes)







