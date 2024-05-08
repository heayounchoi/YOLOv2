import os
import argparse
import time
import numpy as np
import pickle
import torch
from torch.autograd import Variable
from PIL import Image
from yolov2 import Yolov2
from dataset.factory import get_imdb
from dataset.roidb import RoiDataset
from yolo_eval import yolo_eval
from util.visualize import draw_detection_boxes
import matplotlib.pyplot as plt
from util.network import WeightLoader
from torch.utils.data import DataLoader
import config as cfg


def prepare_im_data(img):
    im_info = dict()
    im_info['width'], im_info['height'] = img.size

    H, W = cfg.input_size
    im_data = img.resize((H, W))

    im_data = torch.from_numpy(np.array(im_data)).float() / 255

    im_data = im_data.permute(2, 0, 1).unsqueeze(0)

    return im_data, im_info


def test():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(device)

    conf_thresh = 0.005
    nms_thresh = 0.45

    dataset = 'voc07test'
    imdbval_name = 'voc_2007_test'
    output_dir = 'output'
    model_name = 'yolov2_epoch_25'
    num_workers = 1
    batch_size = 2

    val_imdb = get_imdb(imdbval_name)

    val_dataset = RoiDataset(val_imdb, train=False)
    val_dataloader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False)

    model = Yolov2()

    model_path = os.path.join(output_dir, model_name+'.pth')
    print('loading model from {}'.format(model_path))

    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model'])

    model.to(device)
    model.eval()

    dataset_size = len(val_imdb.image_index)

    all_boxes = [[[] for _ in range(dataset_size)]
                 for _ in range(val_imdb.num_classes)]

    det_file = os.path.join(output_dir, 'detections.pkl')

    img_id = -1
    with torch.no_grad():
        for batch, (im_data, im_infos) in enumerate(val_dataloader):
            im_data_variable = Variable(im_data).to(device)

            yolo_outputs = model(im_data_variable)
            for i in range(im_data.size(0)):
                img_id += 1
                output = [item[i].data for item in yolo_outputs]
                im_info = {'width': im_infos[i][0], 'height': im_infos[i][1]}
                detections = yolo_eval(output, im_info, conf_threshold=conf_thresh,
                                       nms_threshold=nms_thresh)
                print('im detect [{}/{}]'.format(img_id+1, len(val_dataset)))
                if len(detections) > 0:
                    for cls in range(val_imdb.num_classes):
                        inds = torch.nonzero(detections[:, -1] == cls).view(-1)
                        if inds.numel() > 0:
                            cls_det = torch.zeros((inds.numel(), 5))
                            cls_det[:, :4] = detections[inds, :4]
                            cls_det[:, 4] = detections[inds, 4] * \
                                detections[inds, 5]
                            all_boxes[cls][img_id] = cls_det.cpu().numpy()

    with open(det_file, 'wb') as f:
        pickle.dump(all_boxes, f, pickle.HIGHEST_PROTOCOL)

    val_imdb.evaluate_detections(all_boxes, output_dir=output_dir)


if __name__ == '__main__':
    test()
