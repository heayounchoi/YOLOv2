import config as cfg
from dataset.factory import get_imdb
from dataset.roidb import RoiDataset, detection_collate
from yolov2 import Yolov2
from torch.utils.data import DataLoader
from torch import optim
import torch
from torch.autograd import Variable
from torchinfo import summary
from torchvision.models import vgg16_bn, VGG16_BN_Weights
import wandb
import os


def train():
    wandb.init(
        project="YOLOv2",
        group="YOLOv2",
        name="2",
        notes="2",
    )

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(device)

    max_epochs = 160
    start_epoch = 1
    dataset = 'voc07trainval'
    num_workers = 8
    display_interval = 10
    save_interval = 5
    use_cuda = False
    resume = False
    decay_lrs = cfg.decay_lrs
    imdbval_name = 'voc_2007_trainval'
    output_dir = 'output'

    train_dataset = RoiDataset(get_imdb('voc_2007_trainval'))

    train_dataloader = DataLoader(train_dataset, batch_size=cfg.batch_size,
                                  shuffle=True, num_workers=num_workers,
                                  collate_fn=detection_collate, drop_last=True)

    model = Yolov2()

    optimizer = optim.SGD(model.parameters(), lr=cfg.lr,
                          momentum=cfg.momentum, weight_decay=cfg.weight_decay)

    resume_checkpoint_name = 'yolov2_epoch_{}.pth'.format(25)
    resume_checkpoint_path = os.path.join(output_dir, resume_checkpoint_name)
    print('resume from {}'.format(resume_checkpoint_path))

    checkpoint = torch.load(resume_checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model'])
    start_epoch = checkpoint['epoch'] + 1

    model.to(device)
    model.train()

    iters_per_epoch = int(len(train_dataset) / cfg.batch_size)

    for epoch in range(start_epoch, max_epochs+1):
        loss_temp = 0
        train_data_iter = iter(train_dataloader)

        for step in range(iters_per_epoch):

            im_data, boxes, gt_classes, num_obj = next(train_data_iter)
            im_data = im_data.to(device)
            boxes = boxes.to(device)
            gt_classes = gt_classes.to(device)
            num_obj = num_obj.to(device)

            im_data_variable = Variable(im_data)

            box_loss, iou_loss, class_loss = model(
                im_data_variable, boxes, gt_classes, num_obj, training=True)

            loss = box_loss.mean() + iou_loss.mean() \
                + class_loss.mean()

            optimizer.zero_grad()

            loss.backward()
            optimizer.step()

            loss_temp += loss.item()

            if (step + 1) % display_interval == 0:
                loss_temp /= display_interval

                iou_loss_v = iou_loss.mean().item()
                box_loss_v = box_loss.mean().item()
                class_loss_v = class_loss.mean().item()

                print("[epoch %2d][step %4d/%4d] loss: %.4f"
                      % (epoch, step+1, iters_per_epoch, loss_temp))
                print("\t\t\tiou_loss: %.4f, box_loss: %.4f, cls_loss: %.4f"
                      % (iou_loss_v, box_loss_v, class_loss_v))

                n_iter = (epoch - 1) * iters_per_epoch + step + 1
                wandb.log({"loss": loss_temp,
                           "iou_loss": iou_loss_v,
                           "box_loss": box_loss_v,
                           "cls_loss": class_loss_v},
                          step=n_iter)

                loss_temp = 0

        if epoch % save_interval == 0:
            save_name = os.path.join(
                'output', 'yolov2_epoch_{}.pth'.format(epoch))
            torch.save({
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
            }, save_name)


if __name__ == '__main__':
    train()
