import cv2
import numpy as np
from PIL import Image
import config as cfg


def random_scale_translation(img, boxes, jitter=0.2):
    w, h = img.size

    dw = int(w*jitter)
    dh = int(h*jitter)

    pl = np.random.randint(-dw, dw)
    pr = np.random.randint(-dw, dw)
    pt = np.random.randint(-dh, dh)
    pb = np.random.randint(-dh, dh)

    sw = w - pl - pr
    sh = h - pt - pb

    cropped = img.crop((pl, pt, pl + sw - 1, pt + sh - 1))

    boxes[:, 0::2] -= pl
    boxes[:, 1::2] -= pt

    boxes[:, 0::2] = boxes[:, 0::2].clip(0, sw-1)
    boxes[:, 1::2] = boxes[:, 1::2].clip(0, sh-1)

    if np.random.randint(2):
        cropped = cropped.transpose(Image.FLIP_LEFT_RIGHT)
        boxes[:, 0::2] = (sw-1) - boxes[:, 2::-2]

    return cropped, boxes


def convert_color(img, source, dest):
    if source == 'RGB' and dest == 'HSV':
        img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    elif source == 'HSV' and dest == 'RGB':
        img = cv2.cvtColor(img, cv2.COLOR_HSV2RGB)
    return img


def rand_scale(s):
    scale = np.random.uniform(1, s)
    if np.random.randint(1, 10000) % 2:
        return scale
    return 1./scale


def random_distort(img, hue=.1, sat=1.5, val=1.5):

    hue = np.random.uniform(-hue, hue)
    sat = rand_scale(sat)
    val = rand_scale(val)

    img = img.convert('HSV')
    cs = list(img.split())
    cs[1] = cs[1].point(lambda i: i * sat)
    cs[2] = cs[2].point(lambda i: i * val)

    def change_hue(x):
        x += hue * 255
        if x > 255:
            x -= 255
        if x < 0:
            x += 255
        return x

    cs[0] = cs[0].point(change_hue)
    img = Image.merge(img.mode, tuple(cs))

    img = img.convert('RGB')
    return img


def random_hue(img, rate=.1):
    delta = rate * 360.0 / 2

    if np.random.randint(2):
        img[:, :, 0] += np.random.uniform(-delta, delta)
        img[:, :, 0] = np.clip(img[:, :, 0], a_min=0.0, a_max=360.0)

    return img


def random_saturation(img, rate=1.5):
    lower = 0.5  # hard code
    upper = rate

    if np.random.randint(2):
        img[:, :, 1] *= np.random.uniform(lower, upper)
        img[:, :, 1] = np.clip(img[:, :, 1], a_min=0.0, a_max=1.0)

    return img


def random_exposure(img, rate=1.5):
    lower = 0.5  # hard code
    upper = rate

    if np.random.randint(2):
        img[:, :, 2] *= np.random.uniform(lower, upper)
        img[:, :, 2] = np.clip(img[:, :, 2], a_min=0.0, a_max=255.0)

    return img


def augment_img(img, boxes, gt_classes):
    boxes = np.copy(boxes).astype(np.float32)

    for i in range(5):
        img_t, boxes_t = random_scale_translation(
            img.copy(), boxes.copy(), jitter=cfg.jitter)
        keep = (boxes_t[:, 0] != boxes_t[:, 2]) & (
            boxes_t[:, 1] != boxes_t[:, 3])
        boxes_t = boxes_t[keep, :]
        if boxes_t.shape[0] > 0:
            img = img_t
            boxes = boxes_t
            gt_classes = gt_classes[keep]
            break

    img = random_distort(img, cfg.hue, cfg.saturation, cfg.exposure)
    return img, boxes, gt_classes
