import numpy as np
import pylab

import scipy.ndimage as nd

import png
import cv2, os
import os.path as osp

from matplotlib import pyplot as plt
from matplotlib import colors as mpl_colors

import krahenbuhl2013

import findcaffe
import caffe
import shutil

from optparse import OptionParser
parser = OptionParser()
parser.add_option("--images", dest="image_list", type='string',
                  help="Path to image")
parser.add_option("--output", dest="output_dir", type='string',
                  help="Output png file name", default='')
parser.add_option("--model", dest="model", type='string',
                  help="Model weights")
parser.add_option("--net", dest="net", type='string',
                  help="Model weights", default='deploy.prototxt')
parser.add_option("--smooth", dest="smooth", action='store_true',
                  help="Apply postprocessing")
parser.add_option('--gpu', dest='gpu_id',
                        help='GPU device id to use [0]',
                        default=0, type=int)

options, _ = parser.parse_args()

smooth = options.smooth
image_list = options.image_list
output = options.output_dir
if output and (not osp.isdir(output)):
    os.makedirs(output)

mean_pixel = np.array([104.0, 117.0, 123.0])

palette = [(0.0, 0.0, 0.0), (0.5, 0.0, 0.0), (0.0, 0.5, 0.0), (0.5, 0.5, 0.0),
           (0.0, 0.0, 0.5), (0.5, 0.0, 0.5), (0.0, 0.5, 0.5), (0.5, 0.5, 0.5),
           (0.25, 0.0, 0.0), (0.75, 0.0, 0.0), (0.25, 0.5, 0.0), (0.75, 0.5, 0.0),
           (0.25, 0.0, 0.5), (0.75, 0.0, 0.5), (0.25, 0.5, 0.5), (0.75, 0.5, 0.5),
           (0.0, 0.25, 0.0), (0.5, 0.25, 0.0), (0.0, 0.75, 0.0), (0.5, 0.75, 0.0),
           (0.0, 0.25, 0.5), (1.0, 1.0, 1.0)]
my_cmap = mpl_colors.LinearSegmentedColormap.from_list('Custom cmap', palette, 22)
SEED_DATA = None
DATA_DIR = '/workspace/hzl/wsis/data/voc12'

def write_to_png_file(im, f):
    global palette
    palette_int = map(lambda x: map(lambda xx: int(255 * xx), x), palette)
    w = png.Writer(size=(im.shape[1], im.shape[0]), bitdepth=8, palette=palette_int)
    with open(f, "w") as ff:
       w.write(ff, im)
    # cv2.imwrite(f, im)


def preprocess(image, size, mean_pixel=mean_pixel):

    image = np.array(image)

    image = nd.zoom(image.astype('float32'),
                    (size / float(image.shape[0]),
                    size / float(image.shape[1]), 1.0),
                    order=1)

    image = image[:, :, [2, 1, 0]]
    image = image - mean_pixel

    image = image.transpose([2, 0, 1])
    return image


def predict_mask(image_file, smooth=True):

    im = pylab.imread(image_file)

    net.blobs['images'].data[0] = preprocess(im, 321)
    net.forward()

    scores = np.transpose(net.blobs['fc8-SEC'].data[0], [1, 2, 0])
    d1, d2 = float(im.shape[0]), float(im.shape[1])

    scores_exp = np.exp(scores - np.max(scores, axis=2, keepdims=True))
    probs = scores_exp / np.sum(scores_exp, axis=2, keepdims=True)
    probs = nd.zoom(probs, (d1 / probs.shape[0], d2 / probs.shape[1], 1.0), order=1)

    img_id = os.path.splitext(os.path.basename(image_file))[0]
    np.save(img_id, probs)
    eps = 0.00001
    probs[probs < eps] = eps

    if smooth:
        result = np.argmax(krahenbuhl2013.CRF(im, np.log(probs), scale_factor=1.0), axis=2)
    else:
        result = np.argmax(probs, axis=2)

    return result


if __name__ == "__main__":

    caffe.set_device(options.gpu_id)
    caffe.set_mode_gpu()
    net = caffe.Net(options.net, options.model, caffe.TEST)

    image_ids = [i.strip() for i in open(image_list) if not i.strip() == '']
    data_dir = osp.join('/workspace/hzl/wsis/data/voc12/', 'JPEGImages')
    gt_dir = osp.join('/workspace/hzl/wsis/data/voc12/', 'SegmentationClassAug')
    for index, img_id in enumerate(image_ids):
        print index, img_id
        image_file = osp.join(data_dir, img_id+'.jpg')
        mask = predict_mask(image_file, smooth=smooth)
        gt = cv2.imread(osp.join(gt_dir, img_id+'.png'), cv2.IMREAD_GRAYSCALE)

        shutil.copyfile(image_file, osp.join(output, img_id+'_im.jpg'))
        save_path = osp.join(output, img_id+'_pred.png')
        write_to_png_file(mask, save_path)
        write_to_png_file(gt, osp.join(output, img_id+'_gt.png'))
