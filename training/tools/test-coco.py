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
from utils import dense_crf


from optparse import OptionParser
parser = OptionParser()
parser.add_option("--images", dest="image_list", type='string',
                  help="Path to image")
parser.add_option("--dir", dest="data_dir", type='string',
                  help="Path to image")
parser.add_option("--model", dest="model", type='string',
                  help="Model weights")
parser.add_option("--net", dest="net", type='string',
                  help="Model weights", default='deploy.prototxt')
parser.add_option("--output", dest="output_dir", type='string',
                  help="Output png file name", default='')
parser.add_option("--smooth", dest="smooth", action='store_true',
                  help="Apply postprocessing")
parser.add_option('--gpu', dest='gpu_id',
                        help='GPU device id to use [0]',
                        default=0, type=int)
parser.add_option('--class', dest='class_num',
                    help='class number include bg',
                    default=81, type=int)

options, _ = parser.parse_args()

smooth = options.smooth
image_list = options.image_list
root_dir = options.data_dir
model = options.model
output = options.output_dir
class_num = options.class_num
if output and (not osp.isdir(output)):
    os.makedirs(output)

mean_pixel = np.array([104.0, 117.0, 123.0])

palette = [(0.0, 0.0, 0.0), (0.5, 0.0, 0.0), (0.0, 0.5, 0.0), (0.5, 0.5, 0.0),
           (0.0, 0.0, 0.5), (0.5, 0.0, 0.5), (0.0, 0.5, 0.5), (0.5, 0.5, 0.5),
           (0.25, 0.0, 0.0), (0.75, 0.0, 0.0), (0.25, 0.5, 0.0), (0.75, 0.5, 0.0),
           (0.25, 0.0, 0.5), (0.75, 0.0, 0.5), (0.25, 0.5, 0.5), (0.75, 0.5, 0.5),
           (0.0, 0.25, 0.0), (0.5, 0.25, 0.0), (0.0, 0.75, 0.0), (0.5, 0.75, 0.0),
           (0.0, 0.25, 0.5)]
my_cmap = mpl_colors.LinearSegmentedColormap.from_list('Custom cmap', palette, 21)

def get_confusion_matrix(gt_label, pred_label, class_num):
    """
    Calcute the confusion matrix by given label and pred
    :param gt_label: the ground truth label
    :param pred_label: the pred label
    :param class_num: the nunber of class
    :return: the confusion matrix
    """
    index = (gt_label * class_num + pred_label).astype('int32')
    # print index.shape
    label_count = np.bincount(index)
    confusion_matrix = np.zeros((class_num, class_num))

    for i_label in range(class_num):
        for i_pred_label in range(class_num):
            cur_index = i_label * class_num + i_pred_label
            if cur_index < len(label_count):
                confusion_matrix[i_label, i_pred_label] = label_count[cur_index]

    return confusion_matrix

def write_to_png_file(im, f):
    #global palette
    #palette_int = map(lambda x: map(lambda xx: int(255 * xx), x), palette)
    #w = png.Writer(size=(im.shape[1], im.shape[0]), bitdepth=8, palette=palette_int)
    #with open(f, "w") as ff:
    #    w.write(ff, im)
    cv2.imwrite(f, im)


def preprocess(image, size, mean_pixel=mean_pixel):

    image = np.array(image)

    image = nd.zoom(image.astype('float32'),
                    (size / float(image.shape[0]),
                    size / float(image.shape[1]), 1.0),
                    order=1)

    image = image[:, :, [2, 1, 0]]
    image = image - mean_pixel

    image = image.transpose([2, 0, 1])
    return np.expand_dims(image, 0)


def predict_mask(image_file, smooth=True):

    im = pylab.imread(image_file)
    d1, d2 = float(im.shape[0]), float(im.shape[1])

    scores_all = 0
    for size in [481]:
        im_process = preprocess(im, size)
        net.blobs['images'].reshape(*im_process.shape)
        net.blobs['images'].data[...] = im_process
        net.forward()
        scores = np.transpose(net.blobs['fc8-SEC'].data[0], [1, 2, 0])
        scores = nd.zoom(scores, (d1 / scores.shape[0], d2 / scores.shape[1], 1.0), order=1)
        scores_all += scores

    scores_exp = np.exp(scores_all - np.max(scores_all, axis=2, keepdims=True))
    probs = scores_exp / np.sum(scores_exp, axis=2, keepdims=True)

    eps = 0.00001
    probs[probs < eps] = eps

    if smooth:
        result = np.argmax(krahenbuhl2013.CRF(im, np.log(probs), scale_factor=1.0), axis=2)
        # result = np.argmax(dense_crf(probs, im), axis=2)
    else:
        result = np.argmax(probs, axis=2)

    return result.copy()


if __name__ == "__main__":

    caffe.set_device(options.gpu_id)
    caffe.set_mode_gpu()
    net = caffe.Net(options.net, model, caffe.TEST)

    image_ids = [i.strip() for i in open(image_list) if not i.strip() == '']
    data_dir = osp.join(root_dir, 'images/val2014')
    label_dir = osp.join(root_dir, 'seg_labels/val2014')
    confusion_matrix = np.zeros((class_num, class_num))

    for index, img_id in enumerate(image_ids):
        if index % 100 == 0:
            pos = confusion_matrix.sum(1)
            res = confusion_matrix.sum(0)
            tp = np.diag(confusion_matrix)

            IU_array = (tp / np.maximum(1.0, pos + res - tp))
            mean_IU = IU_array.mean()
            print({'meanIU':mean_IU, 'IU_array':IU_array})

        image_file = osp.join(data_dir, img_id+'.jpg')
        label_file = osp.join(label_dir, img_id+'.png')

        seg_pred = predict_mask(image_file, smooth=smooth)
        seg_gt = cv2.imread(label_file, cv2.IMREAD_GRAYSCALE)

        ignore_index = seg_gt != 255
        seg_gt = seg_gt[ignore_index]
        seg_pred = seg_pred[ignore_index]
        # show_all(gt, output)
        confusion_matrix += get_confusion_matrix(seg_gt, seg_pred, class_num)

        if output:
            save_path = osp.join(output, img_id+'.png')
            write_to_png_file(seg_pred, save_path)
        else:
            fig = plt.figure()

            ax = fig.add_subplot('121')
            ax.imshow(pylab.imread(image_file))

            ax = fig.add_subplot('122')
            ax.matshow(seg_pred, vmin=0, vmax=21, cmap=my_cmap)

            plt.show()
