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

options, _ = parser.parse_args()

smooth = options.smooth
image_list = options.image_list
data_dir = options.data_dir
model = options.model
output = options.output_dir
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


def write_to_png_file(im, f):
    #global palette
    #palette_int = map(lambda x: map(lambda xx: int(255 * xx), x), palette)
    #w = png.Writer(size=(im.shape[1], im.shape[0]), bitdepth=8, palette=palette_int)
    #with open(f, "w") as ff:
    #    w.write(ff, im)
    cv2.imwrite(f, im)


# def preprocess(image, size, mean_pixel=mean_pixel):

#     image = np.array(image)
#     # image = nd.zoom(image.astype('float32'),
#     #                 (size / float(image.shape[0]),
#     #                 size / float(image.shape[1]), 1.0),
#     #                 order=1)
#     factor = min(float(size)/image.shape[0], float(size)/image.shape[1])
#     image = cv2.resize(image, None, fx=factor, fy=factor, interpolation=cv2.INTER_LINEAR)
#     shape = image.shape

#     # image = image[:, :, [2, 1, 0]]
#     image = image - mean_pixel
#     img_h, img_w, _ = image.shape
#     pad_h = max(size - img_h, 0)
#     pad_w = max(size - img_w, 0)
#     if pad_h > 0 or pad_w > 0:
#         img_pad = cv2.copyMakeBorder(image, 0, pad_h, 0, 
#             pad_w, cv2.BORDER_CONSTANT, 
#             value=(0.0,0.0,0.0))
#     else:
#         img_pad = image

#     img_h, img_w, _ = img_pad.shape
    
#     h_off = (img_h - size) / 2
#     w_off = (img_w - size) / 2
#     image = np.asarray(img_pad[h_off : h_off+size, w_off : w_off+size], np.float32)

#     image = image.transpose([2, 0, 1])
#     return np.expand_dims(image, 0), shape

def preprocess(image, size, mean_pixel=mean_pixel):

    image = np.array(image)

    image = nd.zoom(image.astype('float32'),
                    (size, size, 1.0),
                    order=1)

    image = image[:, :, [2, 1, 0]]
    image = image - mean_pixel

    image = image.transpose([2, 0, 1])
    return np.expand_dims(image, 0)


def predict_mask(image_file, smooth=True):

    im = pylab.imread(image_file)
    d1, d2 = float(im.shape[0]), float(im.shape[1])

    scores_all = 0
    for size in [0.75, 1, 1.25]: #[385, 513, 641]
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

    return result


if __name__ == "__main__":

    caffe.set_device(options.gpu_id)
    caffe.set_mode_gpu()
    net = caffe.Net(options.net, model, caffe.TEST)

    image_ids = [i.strip() for i in open(image_list) if not i.strip() == '']
    data_dir = osp.join(data_dir, 'JPEGImages')
    for index, img_id in enumerate(image_ids):
        print index, img_id
        image_file = osp.join(data_dir, img_id+'.jpg')
        mask = predict_mask(image_file, smooth=smooth)

        if output:
            save_path = osp.join(output, img_id+'.png')
            write_to_png_file(mask, save_path)
        else:
            fig = plt.figure()

            ax = fig.add_subplot('121')
            ax.imshow(pylab.imread(image_file))

            ax = fig.add_subplot('122')
            ax.matshow(mask, vmin=0, vmax=21, cmap=my_cmap)

            plt.show()
