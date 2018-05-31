# imports
import json
import time
import pickle
import scipy.misc
import skimage.io
import cv2
import caffe

import numpy as np
import os.path as osp

from random import shuffle
from PIL import Image
import random

class ImageSegDataLayer(caffe.Layer):

    """
    This is a simple syncronous datalayer for training a Detection model on
    PASCAL.
    """

    def setup(self, bottom, top):

        self.top_names = ['data', 'label']

        # === Read input parameters ===
        # params is a python dictionary with layer parameters.
        params = eval(self.param_str)

        SimpleTransformer.check_params(params)
        # store input as class variables
        self.batch_size = params['batch_size']
        self.input_shape = params['crop_size']

        # Create a batch loader to load the images.
        self.batch_loader = BatchLoader(params)

        # === reshape tops ===
        # since we use a fixed input image size, we can shape the data layer
        # once. Else, we'd have to do it in the reshape call.
        top[0].reshape(
            self.batch_size, 3, self.input_shape[0], self.input_shape[1])
        # Note the 20 channels (because PASCAL has 20 classes.)
        top[1].reshape(
            self.batch_size, 1, self.input_shape[0], self.input_shape[1])

        print_info("ImageSegDataLayer", params)

    def forward(self, bottom, top):
        """
        Load data.
        """
        for itt in range(self.batch_size):
            # Use the batch loader to load the next image.
            im, label = self.batch_loader.load_next_image()

            # Add directly to the caffe data layer
            top[0].data[itt, ...] = im
            top[1].data[itt, ...] = label

    def reshape(self, bottom, top):
        """
        There is no need to reshape the data, since the input is of fixed size
        (rows and columns)
        """
        pass

    def backward(self, top, propagate_down, bottom):
        """
        These layers does not back propagate
        """
        pass


class BatchLoader(object):

    """
    This class abstracts away the loading of images.
    Images can either be loaded singly, or in a batch. The latter is used for
    the asyncronous data layer to preload batches while other processing is
    performed.
    """

    def __init__(self, params):
        self.batch_size = params['batch_size']
        self.root_folder = params['root_folder']
        self.source = params['source']
        # get list of image indexes.
        self.indexlist = [line.strip().split() for line in open(self.source)]
        self._cur = 0  # current image
        # this class does some simple data-manipulations
        self.transformer = SimpleTransformer(params)

        print "BatchLoader initialized with {} images".format(
            len(self.indexlist))

    def load_next_image(self):
        """
        Load the next image in a batch.
        """
        # Did we finish an epoch?
        if self._cur == len(self.indexlist):
            self._cur = 0
            shuffle(self.indexlist)

        # Load an image
        index = self.indexlist[self._cur]  # Get the image index
        image_file_path, label_file_path = index
        # image = Image.open(osp.join(self.root_folder, image_file_path))
        # label = Image.open(osp.join(self.root_folder, label_file_path))
        image = cv2.imread(self.root_folder+image_file_path, cv2.IMREAD_COLOR)
        label = cv2.imread(self.root_folder+label_file_path, cv2.IMREAD_GRAYSCALE)
        self._cur += 1
        return self.transformer.preprocess(image, label)


class SimpleTransformer:

    """
    SimpleTransformer is a simple class for preprocessing and deprocessing
    images for caffe.
    """

    def __init__(self, params):
        SimpleTransformer.check_params(params)
        self.mean = params['mean']
        self.is_mirror = params['mirror']
        self.crop_h, self.crop_w = params['crop_size']
        self.scale = params['scale']
        self.phase = params['phase']
        self.ignore_label = params['ignore_label']

    def set_mean(self, mean):
        """
        Set the mean to subtract for centering the data.
        """
        self.mean = mean

    def set_scale(self, scale):
        """
        Set the data scaling.
        """
        self.scale = scale

    def pre_test_image(self, image):
        image = np.asarray(image, np.float32)
        image = image[:, :, [2, 1, 0]]
        image -= self.mean
        img_h, img_w, _ = image.shape
        pad_h = max(self.crop_h - img_h, 0)
        pad_w = max(self.crop_w - img_w, 0)
        if pad_h > 0 or pad_w > 0:
            img_pad = cv2.copyMakeBorder(image, 0, pad_h, 0, 
                pad_w, cv2.BORDER_CONSTANT, 
                value=(0.0,0.0,0.0))
        else:
            img_pad = image
        img_h, img_w, _ = img_pad.shape
        
        h_off = (img_h - self.crop_h) / 2
        w_off = (img_w - self.crop_w) / 2
        image = np.asarray(img_pad[h_off : h_off+self.crop_h, w_off : w_off+self.crop_w], np.float32)
        #image = image[:, :, ::-1]  # change to BGR
        image = image.transpose((2, 0, 1))
        return image

    def preprocess(self, image, label=None):
        """
        preprocess() emulate the pre-processing occuring in the vgg16 caffe
        prototxt.
        """
        # image = cv2.convertTo(image, cv2.CV_64F)
        image = np.asarray(image, np.float32)
        image -= self.mean
        image *= self.scale
        
        if label is None:
            img_h, img_w, _ = image.shape
            pad_h = max(self.crop_h - img_h, 0)
            pad_w = max(self.crop_w - img_w, 0)
            if pad_h > 0 or pad_w > 0:
                img_pad = cv2.copyMakeBorder(image, 0, pad_h, 0, 
                    pad_w, cv2.BORDER_CONSTANT, 
                    value=(0.0,0.0,0.0))
            else:
                img_pad = image

            img_h, img_w, _ = img_pad.shape
            
            h_off = (img_h - self.crop_h) / 2
            w_off = (img_w - self.crop_w) / 2
            image = np.asarray(img_pad[h_off : h_off+self.crop_h, w_off : w_off+self.crop_w], np.float32)

            #image = image[:, :, ::-1]  # change to BGR
            image = image.transpose((2, 0, 1))

            return image


        img_h, img_w = label.shape
        pad_h = max(self.crop_h - img_h, 0)
        pad_w = max(self.crop_w - img_w, 0)
        if pad_h > 0 or pad_w > 0:
            img_pad = cv2.copyMakeBorder(image, 0, pad_h, 0, 
                pad_w, cv2.BORDER_CONSTANT, 
                value=(0.0, 0.0, 0.0))
            label_pad = cv2.copyMakeBorder(label, 0, pad_h, 0, 
                pad_w, cv2.BORDER_CONSTANT,
                value=(self.ignore_label,))
        else:
            img_pad, label_pad = image, label

        img_h, img_w = label_pad.shape
        if self.phase == 'Train':
            h_off = random.randint(0, img_h - self.crop_h)
            w_off = random.randint(0, img_w - self.crop_w)
        else:
            h_off = (img_h - self.crop_h) / 2
            w_off = (img_w - self.crop_w) / 2

        # roi = cv2.Rect(w_off, h_off, self.crop_w, self.crop_h);

        image = np.asarray(img_pad[h_off : h_off+self.crop_h, w_off : w_off+self.crop_w], np.float32)
        label = np.asarray(label_pad[h_off : h_off+self.crop_h, w_off : w_off+self.crop_w], np.float32)

        #image = image[:, :, ::-1]  # change to BGR
        image = image.transpose((2, 0, 1))

        if self.is_mirror:
            flip = np.random.choice(2) * 2 - 1
            image = image[:, :, ::flip]
            label = label[:, ::flip]

        return image, label

    @classmethod
    def check_params(cls, params):
        if 'crop_size' not in params:
            params['crop_size'] = (505, 505)
        if 'mean' not in params:
            params['mean'] = [128, 128, 128]
        if 'scale' not in params:
            params['scale'] = 1.0
        if 'mirror' not in params:
            params['mirror'] = False
        if 'phase' not in params:
            params['phase'] = 'Train'
        if 'ignore_label' not in params:
            params['ignore_label'] = 255

def print_info(name, params):
    """
    Ouput some info regarding the class
    """
    print "{} initialized for split: {}, with bs: {}, im_shape: {}.".format(
        name,
        params['source'],
        params['batch_size'],
        params['crop_size'])


if __name__ == '__main__':
    params = {'batch_size': 2,
              'mean': (104.008, 116.669, 122.675),
              'root_folder': 'D:/v-zihuan/segmentation_with_scale/experiment/voc_part/data/',
              'source': 'D:/v-zihuan/segmentation_with_scale/experiment/voc_part/list/train_3s.txt',
              'mirror': True,
              'crop_size': (505, 505)}
    t = SimpleTransformer(params)

    image = Image.open(r'D:/v-zihuan/segmentation_with_scale/experiment/voc_part/data/images/2008_000003.jpg')
    label = Image.open(r'D:/v-zihuan/segmentation_with_scale/experiment/voc_part/data/part_mask_scale_3/2008_000003.png')
    t.preprocess(image, label)
