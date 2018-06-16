import caffe

import numpy as np
from scipy.ndimage import zoom

import theano
import theano.tensor as T

import cPickle
import yaml
import cv2
import os.path as osp
import multiprocessing
from numpy.random import shuffle

from krahenbuhl2013 import CRF
import CC_labeling_8
from sklearn.cluster import KMeans

min_prob = 0.0001
R = 5

class SoftmaxLayer(caffe.Layer):

    def setup(self, bottom, top):

        if len(bottom) != 1:
            raise Exception("Need two inputs to compute distance.")

        preds = T.ftensor4()
        top_diff = T.ftensor4()

        preds_max = T.addbroadcast(T.max(preds, axis=1, keepdims=True), 1)
        preds_exp = np.exp(preds - preds_max)
        probs = preds_exp / T.addbroadcast(T.sum(preds_exp, axis=1, keepdims=True), 1) + min_prob
        probs = probs / T.sum(probs, axis=1, keepdims=True)

        probs_sum = T.sum(probs * top_diff)

        self.forward_theano = theano.function([preds], probs)
        self.backward_theano = theano.function([preds, top_diff], T.grad(probs_sum, preds))

    def reshape(self, bottom, top):
        top[0].reshape(*bottom[0].data.shape)

    def forward(self, bottom, top):
        top[0].data[...] = self.forward_theano(bottom[0].data[...])

    def backward(self, top, prop_down, bottom):
        grad = self.backward_theano(bottom[0].data[...], top[0].diff[...])
        bottom[0].diff[...] = grad


class CRFLayer(caffe.Layer):

    def setup(self, bottom, top):
        if len(bottom) != 2:
            raise Exception("The layer needs two inputs!")

    def reshape(self, bottom, top):
        top[0].reshape(*bottom[0].data.shape)

    def forward(self, bottom, top):

        probs = bottom[0].data
        _, _, h, w = probs.shape
        probs[probs < min_prob] = min_prob
        unary = np.transpose(np.array(probs), [0, 2, 3, 1])

        mean_pixel = np.array([104.0, 117.0, 123.0])
        im = bottom[1].data[...]
        im = zoom(im, (1.0, 1.0, float(h) / im.shape[2], float(w) / im.shape[3]), order=1)
        im = np.transpose(im, [0, 2, 3, 1])
        im = im + mean_pixel[None, None, None, :]
        im = np.round(im)

        N = unary.shape[0]

        self.result = np.zeros(unary.shape)

        for i in range(N):
            self.result[i] = CRF(im[i], unary[i], scale_factor=12.0)

        self.result = np.transpose(self.result, [0, 3, 1, 2])
        self.result[self.result < min_prob] = min_prob
        self.result = self.result / np.sum(self.result, axis=1, keepdims=True)

        top[0].data[...] = np.log(self.result)

    def backward(self, top, prop_down, bottom):
        grad = (1 - self.result) * top[0].diff[...]
        bottom[0].diff[...] = grad


class SeedLossLayer(caffe.Layer):

    def setup(self, bottom, top):
        if len(bottom) != 2:
            raise Exception("The layer needs two inputs!")

        probs = T.ftensor4()
        labels = T.ftensor4()

        count = T.sum(labels, axis=(1, 2, 3), keepdims=True)
        loss_balanced = -T.mean(T.sum(labels * T.log(probs), axis=(1, 2, 3), keepdims=True) / count)

        self.forward_theano = theano.function([probs, labels], loss_balanced)
        self.backward_theano = theano.function([probs, labels], T.grad(loss_balanced, probs))

    def reshape(self, bottom, top):
        top[0].reshape(1)

    def forward(self, bottom, top):
        top[0].data[...] = self.forward_theano(bottom[0].data[...], bottom[1].data[...])

    def backward(self, top, prop_down, bottom):
        grad = self.backward_theano(bottom[0].data[...], bottom[1].data[...])
        bottom[0].diff[...] = grad

class BalancedSeedLossLayer(caffe.Layer):

    def setup(self, bottom, top):
        if len(bottom) != 2:
            raise Exception("The layer needs two inputs!")

        probs = T.ftensor4()
        labels = T.ftensor4()

        probs_bg = probs[:, 0, :, :]
        labels_bg = labels[:, 0, :, :]
        probs_fg = probs[:, 1:, :, :]
        labels_fg = labels[:, 1:, :, :]

        count_bg = T.sum(labels_bg, axis=(1, 2), keepdims=True)
        count_fg = T.sum(labels_fg, axis=(1, 2, 3), keepdims=True)
        loss_1 = -T.mean(T.sum(labels_bg * T.log(probs_bg), axis=(1, 2), keepdims=True) / T.maximum(count_bg, min_prob))
        loss_2 = -T.mean(T.sum(labels_fg * T.log(probs_fg), axis=(1, 2, 3), keepdims=True) / T.maximum(count_fg, min_prob))

        loss_balanced = loss_1 + loss_2

        self.forward_theano = theano.function([probs, labels], loss_balanced)
        self.backward_theano = theano.function([probs, labels], T.grad(loss_balanced, probs))

    def reshape(self, bottom, top):
        top[0].reshape(1)

    def forward(self, bottom, top):
        top[0].data[...] = self.forward_theano(bottom[0].data[...], bottom[1].data[...])

    def backward(self, top, prop_down, bottom):
        grad = self.backward_theano(bottom[0].data[...], bottom[1].data[...])
        bottom[0].diff[...] = grad

class ConstrainLossLayer(caffe.Layer):

    def setup(self, bottom, top):
        if len(bottom) != 2:
            raise Exception("The layer needs two inputs!")

        probs = T.ftensor4()
        probs_smooth_log = T.ftensor4()

        probs_smooth = T.exp(probs_smooth_log)

        loss = T.mean(T.sum(probs_smooth * T.log(T.clip(probs_smooth / probs, 0.05, 20)), axis=1)) #

        self.forward_theano = theano.function([probs, probs_smooth_log], loss)
        self.backward_theano = theano.function([probs, probs_smooth_log], T.grad(loss, [probs, probs_smooth_log]))

    def reshape(self, bottom, top):
        top[0].reshape(1)

    def forward(self, bottom, top):
        top[0].data[...] = self.forward_theano(bottom[0].data[...], bottom[1].data[...])

    def backward(self, top, prop_down, bottom):
        grad = self.backward_theano(bottom[0].data[...], bottom[1].data[...])[0]
        bottom[0].diff[...] = grad
        grad = self.backward_theano(bottom[0].data[...], bottom[1].data[...])[1]
        bottom[1].diff[...] = grad


class ExpandLossLayer(caffe.Layer):

    def setup(self, bottom, top):

        if len(bottom) != 2:
            raise Exception("The layer needs two inputs!")

        probs_tmp = T.ftensor4()
        stat_inp = T.ftensor4()

        stat = stat_inp[:, :, :, 1:]

        probs_bg = probs_tmp[:, 0, :, :]
        probs = probs_tmp[:, 1:, :, :]

        probs_max = T.max(probs, axis=3).max(axis=2)

        q_fg = 0.996
        probs_sort = T.sort(probs.reshape((-1, 20, 41 * 41)), axis=2)
        weights = np.array([q_fg ** i for i in range(41 * 41 - 1, -1, -1)])[None, None, :]
        Z_fg = np.sum(weights)
        weights = T.addbroadcast(theano.shared(weights), 0, 1)
        probs_mean = T.sum((probs_sort * weights) / Z_fg, axis=2)

        q_bg = 0.999
        probs_bg_sort = T.sort(probs_bg.reshape((-1, 41 * 41)), axis=1)
        weights_bg = np.array([q_bg ** i for i in range(41 * 41 - 1, -1, -1)])[None, :]
        Z_bg = np.sum(weights_bg)
        weights_bg = T.addbroadcast(theano.shared(weights_bg), 0)
        probs_bg_mean = T.sum((probs_bg_sort * weights_bg) / Z_bg, axis=1)

        stat_2d = stat[:, 0, 0, :] > 0.5

        loss_1 = -T.mean(T.sum((stat_2d * T.log(probs_mean) / T.sum(stat_2d, axis=1, keepdims=True)), axis=1))
        loss_2 = -T.mean(T.sum(((1 - stat_2d) * T.log(1 - probs_max) / T.sum(1 - stat_2d, axis=1, keepdims=True)), axis=1))
        loss_3 = -T.mean(T.log(probs_bg_mean))

        loss = loss_1 + loss_2 + loss_3

        self.forward_theano = theano.function([probs_tmp, stat_inp], loss)
        self.backward_theano = theano.function([probs_tmp, stat_inp], T.grad(loss, probs_tmp))

    def reshape(self, bottom, top):
        top[0].reshape(1)

    def forward(self, bottom, top):
        top[0].data[...] = self.forward_theano(bottom[0].data[...], bottom[1].data[...])

    def backward(self, top, prop_down, bottom):
        grad = self.backward_theano(bottom[0].data[...], bottom[1].data[...])
        bottom[0].diff[...] = grad

import timeit
from multiprocessing import Pool
def generate_seed_step(item):
    labels, seed_c, probs_refinement, th1, th2 = item

    cls_index = np.where(labels == 1)[0]
    probs_selected = probs_refinement[cls_index]
    probs_c = np.argmax(probs_selected, axis=0)
    probs_p = np.max(probs_selected, axis=0)

    channels, height, width = seed_c.shape
    label_map = np.zeros((height, width))

    index1 = np.where(seed_c > 0)

    label_map[index1[1], index1[2]] = index1[0] + 1 # 1-index
    for (x,y), value in np.ndenumerate(probs_p):
        c = cls_index[probs_c[x,y]]
        if value > th2:
            if not c == 0:
                label_map[x, y] = c + 1
            elif value > th1:
                label_map[x, y] = c + 1

    for c in cls_index:
        mat = (label_map == (c+1))
        mat = mat.astype(int)
        cclab = CC_labeling_8.CC_lab(mat)
        cclab.connectedComponentLabel()
        high_confidence_set_label = set()
        for (x,y), value in np.ndenumerate(mat):
            if value == 1 and seed_c[c, x, y] == 1:
                high_confidence_set_label.add(cclab.labels[x][y])
            elif value == 1 and np.sum(seed_c[:, x, y]) == 1:
                cclab.labels[x][y] = -1

        for (x,y), value in np.ndenumerate(np.array(cclab.labels)):
            if value in high_confidence_set_label:
                seed_c[c, x, y] = 1

    return seed_c

class DSRGLayer(caffe.Layer):

    def setup(self, bottom, top):
        if len(bottom) != 4:
            raise Exception("The layer needs four inputs!")

        # parse the layer parameter string, which must be valid YAML
        layer_params = yaml.load(self.param_str)
        # self._do_save = layer_params['do_save']
        self._th1 = layer_params['th1']
        self._th2 = layer_params['th2']
        if 'iters' not in layer_params:
            layer_params['iters'] = -1
        self._max_iters = layer_params['iters']
        self._iter_index = 0
        self.pool = Pool()

    def reshape(self, bottom, top):
        top[0].reshape(*bottom[1].data.shape)

    def forward(self, bottom, top):
        img_labels, probs, cues, im = bottom[0].data, bottom[1].data, bottom[2].data, bottom[3].data
        num, channels, height, width = probs.shape

        seed_c = self.generate_seed(img_labels, probs, cues, im)
        
        self._iter_index = self._iter_index + 1
        top[0].data[...] = seed_c


    def backward(self, top, prop_down, bottom): 
        bottom[1].diff[...] = top[0].diff

    def refinement(self, probs, im, scale_factor=12.0):
        _, _, h, w = probs.shape
        probs[probs < min_prob] = min_prob
        unary = np.transpose(np.array(probs), [0, 2, 3, 1])

        mean_pixel = np.array([104.0, 117.0, 123.0])
        im = zoom(im, (1.0, 1.0, float(h) / im.shape[2], float(w) / im.shape[3]), order=1)
        im = np.transpose(im, [0, 2, 3, 1])
        im = im + mean_pixel[None, None, None, :]
        im = np.round(im)

        N = unary.shape[0]

        result = np.zeros(unary.shape)

        for i in range(N):
            result[i] = CRF(im[i], unary[i], scale_factor=scale_factor)

        result = np.transpose(result, [0, 3, 1, 2])
        result[result < min_prob] = min_prob
        result = result / np.sum(result, axis=1, keepdims=True)
        return result

    def generate_seed(self, labels, probs, cues, im):
        num, channels, height, width = probs.shape
        probs_refinement = self.refinement(probs, im, 12.0)
        # probs_refinement = np.exp(probs)

        seed_c = np.zeros_like(probs)
        seed_c[...] = cues

        items_for_map = [[labels[batch_id, 0, 0], seed_c[batch_id], probs_refinement[batch_id], self._th1, self._th2] for batch_id in xrange(num)]
        seed_c_all = self.pool.map(generate_seed_step, items_for_map)

        return np.array(seed_c_all)

class AnnotationLayer(caffe.Layer):

    def setup(self, bottom, top):
        if len(bottom) != 2:
            raise Exception("The layer needs two inputs!")

        layer_params = yaml.load(self.param_str)
        if 'cues' not in layer_params:
            layer_params['cues'] = 'localization_cues.pickle'
        self._cue_name = layer_params['cues']

        if not 'mirror' in layer_params:
            layer_params['mirror'] = False
        self.is_mirror = layer_params['mirror']

        this_dir = osp.dirname(__file__)
        self.data_file = cPickle.load(open(osp.join(this_dir, '../../training', 'localization_cues', self._cue_name)))

    def reshape(self, bottom, top):
        top[0].reshape(bottom[0].data.shape[0], 1, 1, 21)
        top[1].reshape(bottom[0].data.shape[0], 21, 41, 41)
        top[2].reshape(*bottom[1].data.shape)

    def forward(self, bottom, top):

        top[0].data[...] = 0.0
        top[1].data[...] = 0.0
        top[2].data[...] = bottom[1].data

        for i, image_id in enumerate(bottom[0].data[...]):

            labels_i = self.data_file['%i_labels' % image_id]
            top[0].data[i, 0, 0, 0] = 1.0
            top[0].data[i, 0, 0, labels_i] = 1.0

            cues_i = self.data_file['%i_cues' % image_id]
            top[1].data[i, cues_i[0], cues_i[1], cues_i[2]] = 1.0

            if self.is_mirror:
                flip = np.random.choice(2) * 2 - 1
                top[1].data[i, ...] = top[1].data[i, :, :, ::flip]
                top[2].data[i, ...] = top[2].data[i, :, :, ::flip]

class AnnotationLayerCOCO(caffe.Layer):

    def setup(self, bottom, top):
        layer_params = eval(self.param_str)
        self.source = layer_params['source']
        self.root_folder = layer_params['root']
         # store input as class variables
        self.batch_size = layer_params['batch_size']

        if not 'mirror' in layer_params:
            self.is_mirror = False
        else:
            self.is_mirror = layer_params['mirror']

        self.mean = layer_params['mean']
        self.new_h, self.new_w = layer_params['new_size']
        if not 'ignore_label' in layer_params:
            self.ignore_label = 255
        else:
            self.ignore_label = layer_params['ignore_label']

        self.indexlist = [line.strip().split() for line in open(self.source)]
        self._cur = 0  # current image
        self.q = multiprocessing.Queue(maxsize=self.batch_size*2)
        # self.start_batch()

        top[0].reshape(self.batch_size, 1, 1, 81)
        top[1].reshape(self.batch_size, 81, self.new_h / 8 + 1, self.new_w / 8 + 1)
        top[2].reshape(self.batch_size, 3, self.new_h, self.new_w)

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

    def forward(self, bottom, top):
        for itt in xrange(self.batch_size):
            # Use the batch loader to load the next image.
            im, label, image_label = self.load_next_image()

            # Add directly to the caffe data layer
            top[0].data[itt, ...] = image_label
            top[1].data[itt, ...] = label
            top[2].data[itt, ...] = im

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
        return self.preprocess(image, label)

    def perpare_next_data(self):
        """
        Load the next image in a batch.
        """
        return self.q.get()

    def start_batch(self):
        thread = multiprocessing.Process(target=self.data_generator_task)
        thread.daemon = True
        thread.start()

    def data_generator_task(self):
        while True:
            output = self.load_next_image()
            self.q.put(output)

    def preprocess(self, image, label):
        """
        preprocess() emulate the pre-processing occuring in the vgg16 caffe
        prototxt.
        """
        # image = cv2.convertTo(image, cv2.CV_64F)
        image = np.array(image)
        image = zoom(image.astype('float32'),
                        (self.new_h / float(image.shape[0]),
                        self.new_w / float(image.shape[1]), 1.0),
                        order=1)

        image = image[:, :, [2, 1, 0]]
        image = image - self.mean

        image = image.transpose([2, 0, 1])

        h, w = label.shape
        cues = np.zeros((81, h, w), dtype=np.uint8)
        for (x,y), v in np.ndenumerate(label):
            if not v == self.ignore_label:
                cues[v, x, y] = 1

        if self.is_mirror:
            flip = np.random.choice(2) * 2 - 1
            image = image[:, :, ::flip]
            cues = cues[:, :, ::flip]

        unique_inst = np.unique(label)
        ignore_ind = np.where(unique_inst == self.ignore_label)[0]
        unique_inst = np.delete(unique_inst, ignore_ind)
        image_label = np.zeros((1, 1, 81))
        for cat_id in unique_inst:
            image_label[0, 0, cat_id] = 1

        return image, cues, image_label
