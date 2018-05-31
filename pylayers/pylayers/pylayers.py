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
from random_walker import random_walker

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


class SeedLossLayer2(caffe.Layer):

    def setup(self, bottom, top):
        if len(bottom) != 3:
            raise Exception("The layer needs three inputs!")

        probs = T.ftensor4()
        labels = T.ftensor4()
        punishment = T.ftensor4()

        count = T.sum(labels, axis=(1, 2, 3), keepdims=True)
        count_ps = T.sum(punishment, axis=(1, 2, 3), keepdims=True)
        loss_1 = -T.mean(T.sum(labels * T.log(probs), axis=(1, 2, 3), keepdims=True) / count)
        loss_2 = -T.mean(T.sum(punishment * T.log(1-probs), axis=(1, 2, 3), keepdims=True) / count_ps)
        loss_balanced = loss_1 + loss_2

        self.forward_theano = theano.function([probs, labels, punishment], loss_balanced)
        self.backward_theano = theano.function([probs, labels, punishment], T.grad(loss_balanced, probs))

    def reshape(self, bottom, top):
        top[0].reshape(1)

    def generate_punishment(self, probs, image_labels):
        num, channels, height, width = probs.shape
        punishment = np.zeros_like(probs)
        for batch_id in xrange(num):
            unpresent_indexs = np.where(image_labels[batch_id, 0, 0, :] == 0)[0]
            selected_probs = probs[batch_id, unpresent_indexs, ...]
            probs_c = np.argmax(selected_probs, axis=0)
            probs_c = np.vectorize(lambda x: unpresent_indexs[x])(probs_c)
            # for (x,y), value in np.ndenumerate(probs_c):
            #     probs_c[x,y] = unpresent_indexs[value]
            z, t = np.indices(probs_c.shape)
            punishment[batch_id, probs_c, z, t] = 1.0
            
        # punishment = punishment * (1.0 - image_labels[:,0,0,:][:, :, np.newaxis, np.newaxis])
        # punishment[probs < 0.01] = 0
        # print image_labels[:,0,0,:]
        # print np.sum(punishment, axis=(2,3))
        return punishment

    def forward(self, bottom, top):
        self.punishment = self.generate_punishment(bottom[0].data[...], bottom[2].data[...])
        top[0].data[...] = self.forward_theano(bottom[0].data[...], bottom[1].data[...], self.punishment)

    def backward(self, top, prop_down, bottom):
        grad = self.backward_theano(bottom[0].data[...], bottom[1].data[...], self.punishment)
        bottom[0].diff[...] = grad


class SeedLossLayer3(caffe.Layer):

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


class SeedLossLayer4(caffe.Layer):

    def setup(self, bottom, top):
        if len(bottom) != 3:
            raise Exception("The layer needs three inputs!")

        probs = T.ftensor4()
        labels = T.ftensor4()
        punishment = T.ftensor4()

        probs_bg = probs[:, 0, :, :]
        labels_bg = labels[:, 0, :, :]
        probs_fg = probs[:, 1:, :, :]
        labels_fg = labels[:, 1:, :, :]

        count_bg = T.sum(labels_bg, axis=(1, 2), keepdims=True)
        count_fg = T.sum(labels_fg, axis=(1, 2, 3), keepdims=True)
        count_ps = T.sum(punishment, axis=(1, 2, 3), keepdims=True)
        loss_1 = -T.mean(T.sum(labels_bg * T.log(probs_bg), axis=(1, 2), keepdims=True) / count_bg)
        loss_2 = -T.mean(T.sum(labels_fg * T.log(probs_fg), axis=(1, 2, 3), keepdims=True) / count_fg)
        loss_3 = -T.mean(T.sum(punishment * T.log(1-probs), axis=(1, 2, 3), keepdims=True) / count_ps)

        loss_balanced = loss_1 + loss_2 + loss_3

        self.forward_theano = theano.function([probs, labels, punishment], loss_balanced)
        self.backward_theano = theano.function([probs, labels, punishment], T.grad(loss_balanced, probs))

    def reshape(self, bottom, top):
        top[0].reshape(1)

    def generate_punishment(self, probs, image_labels):
        num, channels, height, width = probs.shape
        punishment = np.zeros_like(probs)
        for batch_id in xrange(num):
            unpresent_indexs = np.where(image_labels[batch_id, 0, 0, :] == 0)[0]
            selected_probs = probs[batch_id, unpresent_indexs, ...]
            probs_c = np.argmax(selected_probs, axis=0)
            probs_c = np.vectorize(lambda x: unpresent_indexs[x])(probs_c)
            # for (x,y), value in np.ndenumerate(probs_c):
            #     probs_c[x,y] = unpresent_indexs[value]
            z, t = np.indices(probs_c.shape)
            punishment[batch_id, probs_c, z, t] = 1.0
            
        # punishment = punishment * (1.0 - image_labels[:,0,0,:][:, :, np.newaxis, np.newaxis])
        # punishment[probs < 0.01] = 0
        # print image_labels[:,0,0,:]
        # print np.sum(punishment, axis=(1,2,3))
        return punishment

    def forward(self, bottom, top):
        self.punishment = self.generate_punishment(bottom[0].data[...], bottom[2].data[...])
        top[0].data[...] = self.forward_theano(bottom[0].data[...], bottom[1].data[...], self.punishment)

    def backward(self, top, prop_down, bottom):
        grad = self.backward_theano(bottom[0].data[...], bottom[1].data[...], self.punishment)
        bottom[0].diff[...] = grad


class SigmoidCrossEntropyLossLayer(caffe.Layer):

    def setup(self, bottom, top):
        if len(bottom) != 2:
            raise Exception("The layer needs two inputs!")

        scores = T.ftensor4()
        labels = T.ftensor4()

        probs = T.sum(T.nnet.nnet.sigmoid(scores), axis=(2, 3)) 
        stat_2d = labels[:, 0, 0, :] > 0.5

        loss_1 = -T.mean(T.sum((stat_2d * T.log(probs) / T.sum(stat_2d, axis=1, keepdims=True)), axis=1))
        loss_2 = -T.mean(T.sum(((1-stat_2d) * T.log(1-probs) / T.sum(1-stat_2d, axis=1, keepdims=True)), axis=1))
        loss = loss_1 + loss_2

        self.forward_theano = theano.function([scores, labels], loss)
        self.backward_theano = theano.function([scores, labels], T.grad(loss, scores))

    def reshape(self, bottom, top):
        top[0].reshape(1)

    def forward(self, bottom, top):
        top[0].data[...] = self.forward_theano(bottom[0].data[...], bottom[1].data[...])

    def backward(self, top, prop_down, bottom):
        grad = self.backward_theano(bottom[0].data[...], bottom[1].data[...])
        bottom[0].diff[...] = grad

class CrossEntropyLossLayer(caffe.Layer):

    def setup(self, bottom, top):
        if len(bottom) != 2:
            raise Exception("The layer needs two inputs!")

        scores = T.ftensor4()
        labels = T.ftensor4()

        probs = T.sum(scores, axis=(2, 3)) 
        stat_2d = labels[:, 0, 0, :] > 0.5

        loss_1 = -T.mean(T.sum((stat_2d * T.log(probs) / T.sum(stat_2d, axis=1, keepdims=True)), axis=1))
        loss_2 = -T.mean(T.sum(((1-stat_2d) * T.log(1-probs) / T.sum(1-stat_2d, axis=1, keepdims=True)), axis=1))
        loss = loss_1 + loss_2

        self.forward_theano = theano.function([scores, labels], loss)
        self.backward_theano = theano.function([scores, labels], T.grad(loss, scores))

    def reshape(self, bottom, top):
        top[0].reshape(1)

    def forward(self, bottom, top):
        top[0].data[...] = self.forward_theano(bottom[0].data[...], bottom[1].data[...])

    def backward(self, top, prop_down, bottom):
        grad = self.backward_theano(bottom[0].data[...], bottom[1].data[...])
        bottom[0].diff[...] = grad

class LSEPoolingWithLossLayer(caffe.Layer):

    def setup(self, bottom, top):
        if len(bottom) != 2:
            raise Exception("The layer needs two inputs!")

        self._r = 5.0

        probs = T.ftensor4()
        labels = T.ftensor4()

        stat_2d = labels[:, 0, 0, :] > 0.5

        S_probs = T.log(T.mean(T.exp(self._r*probs), axis=(2,3))) / self._r

        loss_1 = -T.mean(T.sum((stat_2d * T.log(S_probs) / T.sum(stat_2d, axis=1, keepdims=True)), axis=1))
        loss_2 = -T.mean(T.sum(((1-stat_2d) * T.log(1-S_probs) / T.sum(1-stat_2d, axis=1, keepdims=True)), axis=1))
        loss = loss_1 + loss_2

        self.forward_theano = theano.function([probs, labels], loss)
        self.backward_theano = theano.function([probs, labels], T.grad(loss, probs))

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


class PSLayer(caffe.Layer):
    def setup(self, bottom, top):
        if len(bottom) != 2:
            raise Exception("The layer needs two inputs!")

    def reshape(self, bottom, top):
        num, channels, height, width = bottom[1].data.shape
        top[0].reshape(num, 1, height, width)

    def forward(self, bottom, top):
        num, channels, height, width = bottom[1].data.shape
        scores = bottom[0].data
        segs = bottom[1].data
        scores_full = np.ones((num, channels), dtype=np.float32)
        scores_full[:, 1:] = scores
        weighted_seg = segs * scores_full[:, :, None, None]
        top[0].data[...] = np.argmax(weighted_seg, axis=1).reshape(*top[0].data.shape)

    def backward(self, top, prop_down, bottom):
        pass


class MaskLayer(caffe.Layer):

    def setup(self, bottom, top):
        if len(bottom) != 4:
            raise Exception("The layer needs four inputs!")

        probs = T.ftensor4()
        mask = T.ftensor4()
        stat_inp = T.ftensor4()

        stat = stat_inp[:, :, :, 1:]
        
        probs_fg = probs[:, 1:, :, :]
        mask_fg = mask[:, 1:, :, :]
        probs_bg = probs[:, 0, :, :]

        probs_all_max = T.max(probs_fg, axis=3).max(axis=2)
        Z = T.sum(mask, axis=(2, 3))
        probs_mean = T.sum(probs * mask, axis=(2, 3)) / Z
        probs_mean_bg = probs_mean[:, 0]
        probs_mean_ob = probs_mean[:, 1:]

        stat_2d = stat[:, 0, 0, :] > 0.5

        loss_1 = -T.mean(T.sum((stat_2d * T.log(probs_mean_ob) / T.sum(stat_2d, axis=1, keepdims=True)), axis=1))
        loss_2 = -T.mean(T.sum(((1 - stat_2d) * T.log(1 - probs_all_max) / T.sum(1 - stat_2d, axis=1, keepdims=True)), axis=1))
        loss_3 = -T.mean(T.log(probs_mean_bg))
        # loss_2 = -T.mean(T.sum(((1 - stat_2d) * T.log(1 - probs_all_max) / T.sum(1 - stat_2d, axis=1, keepdims=True)), axis=1))
        # loss_3 = -T.mean(T.log(probs_bg_mean))

        loss = loss_1 + loss_2 + loss_3

        self.forward_theano = theano.function([probs, mask, stat_inp], loss)
        self.backward_theano = theano.function([probs, mask, stat_inp], T.grad(loss, probs))

    def reshape(self, bottom, top):
        top[0].reshape(1)

    def forward(self, bottom, top):
        self.mask = np.asarray(self.refinement(bottom[1].data[...], bottom[3].data[...], 12.0), dtype=np.float32)
        top[0].data[...] = self.forward_theano(bottom[0].data[...], self.mask, bottom[2].data[...])

    def backward(self, top, prop_down, bottom):
        grad = self.backward_theano(bottom[0].data[...], self.mask, bottom[2].data[...])
        bottom[0].diff[...] = grad
        # grad = self.backward_theano(bottom[0].data[...], bottom[1].data[...], bottom[2].data[...])[1]
        # bottom[1].diff[...] = grad
    def refinement(self, probs, im, scale_factor=12.0):
        probs[probs < min_prob] = min_prob
        unary = np.transpose(np.array(probs), [0, 2, 3, 1])

        mean_pixel = np.array([104.0, 117.0, 123.0])
        im = zoom(im, (1.0, 1.0, 41.0 / im.shape[2], 41.0 / im.shape[3]), order=1)
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

class UpdateSeedLayer2(caffe.Layer):

    def setup(self, bottom, top):
        if len(bottom) != 4:
            raise Exception("The layer needs four inputs!")

        # parse the layer parameter string, which must be valid YAML
        layer_params = yaml.load(self.param_str)
        # self._do_save = layer_params['do_save']
        self._th1 = layer_params['th1']
        self._th2 = layer_params['th2']
        if 'th3' not in layer_params:
            layer_params['th3'] = 0.5
        self._th3 = layer_params['th3']

    def reshape(self, bottom, top):
        top[0].reshape(*bottom[1].data.shape)

    def forward(self, bottom, top):
        img_labels, probs, cues = bottom[0].data, bottom[1].data, bottom[2].data
        num, channels, height, width = probs.shape

        markers, unique_labels = self.generate_mask_cues(img_labels, probs, cues)

        mean_pixel = np.array([104.0, 117.0, 123.0])
        im = bottom[3].data[...]
        im = zoom(im, (1.0, 1.0, 41.0 / im.shape[2], 41.0 / im.shape[3]), order=1)
        im = im + mean_pixel[None, :, None, None]
        im = np.transpose(np.round(im), [0, 2, 3, 1])

        seed_new = np.zeros_like(probs)
        seed_new[...] = cues
        for batch_id in xrange(num):

            prob = np.transpose(probs[batch_id], (1, 2, 0))
            labels = random_walker(im[batch_id], prob, markers[batch_id], alpha=200, beta=0.05, mode='bf', multichannel=True, return_full_prob=True)

            class_index = np.argmax(labels, axis=0)
            if not class_index.shape == (height, width):
                continue
            for (x,y), index in np.ndenumerate(class_index):
                if index >= len(unique_labels[batch_id]):
                    print np.max(markers[batch_id]), index, unique_labels[batch_id], sum(img_labels[batch_id, 0, 0]), labels.shape
                if labels[index, x, y] * probs[batch_id, unique_labels[batch_id][index], x, y] > self._th3:
                    seed_new[batch_id, unique_labels[batch_id][index], x, y] = 1.0


        top[0].data[...] = seed_new

        # seed_show = np.zeros((num, height, width))
        # seed_show.fill(21)
        # pos = np.where(seed_new == 1)
        # seed_show[pos[0], pos[2], pos[3]] = pos[1]
        # markers[markers == 0] = 22
        # markers = markers - 1
        # self.show_result(np.array(im[0], dtype=np.uint8), markers[0], seed_show[0])

    def backward(self, top, prop_down, bottom):
        pass

    def generate_mask_cues(self, labels, probs, cues):
        num, channels, height, width = probs.shape

        markers = np.zeros((num, height, width))
        unique_labels = []
        for batch_id in xrange(num):
            label = []
            for index, i in enumerate(labels[batch_id, 0, 0]):
                if i == 1 or index == 0:
                    label.append(index)
                    prob = probs[batch_id, index]
                    cue = cues[batch_id, index]
                    pos = np.where(cue == 1)
                    if len(pos[0]):
                        markers[batch_id, pos[0], pos[1]] = index + 1
                    else:
                        pos = np.unravel_index(prob.argmax(), prob.shape)
                        markers[batch_id, pos[0], pos[1]] = index + 1
            unique_labels.append(label)
        return markers, unique_labels

    def generate_mask(self, labels, probs):
        num, channels, height, width = probs.shape

        markers = np.zeros((num, height, width))
        unique_labels = []
        for batch_id in xrange(num):
            label = []
            for index, i in enumerate(labels[batch_id, 0, 0]):
                if i == 1 or index == 0:
                    label.append(index)
                    prob = probs[batch_id, index]
                    if index == 0:
                        cues = np.where(prob > self._th1)
                    else:
                        cues = np.where(prob > self._th2)
                    
                    if len(cues[0]):
                        markers[batch_id, cues[0], cues[1]] = index + 1
                    else:
                        pos = np.unravel_index(prob.argmax(), prob.shape)
                        markers[batch_id, pos[0], pos[1]] = index + 1
            unique_labels.append(label)
        return markers, unique_labels

    def show_result(self, img, seed_history, seed_new):
        from matplotlib import pyplot as plt
        from matplotlib import colors as mpl_colors
        palette = [(0.0, 0.0, 0.0), (0.5, 0.0, 0.0), (0.0, 0.5, 0.0), (0.5, 0.5, 0.0),
               (0.0, 0.0, 0.5), (0.5, 0.0, 0.5), (0.0, 0.5, 0.5), (0.5, 0.5, 0.5),
               (0.25, 0.0, 0.0), (0.75, 0.0, 0.0), (0.25, 0.5, 0.0), (0.75, 0.5, 0.0),
               (0.25, 0.0, 0.5), (0.75, 0.0, 0.5), (0.25, 0.5, 0.5), (0.75, 0.5, 0.5),
               (0.0, 0.25, 0.0), (0.5, 0.25, 0.0), (0.0, 0.75, 0.0), (0.5, 0.75, 0.0),
               (0.0, 0.25, 0.5), (1.0, 1.0, 1.0)]
        my_cmap = mpl_colors.LinearSegmentedColormap.from_list('Custom cmap', palette, 22)

        fig = plt.figure()

        ax = fig.add_subplot('221')
        ax.imshow(img)

        ax = fig.add_subplot('222')
        ax.matshow(seed_history, vmin=0, vmax=22, cmap=my_cmap)

        ax = fig.add_subplot('223')
        ax.matshow(seed_new, vmin=0, vmax=22, cmap=my_cmap)

        plt.show()

class UpdateSeedLayer(caffe.Layer):

    def setup(self, bottom, top):
        if len(bottom) != 3:
            raise Exception("The layer needs three inputs!")

        # parse the layer parameter string, which must be valid YAML
        layer_params = yaml.load(self.param_str)
        # self._do_save = layer_params['do_save']
        self._th1 = layer_params['th1']
        self._th2 = layer_params['th2']

    def reshape(self, bottom, top):
        num, channels, height, width = bottom[1].data.shape
        top[0].reshape(*bottom[1].data.shape)
        top[1].reshape(num, channels, height, width)

    def forward(self, bottom, top):
        img_labels, probs, im = bottom[0].data, bottom[1].data, bottom[2].data
        num, channels, height, width = probs.shape

        seed_c, seed_p = self.generate_seed(img_labels, probs, im)

        top[0].data[...] = seed_c
        top[1].data[...] = seed_p


    def backward(self, top, prop_down, bottom):
        pass

    def limit_background_count(self, probs_p, seed_p, labels):
        num, channels, height, width = seed_p.shape

        for batch_id in xrange(num):
            fg_count = np.sum(seed_p[batch_id, 1:, :, :]) / np.sum(labels[batch_id, 0, 0, 1:])
            bg_count = np.sum(seed_p[batch_id,  0, :, :])
            if bg_count <= fg_count:
                continue
            else:
                pos = np.where(seed_p[batch_id,  0] > 0)
                bg_list = [(probs_p[batch_id,x,y], (x,y)) for x,y in zip(pos[0], pos[1])]
                newlist = sorted(bg_list, key=lambda x: x[0])
                for i in xrange(bg_count - fg_count):
                    x, y = newlist[i][1]
                    seed_p[batch_id, 0, x, y] = 0
        # seed_p[batch_id, 0] = 0

    def refinement(self, probs, im, scale_factor=12.0):
        probs[probs < min_prob] = min_prob
        unary = np.transpose(np.array(probs), [0, 2, 3, 1])

        mean_pixel = np.array([104.0, 117.0, 123.0])
        im = zoom(im, (1.0, 1.0, 41.0 / im.shape[2], 41.0 / im.shape[3]), order=1)
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

    def generate_seed(self, labels, probs, im):
        num, channels, height, width = probs.shape

        seed_c = np.zeros_like(probs)
        seed_p = np.zeros_like(probs)

        for batch_id in xrange(num):
            probs_refinement = self.refinement(probs, im, 16.0)
            cls_index = np.where(labels[batch_id, 0, 0] == 1)[0]
            probs_selected = probs_refinement[batch_id, cls_index]
            probs_c = np.argmax(probs_selected, axis=0)
            probs_p = np.max(probs_selected, axis=0)

            for (x,y), value in np.ndenumerate(probs_c):
                seed_c[batch_id, cls_index[value], x, y] = 1
                seed_p[batch_id, cls_index[value], x, y] = probs_p[x,y]

        # for batch_id in xrange(num):
        #     for (x,y), c in np.ndenumerate(probs_c[batch_id]):
        #         if probs_p[batch_id,x,y] < self._th2 or (not c in labels[batch_id, 0, 0]):
        #             seed_c[batch_id, :, x, y] = 0
        #             seed_p[batch_id, :, x, y] = 0
        #         else:
        #             seed_c[batch_id, c, x, y] = 1
        #             if probs_p[batch_id, x, y] > self._th1:
        #                 seed_p[batch_id, c, x, y] = 1
        #             elif c == 1: 
        #                 seed_p[batch_id, c, x, y] = 0
        #             else:
        #                 seed_p[batch_id, c, x, y] = probs_p[batch_id,x,y]

        # self.limit_background_count(probs_p, seed_p, labels)
        return seed_c, seed_p

class FilterSegmentationLayer(caffe.Layer):

    def setup(self, bottom, top):
        if len(bottom) != 2:
            raise Exception("The layer needs two inputs!")
        self._r = 5.0

    def reshape(self, bottom, top):
        top[0].reshape(*bottom[0].data.shape)

    def forward(self, bottom, top):
        def sigmoid(x):
            return 1 / (1 + np.exp(-x))
        probs, scores = bottom[0].data, bottom[1].data
        num, channels, height, width = scores.shape
        # print num, channels, height, width

        s_score = np.log(np.mean(np.exp(self._r*scores), axis=(2,3), keepdims=True)) / self._r

        probs_filted = probs * s_score
        probs_filted_c = np.argmax(probs_filted, axis=1)
        seed_new = np.zeros_like(probs)

        x, z, t = np.indices(probs_filted_c.shape)
        seed_new[x, probs_filted_c, z, t] = 1.0
        # markers_new = np.zeros((num, height, width))
        # markers_new.fill(21)
        # pos = np.where(seed_c == 1)
        # markers_new[pos[0], pos[2], pos[3]] = pos[1]

        # markers_old = np.zeros((num, height, width))
        # markers_old.fill(21)
        # pos = np.where(cues == 1)
        # markers_old[pos[0], pos[2], pos[3]] = pos[1]

        # self.show_result(markers_old[0], markers_new[0])
        top[0].data[...] = seed_new

    def backward(self, top, prop_down, bottom): 
        pass

    def show_result(self, seed_history, seed_new):
        from matplotlib import pyplot as plt
        from matplotlib import colors as mpl_colors
        palette = [(0.0, 0.0, 0.0), (0.5, 0.0, 0.0), (0.0, 0.5, 0.0), (0.5, 0.5, 0.0),
               (0.0, 0.0, 0.5), (0.5, 0.0, 0.5), (0.0, 0.5, 0.5), (0.5, 0.5, 0.5),
               (0.25, 0.0, 0.0), (0.75, 0.0, 0.0), (0.25, 0.5, 0.0), (0.75, 0.5, 0.0),
               (0.25, 0.0, 0.5), (0.75, 0.0, 0.5), (0.25, 0.5, 0.5), (0.75, 0.5, 0.5),
               (0.0, 0.25, 0.0), (0.5, 0.25, 0.0), (0.0, 0.75, 0.0), (0.5, 0.75, 0.0),
               (0.0, 0.25, 0.5), (1.0, 1.0, 1.0)]
        my_cmap = mpl_colors.LinearSegmentedColormap.from_list('Custom cmap', palette, 22)

        fig = plt.figure()

        ax = fig.add_subplot('121')
        ax.set_title('old')
        ax.matshow(seed_history, vmin=0, vmax=22, cmap=my_cmap)

        ax = fig.add_subplot('122')
        ax.set_title('new')
        ax.matshow(seed_new, vmin=0, vmax=22, cmap=my_cmap)

        plt.show()

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

class UpdateSeedLayer3(caffe.Layer):

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
        # mean_pixel = np.array([104.0, 117.0, 123.0])
        # im = zoom(im, (1.0, 1.0, 41.0 / im.shape[2], 41.0 / im.shape[3]), order=1)
        # im = np.transpose(im, [0, 2, 3, 1])
        # im = im + mean_pixel[None, None, None, :]
        # im = np.round(im)

        # markers_new = np.zeros((num, height, width))
        # markers_new.fill(255)
        # pos = np.where(seed_c == 1)
        # markers_new[pos[0], pos[2], pos[3]] = pos[1]

        # markers_old = np.zeros((num, height, width))
        # markers_old.fill(255)
        # pos = np.where(cues == 1)
        # markers_old[pos[0], pos[2], pos[3]] = pos[1]

        # self.show_result(im[0], markers_old[0], markers_new[0])

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

    def label_propagation(self, probs_c, probs_p, seed_c, image_labels):
        channels, height, width = seed_c.shape
        label_map = np.zeros((height, width))
        result = np.zeros((height, width))

        cls_index = np.where(image_labels == 1)[0]
        index1 = np.where(seed_c > 0)
        index2 = np.where(probs_p > self._th2)

        label_map[index1[1], index1[2]] = index1[0] + 1 # 1-index
        for (x,y), value in np.ndenumerate(probs_p):
            c = cls_index[probs_c[x,y]]
            if value > self._th2:
                if not c == 0:
                    label_map[x, y] = c + 1
                elif value > 1:
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


    def generate_seed(self, labels, probs, cues, im):
        num, channels, height, width = probs.shape
        probs_refinement = self.refinement(probs, im, 12.0)
        # probs_refinement = np.exp(probs)

        seed_c = np.zeros_like(probs)
        seed_c[...] = cues

        items_for_map = [[labels[batch_id, 0, 0], seed_c[batch_id], probs_refinement[batch_id], self._th1, self._th2] for batch_id in xrange(num)]
        seed_c_all = self.pool.map(generate_seed_step, items_for_map)

        return np.array(seed_c_all)

    def show_result(self, img, seed_history, seed_new):
        from matplotlib import pyplot as plt
        from matplotlib import colors as mpl_colors
        palette = [(0.0, 0.0, 0.0), (0.5, 0.0, 0.0), (0.0, 0.5, 0.0), (0.5, 0.5, 0.0),
               (0.0, 0.0, 0.5), (0.5, 0.0, 0.5), (0.0, 0.5, 0.5), (0.5, 0.5, 0.5),
               (0.25, 0.0, 0.0), (0.75, 0.0, 0.0), (0.25, 0.5, 0.0), (0.75, 0.5, 0.0),
               (0.25, 0.0, 0.5), (0.75, 0.0, 0.5), (0.25, 0.5, 0.5), (0.75, 0.5, 0.5),
               (0.0, 0.25, 0.0), (0.5, 0.25, 0.0), (0.0, 0.75, 0.0), (0.5, 0.75, 0.0),
               (0.0, 0.25, 0.5)]
        my_cmap = mpl_colors.LinearSegmentedColormap.from_list('Custom cmap', palette, 21)

        fig = plt.figure()

        ax = fig.add_subplot('221')
        ax.set_title('image')
        ax.imshow(img)

        ax = fig.add_subplot('222')
        ax.set_title('old')
        ax.matshow(seed_history, vmin=0, vmax=21, cmap=my_cmap)

        ax = fig.add_subplot('223')
        ax.set_title('new')
        ax.matshow(seed_new, vmin=0, vmax=21, cmap=my_cmap)

        plt.show()


def generate_seed_step4(item):
    labels, seed_c, probs_refinement, th1, th2 = item

    cls_index = np.where(labels == 1)[0]
    probs_selected = probs_refinement[cls_index]
    probs_c = np.argmax(probs_selected, axis=0)
    probs_p = np.max(probs_selected, axis=0)

    channels, height, width = seed_c.shape
    label_map = np.zeros((height, width))
    CAM_map = np.zeros((height, width))

    index1 = np.where(seed_c > 0)

    label_map[index1[1], index1[2]] = index1[0] + 1 # 1-index; 0 stands for No Man's district
    CAM_map[index1[1], index1[2]] = index1[0] + 1

    for (x,y), value in np.ndenumerate(probs_p):
        c = cls_index[probs_c[x,y]]
        if value > th2:
            if not c == 0:
                label_map[x, y] = c + 1
            elif value > th1:
                label_map[x, y] = c + 1

    for c in cls_index:
        CAM_mat = (CAM_map == (c+1))
        CAM_mat = CAM_mat.astype(int)
        cclab = CC_labeling_8.CC_lab(CAM_mat)
        cclab.connectedComponentLabel()
        # for (x,y), value in np.ndenumerate(CAM_mat):
        #     if value == 1:
        #         CAM_mat[x, y] = cclab.labels[x][y]
        CAM_mat = CAM_mat * np.array(cclab.labels)

        mat = (label_map == (c+1))
        mat = mat.astype(int)
        cclab = CC_labeling_8.CC_lab(mat)
        cclab.connectedComponentLabel()
        connected_component_label = np.array(cclab.labels)
        propagate_dict = dict()
        for (x,y), value in np.ndenumerate(mat):
            if value == 1 and seed_c[c, x, y] == 1:
                key = connected_component_label[x, y]
                if not key in propagate_dict:
                    propagate_dict[key] = set()
                propagate_dict[key].add(CAM_mat[x, y])
            elif value == 1 and np.sum(seed_c[:, x, y]) == 1:
                connected_component_label[x, y] = -1

        for k,v in propagate_dict.iteritems():
            if len(v) > 1:
                pos = np.where(connected_component_label == k)
                seed_c[c, pos[0], pos[1]] = 1

    return seed_c

class UpdateSeedLayer4(caffe.Layer):

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
        # mean_pixel = np.array([104.0, 117.0, 123.0])
        # im = zoom(im, (1.0, 1.0, 41.0 / im.shape[2], 41.0 / im.shape[3]), order=1)
        # im = np.transpose(im, [0, 2, 3, 1])
        # im = im + mean_pixel[None, None, None, :]
        # im = np.round(im)

        # markers_new = np.zeros((num, height, width))
        # markers_new.fill(255)
        # pos = np.where(seed_c == 1)
        # markers_new[pos[0], pos[2], pos[3]] = pos[1]

        # markers_old = np.zeros((num, height, width))
        # markers_old.fill(255)
        # pos = np.where(cues == 1)
        # markers_old[pos[0], pos[2], pos[3]] = pos[1]

        # self.show_result(im[0], markers_old[0], markers_new[0])

        top[0].data[...] = seed_c
        


    def backward(self, top, prop_down, bottom): 
        pass

    def refinement(self, probs, im, scale_factor=12.0):
        probs[probs < min_prob] = min_prob
        unary = np.transpose(np.array(probs), [0, 2, 3, 1])

        mean_pixel = np.array([104.0, 117.0, 123.0])
        im = zoom(im, (1.0, 1.0, 41.0 / im.shape[2], 41.0 / im.shape[3]), order=1)
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

        seed_c = np.zeros_like(probs)
        seed_c[...] = cues

        items_for_map = [[labels[batch_id, 0, 0], seed_c[batch_id], probs_refinement[batch_id], self._th1, self._th2] for batch_id in xrange(num)]
        seed_c_all = self.pool.map(generate_seed_step4, items_for_map)

        return np.array(seed_c_all)

class RegionSelectionWithLossLayer(caffe.Layer):

    def setup(self, bottom, top):

        if len(bottom) != 2:
            raise Exception("The layer needs two inputs!")

        self.pool = Pool()

        probs = T.ftensor4()
        masks = T.ftensor4()
        stat_inp = T.ftensor4()

        stat = stat_inp[:, :, :, 1:]
        r = T.addbroadcast(theano.shared(R))

        probs_fg = probs[:, 1:, :, :]
        masks_fg = masks[:, 1:, :, :]
        probs_bg = probs[:,  0, :, :]
        masks_bg = masks[:,  0, :, :]

        Z_fg = T.sum(masks_fg, axis=(2, 3), keepdims=True)
        Z_bg = T.sum(masks_bg, axis=(1, 2), keepdims=True)

        S_f = T.log(T.sum(masks_fg * T.exp(r * probs_fg), axis=(2, 3), keepdims=True) / Z_fg) / r
        S_b = T.log(T.sum(masks_bg * T.exp(r * probs_bg), axis=(1, 2), keepdims=True) / Z_bg) / r

        stat_2d = stat[:, 0, 0, :] > 0.5

        loss_1 = -T.mean(T.sum((stat_2d * T.log(S_f) / T.sum(stat_2d, axis=1, keepdims=True)), axis=1))
        loss_2 = -T.mean(T.sum(((1 - stat_2d) * T.log(1 - S_f) / T.sum(1 - stat_2d, axis=1, keepdims=True)), axis=1))
        loss_3 = -T.mean(T.log(S_b))

        loss = loss_1 + loss_2 + loss_3

        self.forward_theano = theano.function([probs, masks, stat_inp], loss)
        self.backward_theano = theano.function([probs, masks, stat_inp], T.grad(loss, probs))

    def reshape(self, bottom, top):
        top[0].reshape(1)

    def forward(self, bottom, top):
        self.masks = self.regionSelection(bottom[0].data)
        # num, channels, height, width = self.masks.shape
        # markers_old = np.zeros((num, height, width))
        # markers_old.fill(255)
        # pos = np.where(self.masks == 1)
        # markers_old[pos[0], pos[2], pos[3]] = pos[1]
        # self.show_result(markers_old[0])

        top[0].data[...] = self.forward_theano(bottom[0].data[...], self.masks, bottom[1].data[...])

    def backward(self, top, prop_down, bottom):
        grad = self.backward_theano(bottom[0].data[...], self.masks, bottom[1].data[...])
        bottom[0].diff[...] = grad

    def regionSelection(self, probs):
        masks = self.pool.map(generate_mask, probs)
        return np.array(masks, dtype=np.float32)

    def show_result(self, seed):
        from matplotlib import pyplot as plt
        from matplotlib import colors as mpl_colors
        palette = [(0.0, 0.0, 0.0), (0.5, 0.0, 0.0), (0.0, 0.5, 0.0), (0.5, 0.5, 0.0),
               (0.0, 0.0, 0.5), (0.5, 0.0, 0.5), (0.0, 0.5, 0.5), (0.5, 0.5, 0.5),
               (0.25, 0.0, 0.0), (0.75, 0.0, 0.0), (0.25, 0.5, 0.0), (0.75, 0.5, 0.0),
               (0.25, 0.0, 0.5), (0.75, 0.0, 0.5), (0.25, 0.5, 0.5), (0.75, 0.5, 0.5),
               (0.0, 0.25, 0.0), (0.5, 0.25, 0.0), (0.0, 0.75, 0.0), (0.5, 0.75, 0.0),
               (0.0, 0.25, 0.5)]
        my_cmap = mpl_colors.LinearSegmentedColormap.from_list('Custom cmap', palette, 21)

        fig = plt.figure()

        ax = fig.add_subplot('121')
        ax.set_title('mask')
        ax.matshow(seed, vmin=0, vmax=21, cmap=my_cmap)

        plt.show()

class AnnotationLayer(caffe.Layer):

    def setup(self, bottom, top):
        if len(bottom) != 1:
            raise Exception("The layer needs two inputs!")

        # parse the layer parameter string, which must be valid YAML
        layer_params = yaml.load(self.param_str)
        if 'cues' not in layer_params:
            layer_params['cues'] = 'localization_cues.pickle'
        self._cue_name = layer_params['cues']

        this_dir = osp.dirname(__file__)
        self.data_file = cPickle.load(open(osp.join(this_dir, '../../training', 'localization_cues', self._cue_name)))

    def reshape(self, bottom, top):
        top[0].reshape(bottom[0].data.shape[0], 1, 1, 21)
        top[1].reshape(bottom[0].data.shape[0], 21, 41, 41)

    def forward(self, bottom, top):

        top[0].data[...] = 0.0
        top[1].data[...] = 0.0

        for i, image_id in enumerate(bottom[0].data[...]):

            labels_i = self.data_file['%i_labels' % image_id]
            top[0].data[i, 0, 0, 0] = 1.0
            top[0].data[i, 0, 0, labels_i] = 1.0

            cues_i = self.data_file['%i_cues' % image_id]
            top[1].data[i, cues_i[0], cues_i[1], cues_i[2]] = 1.0


class AnnotationLayer2(caffe.Layer):

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
        top[3].reshape(bottom[0].data.shape[0], 1, 1, 20)

    def forward(self, bottom, top):

        top[0].data[...] = 0.0
        top[1].data[...] = 0.0
        top[2].data[...] = bottom[1].data
        top[3].data[...] = 0.0

        for i, image_id in enumerate(bottom[0].data[...]):

            labels_i = self.data_file['%i_labels' % image_id]
            top[0].data[i, 0, 0, 0] = 1.0
            top[0].data[i, 0, 0, labels_i] = 1.0
            top[3].data[i, 0, 0, labels_i - 1] = 1.0

            cues_i = self.data_file['%i_cues' % image_id]
            top[1].data[i, cues_i[0], cues_i[1], cues_i[2]] = 1.0

            if self.is_mirror:
                flip = np.random.choice(2) * 2 - 1
                top[1].data[i, ...] = top[1].data[i, :, :, ::flip]
                top[2].data[i, ...] = top[2].data[i, :, :, ::flip]

class AnnotationLayer3(caffe.Layer):

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
