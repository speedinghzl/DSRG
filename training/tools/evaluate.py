import os, sys
import numpy as np
import argparse
import cv2

from multiprocessing import Pool 
import copy_reg
import types
def _pickle_method(m):
    if m.im_self is None:
        return getattr, (m.im_class, m.im_func.func_name)
    else:
        return getattr, (m.im_self, m.im_func.func_name)

copy_reg.pickle(types.MethodType, _pickle_method)

class ConfusionMatrix(object):

    def __init__(self, nclass, classes=None):
        self.nclass = nclass
        self.classes = classes
        self.M = np.zeros((nclass, nclass))

    def add(self, gt, pred):
        assert(np.max(pred) <= self.nclass)
        assert(len(gt) == len(pred))
        for i in range(len(gt)):
            if not gt[i] == 255:
                self.M[gt[i], pred[i]] += 1.0

    def addM(self, matrix):
        assert(matrix.shape == self.M.shape)
        self.M += matrix

    def __str__(self):
        pass

    def recall(self):
        recall = 0.0
        for i in xrange(self.nclass):
            recall += self.M[i, i] / np.sum(self.M[:, i])

        return recall/self.nclass

    def accuracy(self):
        accuracy = 0.0
        for i in xrange(self.nclass):
            accuracy += self.M[i, i] / np.sum(self.M[i, :])

        return accuracy/self.nclass

    def jaccard(self):
        jaccard = 0.0
        jaccard_perclass = []
        for i in xrange(self.nclass):
            if not self.M[i, i] == 0:
                jaccard_perclass.append(self.M[i, i] / (np.sum(self.M[i, :]) + np.sum(self.M[:, i]) - self.M[i, i]))

        return np.sum(jaccard_perclass)/len(jaccard_perclass), jaccard_perclass, self.M

    def generateM(self, item):
        gt, pred = item
        m = np.zeros((self.nclass, self.nclass))
        assert(len(gt) == len(pred))
        for i in range(len(gt)):
            if gt[i] < self.nclass: #and pred[i] < self.nclass:
                m[gt[i], pred[i]] += 1.0
        return m


def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='evaluate segmentation result')
    parser.add_argument('--pred', dest='pred_dir',
                        help='prediction result dir',
                        default=None, type=str)
    parser.add_argument('--class_num', dest='class_num',
                        help='class number include bg',
                        default=7, type=int)
    parser.add_argument('--gt', dest='gt_dir',
                        help='ground truth dir',
                        default='../../dataset/data/part_mask/', type=str)
    parser.add_argument('--test_ids', dest='test_ids',
                        help='test ids file path',
                        default='list/test_id.txt', type=str)
    parser.add_argument('--save_path', dest='save_path',
                        help='result file path',
                        default='result/test_id.txt', type=str)
      

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()
    return args


def show_all(gt, pred):
    import matplotlib.pyplot as plt
    from matplotlib import colors
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    fig, axes = plt.subplots(1, 2)
    ax1, ax2 = axes

    classes = np.array(('background',  # always index 0
               'aeroplane', 'bicycle', 'bird', 'boat',
               'bottle', 'bus', 'car', 'cat', 'chair',
                         'cow', 'diningtable', 'dog', 'horse',
                         'motorbike', 'person', 'pottedplant',
                         'sheep', 'sofa', 'train', 'tvmonitor'))
    colormap = [(0,0,0),(0.5,0,0),(0,0.5,0),(0.5,0.5,0),(0,0,0.5),(0.5,0,0.5),(0,0.5,0.5), 
                    (0.5,0.5,0.5),(0.25,0,0),(0.75,0,0),(0.25,0.5,0),(0.75,0.5,0),(0.25,0,0.5), 
                    (0.75,0,0.5),(0.25,0.5,0.5),(0.75,0.5,0.5),(0,0.25,0),(0.5,0.25,0),(0,0.75,0), 
                    (0.5,0.75,0),(0,0.25,0.5)]
    cmap = colors.ListedColormap(colormap)
    bounds=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21]
    norm = colors.BoundaryNorm(bounds, cmap.N)

    ax1.set_title('gt')
    ax1.imshow(gt, cmap=cmap, norm=norm)

    ax2.set_title('pred')
    ax2.imshow(pred, cmap=cmap, norm=norm)

    plt.show()


if __name__ == '__main__':
    args = parse_args()

    m_list = []
    data_list = []
    test_ids = [i.strip() for i in open(args.test_ids) if not i.strip() == '']
    for index, img_id in enumerate(test_ids):
        if index % 100 == 0:
            print('%d processd'%(index))
        pred_img_path = os.path.join(args.pred_dir, img_id+'.png')
        gt_img_path = os.path.join(args.gt_dir, img_id+'.png')
        pred = cv2.imread(pred_img_path, cv2.IMREAD_GRAYSCALE)
        gt = cv2.imread(gt_img_path, cv2.IMREAD_GRAYSCALE)
        # show_all(gt, pred)
        data_list.append([gt.flatten(), pred.flatten()])

    ConfM = ConfusionMatrix(args.class_num)
    f = ConfM.generateM
    pool = Pool() 
    m_list = pool.map(f, data_list)
    pool.close() 
    pool.join() 
    
    for m in m_list:
        ConfM.addM(m)

    aveJ, j_list, M = ConfM.jaccard()
    with open(args.save_path, 'w') as f:
        f.write('meanIOU: ' + str(aveJ) + '\n')
        f.write(str(j_list)+'\n')
        f.write(str(M)+'\n')
